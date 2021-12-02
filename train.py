import argparse
import bisect
import copy
import os
import time

import megengine as mge
import megengine.distributed as dist
from megengine.autodiff import GradManager
from megengine.data import DataLoader, Infinite, RandomSampler
from megengine.data import transform as T
from megengine.data.dataset import COCO
from megengine.optimizer import SGD

from CrowdHuman import CrowdHuman
from dataset import DetectionPadCollator, GroupedRandomSampler
from model import build_network
from utils import AverageMeter, get_config_info

logger = mge.logger.get_logger()
logger.setLevel("INFO")


data_mapper = dict(
    coco=COCO,
    crowdhuman=CrowdHuman
)


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--mf", default=True, type=bool, help="whether use 3D MaxFilter",
    )
    parser.add_argument(
        "--gn", default=True, type=bool, help="3D MaxFilter with/without GroupNorm"
    )
    parser.add_argument(
        "--aux", default=True, type=bool, help="whether use aux loss"
    )
    parser.add_argument(
        "--dataset_name", default='coco', type=str, help="The dataset name"
    )
    parser.add_argument(
        "-w", "--weight_file", default=None, type=str, help="weights file",
    )
    parser.add_argument(
        "-n", "--devices", default=1, type=int, help="total number of gpus for training",
    )
    parser.add_argument(
        "-b", "--batch_size", default=2, type=int, help="batch size for training",
    )
    parser.add_argument(
        "-d", "--dataset_dir", default="/data/datasets", type=str,
    )
    parser.add_argument(
        "-j", "--workers", default=4, type=int,
    )
    parser.add_argument(
        "-s", "--save_path", default='./ckpt', type=str,
    )
    parser.add_argument(
        "--tag",
        default="test",
        type=str,
        help='the tag for identifying the log and model files. Just a string.'
    )
    parser.add_argument(
        "--resume",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--resume_epoch",
        default=0,
        type=int,
    )
    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()
    print("Start trainning!!!!")
    # ------------------------ begin training -------------------------- #
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
        print(args.save_path)

    logger.info("Device Count = %d", args.devices)
    if args.devices > 1:
        trainer = dist.launcher(worker, n_gpus=args.devices)
        trainer(args)
    else:
        worker(args)


def worker(args):
    print("bulid network!!!")
    model_cfg, model = build_network(
        args.mf, args.gn, args.aux, args.dataset_name, dist.get_rank(), args.resume)
    model.train()

    if dist.get_rank() == 0:
        mge.logger.set_log_file(
            os.path.join(args.save_path, args.tag + 'log.txt'))
        logger.info(get_config_info(model.cfg))
        logger.info(repr(model))
    print(f"freeze backbone at {model.cfg.backbone_freeze_at}")
    params_with_grad = []
    for name, param in model.named_parameters():
        if "bottom_up.conv1" in name and model.cfg.backbone_freeze_at >= 1:
            continue
        if "bottom_up.layer1" in name and model.cfg.backbone_freeze_at >= 2:
            continue
        params_with_grad.append(param)

    print("")
    opt = SGD(
        params_with_grad,
        lr=model.cfg.basic_lr * args.batch_size * dist.get_world_size(),
        momentum=model.cfg.momentum,
        weight_decay=model.cfg.weight_decay,
    )

    gm = GradManager()
    if dist.get_world_size() > 1:
        gm.attach(
            params_with_grad,
            callbacks=[dist.make_allreduce_cb("mean", dist.WORLD)]
        )
    else:
        gm.attach(params_with_grad)

    if args.weight_file is not None:
        weights = mge.load(args.weight_file)
        model.backbone.bottom_up.load_state_dict(weights, strict=False)
    if args.resume and dist.get_rank() == 0:
        checkpoint = mge.load(args.resume)
        if 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
        model.load_state_dict(checkpoint)
        print(f"load {args.resume}")

    if dist.get_world_size() > 1:
        dist.bcast_list_(model.parameters())  # sync parameters
        dist.bcast_list_(model.buffers())  # sync buffers

    if dist.get_rank() == 0:
        logger.info("Prepare dataset")
    train_loader = iter(build_dataloader(args, model.cfg))
    for epoch in range(args.resume_epoch, model.cfg.max_epoch):
        train_one_epoch(model, train_loader, opt, gm, epoch, args)
        if dist.get_rank() == 0:
            mge.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict()
                },
                args.save_path + f"epoch_{epoch}.pkl"
            )
            logger.info("dump weights to %s", args.save_path)


def train_one_epoch(model, data_queue, opt, gm, epoch, args):
    def train_func(images, im_info, gt_boxes):
        with gm:
            loss_dict = model(images=images, im_info=im_info,
                              gt_boxes=gt_boxes)
            gm.backward(loss_dict["total_loss"])
            loss_list = list(loss_dict.values())
        opt.step().clear_grad()
        return loss_list

    meter = AverageMeter(record_len=model.cfg.num_losses)
    time_meter = AverageMeter(record_len=2)
    log_interval = model.cfg.log_interval
    tot_step = model.cfg.nr_images_epoch // (
        args.batch_size * dist.get_world_size())
    for step in range(tot_step):
        adjust_learning_rate(opt, epoch, step, model.cfg, args)

        data_tik = time.time()
        mini_batch = next(data_queue)
        data_tok = time.time()

        tik = time.time()

        loss_list = train_func(
            images=mge.tensor(mini_batch["data"]),
            im_info=mge.tensor(mini_batch["im_info"]),
            gt_boxes=mge.tensor(mini_batch["gt_boxes"])
        )
        tok = time.time()

        time_meter.update([tok - tik, data_tok - data_tik])

        if dist.get_rank() == 0:
            info_str = "e%d, %d/%d, lr:%f, "
            loss_str = ", ".join(
                ["{}:%f".format(loss) for loss in model.cfg.losses_keys]
            )
            time_str = ", train_time:%.3fs, data_time:%.3fs"
            log_info_str = info_str + loss_str + time_str
            meter.update([loss.numpy() for loss in loss_list])
            if step % log_interval == 0:
                logger.info(
                    log_info_str,
                    epoch,
                    step,
                    tot_step,
                    opt.param_groups[0]["lr"],
                    *meter.average(),
                    *time_meter.average()
                )
                meter.reset()
                time_meter.reset()


def adjust_learning_rate(optimizer, epoch, step, cfg, args):
    base_lr = (
        cfg.basic_lr * args.batch_size * dist.get_world_size() * (
            cfg.lr_decay_rate
            ** bisect.bisect_right(cfg.lr_decay_stages, epoch)
        )
    )
    # Warm up
    lr_factor = 1.0
    if epoch == 0 and step < cfg.warm_iters:
        lr_factor = (step + 1.0) / cfg.warm_iters
    for param_group in optimizer.param_groups:
        param_group["lr"] = base_lr * lr_factor


def build_dataset(dataset_dir, cfg):
    data_cfg = copy.deepcopy(cfg.train_dataset)
    data_name = data_cfg.pop("name")

    data_cfg["root"] = os.path.join(dataset_dir, data_name, data_cfg["root"])

    if "ann_file" in data_cfg:
        data_cfg["ann_file"] = os.path.join(
            dataset_dir, data_name, data_cfg["ann_file"])

    data_cfg["order"] = ["image", "boxes", "boxes_category", "info"]

    return data_mapper[data_name](**data_cfg)


# pylint: disable=dangerous-default-value
def build_sampler(train_dataset, batch_size, aspect_grouping=[1]):
    def _compute_aspect_ratios(dataset):
        aspect_ratios = []
        for i in range(len(dataset)):
            info = dataset.get_img_info(i)
            aspect_ratios.append(info["height"] / info["width"])
        return aspect_ratios

    def _quantize(x, bins):
        return list(map(lambda y: bisect.bisect_right(sorted(bins), y), x))

    if len(aspect_grouping) == 0:
        return Infinite(RandomSampler(train_dataset, batch_size, drop_last=True))

    aspect_ratios = _compute_aspect_ratios(train_dataset)
    group_ids = _quantize(aspect_ratios, aspect_grouping)
    return Infinite(GroupedRandomSampler(train_dataset, batch_size, group_ids))


def build_dataloader(args, cfg):
    train_dataset = build_dataset(args.dataset_dir, cfg)
    train_sampler = build_sampler(train_dataset, args.batch_size)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        transform=T.Compose(
            transforms=[
                T.ShortestEdgeResize(
                    cfg.train_image_short_size,
                    cfg.train_image_max_size,
                    sample_style="choice",
                ),
                T.RandomHorizontalFlip(),
                T.ToMode(),
            ],
            order=["image", "boxes", "boxes_category"],
        ),
        collator=DetectionPadCollator(),
        num_workers=args.workers,
    )
    return train_dataloader


if __name__ == "__main__":
    main()
