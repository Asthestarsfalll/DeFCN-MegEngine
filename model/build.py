from .config import DeFCNConfig
from .defcn import DeFCN


def build_network(use_3dmf=True, use_gn=True, use_auxloss=True,
                  dataset_name='coco', rank=0, resume=None):
    print("build base_config")
    base_config = DeFCNConfig()
    base_config.use_3dmf = use_3dmf
    base_config.use_gn = use_gn
    base_config.aux_gt = use_auxloss
    if rank != 0 or resume is not None:
        base_config.backbone_pretrained = False

    if dataset_name == "crowdhuman":
        base_config.num_classes = 1
        base_config.train_dataset = dict(
            name="crowdhuman",
            root="Images",
            ann_file="annotation_train.json",
        )
        base_config.test_dataset = dict(
            name="crowdhuman",
            root="Images",
            ann_file="annotation_val.json"
        )
        base_config.center_sampling_radius = 0.0  # inside gt box
        base_config.train_image_short_size = (800,)
        base_config.train_image_max_size = 1400
        base_config.test_image_max_size = 1400
        base_config.max_epoch = 6
        base_config.lr_decay_stages = [4, 5]
    print("build DeFCN")
    net = DeFCN(base_config)
    return base_config, net
