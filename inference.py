# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import argparse

import cv2
import megengine as mge
from megengine.data.dataset import COCO

from CrowdHuman import CrowdHuman
from dataset import DetEvaluator
from model import build_network

logger = mge.get_logger(__name__)
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
    parser.add_argument("-i", "--image", type=str)
    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()
    model_cfg, model = build_network(
        args.mf, args.gn, args.aux, args.dataset_name, 1)
    model.eval()

    state_dict = mge.load(args.weight_file)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict)

    evaluator = DetEvaluator(model)

    ori_img = cv2.imread(args.image)
    image, im_info = DetEvaluator.process_inputs(
        ori_img.copy(), model.cfg.test_image_short_size, model.cfg.test_image_max_size,
    )
    pred_res = evaluator.predict(
        image=mge.tensor(image),
        im_info=mge.tensor(im_info)
    )
    res_img = DetEvaluator.vis_det(
        ori_img,
        pred_res,
        is_show_label=True,
        classes=data_mapper[args.dataset_name].class_names,
    )
    cv2.imwrite("./results.jpg", res_img)


if __name__ == "__main__":
    main()
