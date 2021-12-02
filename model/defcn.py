import megengine as mge
import megengine.functional as F
import numpy as np
from megengine import hub
from scipy.optimize import linear_sum_assignment

from .head import DeFCNHead
from .utils import (all_reduce_mean, cat, get_clipped_boxes,
                    get_padded_tensor, permute_to_N_HWA_K)
from .loss import focal_loss, iou_loss, pairwise_iou

modelhub = hub.import_module(
    repo_info='megengine/models', git_host='github.com')


class DeFCN(modelhub.FCOS):
    """
    Modify from here(https://github.com/MegEngine/Models/blob/master/official/vision/detection/models/fcos.py)
    """

    def __init__(self, cfg):
        super(DeFCN, self).__init__(cfg)
        print(f"start init, {cfg.backbone_pretrained}")
        self.aux_gt = cfg.aux_gt
        # ----------------------- poto ----------------------- #
        self.poto_alpha = cfg.poto_alpha
        self.poto_aux_topk = cfg.poto_aux_topk
        # ----------------------- Head ----------------------- #
        backbone_shape = self.backbone.output_shape()
        feature_shapes = [backbone_shape[f] for f in self.in_features]
        self.head = DeFCNHead(cfg, feature_shapes)

    def preprocess_image(self, image):
        normed_image = (
            image
            - np.array(self.cfg.img_mean, dtype="float32")[None, :, None, None]
        ) / np.array(self.cfg.img_std, dtype="float32")[None, :, None, None]
        padded_image = get_padded_tensor(normed_image, 32, 0.0)
        return padded_image

    def forward(self, images, im_info, gt_boxes=None):
        """
            Args:
            images: images with padding in shape (b ,c ,h_m, w_m) .
                h_m and w_m is the max one in this batch
            im_info: contain current/origin height/width and instance number of each image, shape:(b, 5)
            gt_boxes: ground truth with shape (b, N ,5)
                N is the max instance number in this batch
                gt_boxes[:, :, 4] contains classification of each instance
        """
        images = self.preprocess_image(images)

        features = self.backbone(images)
        # A list contains some level features of fpn
        features = [features[f] for f in self.in_features]

        box_logits, box_offsets, box_filters = self.head(
            features)  # b, 80\4\1, hi,wi

        box_logits = [permute_to_N_HWA_K(
            x, self.cfg.num_classes) for x in box_logits]
        box_offsets = [permute_to_N_HWA_K(x, 4) for x in box_offsets]
        box_filters = [permute_to_N_HWA_K(x, 1) for x in box_filters]
        # list[Tensor], one per level, each has shape (N, Hi x Wi x A, K or 4)

        # shared anchors across batch with shape (N, 2)
        anchors_list = self.anchor_generator(features)

        if self.training:
            box_logits = F.concat(box_logits, axis=1)
            box_offsets = F.concat(box_offsets, axis=1)
            box_filters = F.concat(box_filters, axis=1)

            gt_labels, gt_offsets = self.get_ground_truth(
                anchors_list, gt_boxes, im_info[:, 4].astype(np.int32),
                box_logits, box_offsets, box_filters
            )

            losses = self.losses(gt_labels, gt_offsets,
                                 box_logits, box_offsets, box_filters)
            if self.aux_gt:
                gt_classes = self.get_aux_ground_truth(
                    anchors_list, gt_boxes, im_info[:, 4].astype(np.int32), box_logits, box_offsets)
                losses.update(self.aux_losses(gt_classes, box_logits))
            losses['total_loss'] = losses['loss_cls'] + \
                losses['loss_bbox'] + losses['loss_cls_aux']
            self.cfg.losses_keys = list(losses.keys())
            return losses
        else:
            return self.inference(
                box_logits, box_offsets, box_filters, anchors_list, im_info[0])

    def get_ground_truth(self, anchors_list, batched_gt_boxes, batched_num_gts, box_logits, box_offsets, box_filters):
        box_logits = box_logits.detach()  # b, N, 80
        box_offsets = box_offsets.detach()  # b, N, 4
        box_filters = box_filters.detach()  # b, N, 1
        box_logits = (F.sigmoid(box_logits) * F.sigmoid(box_filters)).detach()
        del box_filters

        gt_logits = []
        gt_anchor_offsets = []
        all_level_anchors = F.concat(anchors_list, axis=0)

        for targets_per_image, box_logits_per_image, box_offsets_per_image, num_instence in zip(
                batched_gt_boxes, box_logits, box_offsets, batched_num_gts):
            gt_boxes = targets_per_image[:num_instence]
            # contain instance cls_id pre-image
            logist_idx = mge.tensor([int(x)
                                    for x in gt_boxes[:num_instence, 4]])

            prob = box_logits_per_image[:, logist_idx].T

            boxes = self.point_coder.decode(
                all_level_anchors, box_offsets_per_image)

            iou = pairwise_iou(gt_boxes[:, :4], boxes)

            quality = prob ** (1 - self.poto_alpha) * iou ** self.poto_alpha

            offsets = self.point_coder.encode(
                all_level_anchors, F.expand_dims(gt_boxes[:, :4], axis=1))

            if self.cfg.center_sampling_radius > 0:
                gt_centers = (gt_boxes[:, :2] + gt_boxes[:, 2:4]) / 2
                is_in_boxes = []
                for stride, anchors_i in zip(self.cfg.stride, anchors_list):
                    radius = stride * self.cfg.center_sampling_radius
                    center_boxes = F.concat([
                        F.maximum(gt_centers - radius, gt_boxes[:, :2]),
                        F.minimum(gt_centers + radius, gt_boxes[:, 2:4]),
                    ], axis=1)
                    center_offsets = self.point_coder.encode(
                        anchors_i, F.expand_dims(center_boxes, axis=1))
                    is_in_boxes.append(F.min(center_offsets, axis=2) > 0)
                is_in_boxes = F.concat(is_in_boxes, axis=1)
            else:
                is_in_boxes = F.min(offsets, axis=2) > 0

            quality[~is_in_boxes] = -1

            gt_idxs, anchor_idxs = linear_sum_assignment(
                quality.numpy(), maximize=True)

            gt_logits_i = F.full((len(all_level_anchors),),
                                 self.cfg.num_classes + 1, dtype="int32")
            gt_anchor_reg_offset_i = F.zeros(
                (len(all_level_anchors), 4), dtype=all_level_anchors.dtype)

            if len(targets_per_image) > 0:
                gt_logits_i[anchor_idxs] = logist_idx[gt_idxs]
                gt_anchor_reg_offset_i[anchor_idxs] = self.point_coder.encode(
                    all_level_anchors[anchor_idxs], gt_boxes[gt_idxs, :4]
                )

            gt_logits.append(gt_logits_i)
            gt_anchor_offsets.append(gt_anchor_reg_offset_i)
        return F.stack(gt_logits, axis=0), F.stack(gt_anchor_offsets, axis=0)

    def get_aux_ground_truth(self, anchors_list, batched_gt_boxes, batched_num_gts, box_logits, box_offsets):
        box_logits = F.sigmoid(box_logits).detach()  # b, N, 80
        box_offsets = box_offsets.detach()  # b, N, 4

        gt_logits = []
        all_level_anchors = F.concat(anchors_list, axis=0)

        for targets_per_image, box_logits_per_image, box_offsets_per_image, num_instence in zip(
                batched_gt_boxes, box_logits, box_offsets, batched_num_gts):

            gt_boxes = targets_per_image[:num_instence]

            logist_idx = mge.tensor([int(x)
                                     for x in gt_boxes[:num_instence, 4]])

            prob = box_logits_per_image[:, logist_idx].T

            boxes = self.point_coder.decode(
                all_level_anchors, box_offsets_per_image)

            iou = pairwise_iou(gt_boxes, boxes)
            quality = prob ** (1 - self.poto_alpha) * iou ** self.poto_alpha

            candidata_idxs = []
            st, ed = 0, 0
            for anchor_i in anchors_list:
                ed += len(anchor_i)
                _, topk_idxs = F.topk(
                    quality[:, st:ed], self.poto_aux_topk, descending=True)
                candidata_idxs.append(st + topk_idxs)
                st = ed
            candidata_idxs = F.concat(candidata_idxs, axis=1)

            is_in_boxes = self.point_coder.encode(
                all_level_anchors, F.expand_dims(gt_boxes[:, :4], axis=1)
            ).min(axis=-1) > 0

            candidata_qua = F.gather(quality, axis=1, index=candidata_idxs)
            quality_thread = candidata_qua.mean(
                axis=1, keepdims=True) + F.std(candidata_qua, axis=1, keepdims=True)
            is_foreground = F.scatter(
                F.zeros_like(is_in_boxes),
                axis=1,
                index=candidata_idxs,
                source=F.ones_like(candidata_idxs).astype(bool))
            is_foreground &= quality >= quality_thread

            quality[~is_in_boxes] = -1
            quality[~is_foreground] = -1

            # if there are still more than one objects for a position,
            # we choose the one with maximum quality
            positions_max_quality = quality.max(axis=0)
            gt_matched_idxs = F.argmax(quality, axis=0)

            # ground truth classes
            if len(targets_per_image) > 0:
                # logist_idx need to be tensor
                gt_logits_i = logist_idx[gt_matched_idxs]
                # Shifts with quality -1 are treated as background.
                gt_logits_i[positions_max_quality == -
                            1] = self.cfg.num_classes + 1
            else:
                gt_logits_i = F.zeros_like(
                    gt_matched_idxs) + self.cfg.num_classes + 1
            gt_logits.append(gt_logits_i)
        return F.stack(gt_logits, axis=0)

    def losses(self, gt_classes, gt_anchor_offsets, pred_logits, pred_offsets, pred_filter):
        pred_logits = pred_logits.reshape(-1, self.cfg.num_classes)  # b, N, 80
        pred_offsets = pred_offsets.reshape(-1, 4)  # b, N, 4
        pred_filter = pred_filter.reshape(-1, 1)  # b, N, 1
        pred_logits = F.sigmoid(pred_logits) * F.sigmoid(pred_filter)
        del pred_filter

        gt_classes = gt_classes.flatten()
        gt_anchor_offsets = gt_anchor_offsets.reshape(-1, 4)

        valid_mask = gt_classes >= 0

        foreground_mask = (gt_classes >= 0) & (
            gt_classes != self.cfg.num_classes + 1)

        # add detach() to avoid syncing across ranks in backward
        num_fg = all_reduce_mean(foreground_mask.sum()).detach()

        gt_targets = F.zeros_like(pred_logits)
        gt_targets[foreground_mask, gt_classes[foreground_mask] - 1] = 1

        # logits loss
        loss_cls = focal_loss(
            pred_logits[valid_mask],
            gt_targets[valid_mask],
            alpha=self.cfg.focal_loss_alpha,
            gamma=self.cfg.focal_loss_gamma,
            with_logits=False) / F.maximum(num_fg, 1)

        # regression loss
        loss_bbox = iou_loss(
            pred_offsets[foreground_mask],
            gt_anchor_offsets[foreground_mask],
            box_mode='ltrb',
            loss_type=self.cfg.iou_loss_type).sum() / F.maximum(1, num_fg) * self.cfg.loss_bbox_weight

        return {
            "loss_cls": loss_cls,
            "loss_bbox": loss_bbox,
        }

    def aux_losses(self, gt_classes, pred_logits):
        pred_logits = pred_logits.reshape(-1, self.cfg.num_classes)
        gt_classes = gt_classes.flatten()

        valid_mask = gt_classes >= 0
        foreground_mask = (gt_classes >= 0) & (
            gt_classes != self.cfg.num_classes + 1)

        num_fg = all_reduce_mean(foreground_mask.sum()).detach()

        gt_targets = F.zeros_like(pred_logits)
        gt_targets[foreground_mask, gt_classes[foreground_mask] - 1] = 1

        # logits loss
        loss_cls_aux = focal_loss(
            F.sigmoid(pred_logits[valid_mask]),
            gt_targets[valid_mask],
            alpha=self.cfg.focal_loss_alpha,
            gamma=self.cfg.focal_loss_gamma,
            with_logits=False) / F.maximum(num_fg, 1)

        return {'loss_cls_aux': loss_cls_aux}

    def inference(self, box_logits, box_offsets, box_filters, anchors_list, im_info):
        boxes_all = []
        scores_all = []
        classes_idxs_all = []

        for box_log_i, box_reg_i, box_fil_i, anchors_i in zip(
                box_logits, box_offsets, box_filters, anchors_list):
            box_log_i = (F.sigmoid(box_fil_i) * F.sigmoid(box_log_i)).flatten()
            del box_fil_i
            box_reg_i = box_reg_i[0]
            num_topk = min(self.cfg.topk_candidates, box_reg_i.shape[0])

            pred_prob, topk_idxs = F.topk(box_log_i, num_topk, descending=True)

            # filter out the proposals with low confidence score
            keep_idxs = pred_prob > self.cfg.test_cls_threshold
            pred_prob = pred_prob[keep_idxs]
            topk_idxs = topk_idxs[keep_idxs]

            anchor_idxs = topk_idxs // self.cfg.num_classes
            classes_idxs = topk_idxs % self.cfg.num_classes

            box_reg_i = box_reg_i[anchor_idxs]
            anchors_i = anchors_i[anchor_idxs]

            # predict boxes
            pred_boxes = self.point_coder.decode(
                anchors_i, box_reg_i).reshape(-1, 4)

            scale_w = im_info[1] / im_info[3]
            scale_h = im_info[0] / im_info[2]
            pred_boxes = pred_boxes / F.concat(
                [scale_w, scale_h, scale_w, scale_h], axis=0)

            clipped_boxes = get_clipped_boxes(
                pred_boxes, im_info[2:4]
            ).reshape(-1, 4)

            scores_all.append(pred_prob)
            boxes_all.append(clipped_boxes)
            classes_idxs_all.append(classes_idxs)

        return [cat(x) for x in [scores_all, boxes_all, classes_idxs_all]]


if __name__ == '__main__':
    pass
