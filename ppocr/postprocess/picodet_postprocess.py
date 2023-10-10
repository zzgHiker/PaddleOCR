# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from scipy.special import softmax


def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    """
    Args:
        box_scores (N, 5): boxes in corner-form and probabilities.
        iou_threshold: intersection over union threshold.
        top_k: keep top_k results. If k <= 0, keep all the results.
        candidate_size: only consider the candidates with the highest scores.
    Returns:
         picked: a list of indexes of the kept boxes
    """
    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked = []
    indexes = np.argsort(scores)
    indexes = indexes[-candidate_size:]
    while len(indexes) > 0:
        current = indexes[-1]
        picked.append(current)
        if 0 < top_k == len(picked) or len(indexes) == 1:
            break
        current_box = boxes[current, :]
        indexes = indexes[:-1]
        rest_boxes = boxes[indexes, :]
        iou = iou_of(
            rest_boxes,
            np.expand_dims(
                current_box, axis=0), )
        indexes = indexes[iou <= iou_threshold]

    return box_scores[picked, :]


def iou_of(boxes0, boxes1, eps=1e-5):
    """Return intersection-over-union (Jaccard index) of boxes.
    Args:
        boxes0 (N, 4): ground truth boxes.
        boxes1 (N or 1, 4): predicted boxes.
        eps: a small number to avoid 0 as denominator.
    Returns:
        iou (N): IoU values.
    """
    overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)


def area_of(left_top, right_bottom):
    """Compute the areas of rectangles given two corners.
    Args:
        left_top (N, 2): left top corner.
        right_bottom (N, 2): right bottom corner.
    Returns:
        area (N): return the area.
    """
    hw = np.clip(right_bottom - left_top, 0.0, None)
    return hw[..., 0] * hw[..., 1]


def is_union(rect1, rect2):
    # 计算两个矩形的交集
    x1 = max(rect1[0], rect2[0])
    y1 = max(rect1[1], rect2[1])
    x2 = min(rect1[2], rect2[2])
    y2 = min(rect1[3], rect2[3])

    # 计算交集的面积
    intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

    # 计算两个矩形的并集的面积
    area1 = (rect1[2] - rect1[0] + 1) * (rect1[3] - rect1[1] + 1)
    area2 = (rect2[2] - rect2[0] + 1) * (rect2[3] - rect2[1] + 1)
    union_area = area1 + area2 - intersection_area

    # 计算IoU
    iou = intersection_area / union_area
    print("iou", iou)
    return iou > 0


def merge_union_boxes(boxes):
    """将相连Box合并起来"""

    def merge_boxes(rect1, rect2):
        """计算合并后的矩形坐标"""
        x1 = min(rect1[0], rect2[0])
        y1 = min(rect1[1], rect2[1])
        x2 = max(rect1[2], rect2[2])
        y2 = max(rect1[3], rect2[3])

        return [x1, y1, x2, y2]

    # 合并后的Box集合
    union_boxes = []
    # 已处理的
    done_indexes = []
    for i, box1 in enumerate(boxes[:-1]):
        if i in done_indexes:
            # 已处理，跳过
            continue

        # 将相连的Box合并成新的box
        union_box = box1
        for j in range(i + 1, len(boxes)):
            box2 = boxes[j]
            if is_union(union_box['bbox'], box2['bbox']):
                # 相连合并
                done_indexes.append(j)
                union_box['bbox'] = merge_boxes(union_box['bbox'], box2['bbox'])

        done_indexes.append(i)
        union_boxes.append(union_box)

    return union_boxes


class PicoDetPostProcess(object):
    """
    Args:
        input_shape (int): network input image size
        ori_shape (int): ori image shape of before padding
        scale_factor (float): scale factor of ori image
        enable_mkldnn (bool): whether to open MKLDNN
    """

    def __init__(self,
                 layout_dict_path,
                 strides=[8, 16, 32, 64],
                 score_threshold=0.4,
                 nms_threshold=0.5,
                 nms_top_k=1000,
                 keep_top_k=100):
        self.labels = self.load_layout_dict(layout_dict_path)
        self.strides = strides
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.nms_top_k = nms_top_k
        self.keep_top_k = keep_top_k

    def load_layout_dict(self, layout_dict_path):
        with open(layout_dict_path, 'r', encoding='utf-8') as fp:
            labels = fp.readlines()
        return [label.strip('\n') for label in labels]

    def warp_boxes(self, boxes, ori_shape):
        """Apply transform to boxes
        """
        width, height = ori_shape[1], ori_shape[0]
        n = len(boxes)
        if n:
            # warp points
            xy = np.ones((n * 4, 3))
            xy[:, :2] = boxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(
                n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            # xy = xy @ M.T  # transform
            xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)  # rescale
            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            xy = np.concatenate(
                (x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T
            # clip boxes
            xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
            xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
            return xy.astype(np.float32)
        else:
            return boxes

    def img_info(self, ori_img, img):
        origin_shape = ori_img.shape
        resize_shape = img.shape
        im_scale_y = resize_shape[2] / float(origin_shape[0])
        im_scale_x = resize_shape[3] / float(origin_shape[1])
        scale_factor = np.array([im_scale_y, im_scale_x], dtype=np.float32)
        img_shape = np.array(img.shape[2:], dtype=np.float32)

        input_shape = np.array(img).astype('float32').shape[2:]
        ori_shape = np.array((img_shape,)).astype('float32')
        scale_factor = np.array((scale_factor,)).astype('float32')
        return ori_shape, input_shape, scale_factor

    def __call__(self, ori_img, img, preds):
        scores, raw_boxes = preds['boxes'], preds['boxes_num']
        batch_size = raw_boxes[0].shape[0]
        reg_max = int(raw_boxes[0].shape[-1] / 4 - 1)
        out_boxes_num = []
        out_boxes_list = []
        ori_shape, input_shape, scale_factor = self.img_info(ori_img, img)

        for batch_id in range(batch_size):
            # generate centers
            decode_boxes = []
            select_scores = []
            for stride, box_distribute, score in zip(self.strides, raw_boxes,
                                                     scores):
                box_distribute = box_distribute[batch_id]
                score = score[batch_id]
                # centers
                fm_h = input_shape[0] / stride
                fm_w = input_shape[1] / stride
                h_range = np.arange(fm_h)
                w_range = np.arange(fm_w)
                ww, hh = np.meshgrid(w_range, h_range)
                ct_row = (hh.flatten() + 0.5) * stride
                ct_col = (ww.flatten() + 0.5) * stride
                center = np.stack((ct_col, ct_row, ct_col, ct_row), axis=1)

                # box distribution to distance
                reg_range = np.arange(reg_max + 1)
                box_distance = box_distribute.reshape((-1, reg_max + 1))
                box_distance = softmax(box_distance, axis=1)
                box_distance = box_distance * np.expand_dims(reg_range, axis=0)
                box_distance = np.sum(box_distance, axis=1).reshape((-1, 4))
                box_distance = box_distance * stride

                # top K candidate
                topk_idx = np.argsort(score.max(axis=1))[::-1]
                topk_idx = topk_idx[:self.nms_top_k]
                center = center[topk_idx]
                score = score[topk_idx]
                box_distance = box_distance[topk_idx]

                # decode box
                decode_box = center + [-1, -1, 1, 1] * box_distance

                select_scores.append(score)
                decode_boxes.append(decode_box)

            # nms
            bboxes = np.concatenate(decode_boxes, axis=0)
            confidences = np.concatenate(select_scores, axis=0)
            picked_box_probs = []
            picked_labels = []
            for class_index in range(0, confidences.shape[1]):
                probs = confidences[:, class_index]
                mask = probs > self.score_threshold
                probs = probs[mask]
                if probs.shape[0] == 0:
                    continue
                subset_boxes = bboxes[mask, :]
                box_probs = np.concatenate(
                    [subset_boxes, probs.reshape(-1, 1)], axis=1)
                box_probs = hard_nms(
                    box_probs,
                    iou_threshold=self.nms_threshold,
                    top_k=self.keep_top_k, )
                picked_box_probs.append(box_probs)
                picked_labels.extend([class_index] * box_probs.shape[0])

            if len(picked_box_probs) == 0:
                out_boxes_list.append(np.empty((0, 4)))
                out_boxes_num.append(0)

            else:
                picked_box_probs = np.concatenate(picked_box_probs)

                # resize output boxes
                picked_box_probs[:, :4] = self.warp_boxes(
                    picked_box_probs[:, :4], ori_shape[batch_id])
                im_scale = np.concatenate([
                    scale_factor[batch_id][::-1], scale_factor[batch_id][::-1]
                ])
                picked_box_probs[:, :4] /= im_scale
                # clas score box
                out_boxes_list.append(
                    np.concatenate(
                        [
                            np.expand_dims(
                                np.array(picked_labels),
                                axis=-1), np.expand_dims(
                            picked_box_probs[:, 4], axis=-1),
                            picked_box_probs[:, :4]
                        ],
                        axis=1))
                out_boxes_num.append(len(picked_labels))

        out_boxes_list = np.concatenate(out_boxes_list, axis=0)
        out_boxes_num = np.asarray(out_boxes_num).astype(np.int32)

        regions = []
        for dt in out_boxes_list:
            clsid, bbox, score = int(dt[0]), dt[2:], dt[1]
            label = self.labels[clsid]
            region = {'bbox': bbox, 'label': label}
            regions.append(region)

        # 合并相连区域
        regions = sorted(regions, key=lambda it: it["bbox"][1])

        return merge_union_boxes(regions)
