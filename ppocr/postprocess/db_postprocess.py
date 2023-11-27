# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
"""
This code is refered from:
https://github.com/WenmuZhou/DBNet.pytorch/blob/master/post_processing/seg_detector_representer.py
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

import numpy as np
import cv2
import paddle
from shapely.geometry import Polygon
import pyclipper


def cal_area(roi):
    """计算矩形面积"""
    w = roi[2] - roi[0]
    h = roi[3] - roi[1]
    return max(0, w) * max(0, h)


def bbox2rect(rect):
    """
    矩形坐标   [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    转ROI坐标  [x1, y1, x2, y2]
    """
    assert len(rect) in [2, 4]

    if len(rect) == 4:
        return np.array(rect[::2]).flatten()
    elif len(rect) == 2:
        return np.array(rect).flatten()


def rect2bbox(roi):
    assert len(roi) == 4

    x1, y1, x2, y2 = roi
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def is_union_horizontal(bbox_1, bbox_2, thresh=0.5):
    """判断是否水平相连"""
    rect_1 = bbox2rect(bbox_1)
    rect_2 = bbox2rect(bbox_2)

    # 计算两个矩形的交集RECT坐标
    i_x1 = max(rect_1[0], rect_2[0])
    i_y1 = max(rect_1[1], rect_2[1])
    i_x2 = min(rect_1[2], rect_2[2])
    i_y2 = min(rect_1[3], rect_2[3])

    # 计算交集的面积
    intersection_area = cal_area([i_x1, i_y1, i_x2, i_y2])
    v_intersection_area1 = cal_area([i_x1, rect_1[1], i_x2, rect_1[3]])
    v_intersection_area2 = cal_area([i_x1, rect_2[1], i_x2, rect_2[3]])

    # 计算两个矩形的并集的面积
    area1 = cal_area(rect_1)
    area2 = cal_area(rect_2)
    union_area = area1 + area2 - intersection_area

    # 计算交并比（IoU）
    iou = intersection_area / union_area
    # 垂直方向重叠率
    iov = 0
    if iou > 0:
        iov = intersection_area / min(v_intersection_area1, v_intersection_area2)
    return iou > 0 and iov > thresh


def merge_union_boxes(boxes):
    """将相连Box合并起来"""

    def merge_boxes(bbox1, bbox2):
        """计算合并后的矩形坐标"""
        rect1 = bbox2rect(bbox1)
        rect2 = bbox2rect(bbox2)

        x1 = min(rect1[0], rect2[0])
        y1 = min(rect1[1], rect2[1])
        x2 = max(rect1[2], rect2[2])
        y2 = max(rect1[3], rect2[3])

        return rect2bbox([x1, y1, x2, y2])

    # 存放合并后的Box集合
    union_boxes = []
    # 存放已处理的Box序号
    done_indexes = []

    # 按照从上到下，从左到右排序
    size = len(boxes)
    for i, box1 in enumerate(boxes):
        if i in done_indexes:
            # 已处理，跳过
            continue
        elif i == size - 1:
            # 最后一个
            union_boxes.append(copy.deepcopy(box1))
            break

        # 将相连的边界框合并成新的边界框
        union_box = copy.deepcopy(box1)
        for j in range(i + 1, size):
            # 与其他边界框进行比较处理
            box2 = boxes[j]
            if is_union_horizontal(union_box, box2):
                # 相连合并成新边界框
                done_indexes.append(j)
                union_box = merge_boxes(union_box, box2)

        for k, box2 in enumerate(union_boxes):
            # 与已处理的边界框进行比较处理，避免遗漏
            if is_union_horizontal(union_box, box2):
                # 和已处理的边界框相连
                done_indexes.append(i)
                union_boxes[k] = merge_boxes(union_box, box2)

        if i not in done_indexes:
            # 将新增合并边界框添加到集合中
            done_indexes.append(i)
            union_boxes.append(union_box)

    return union_boxes


class DBPostProcess(object):
    """
    The post process for Differentiable Binarization (DB).
    """

    def __init__(self,
                 thresh=0.3,
                 box_thresh=0.7,
                 max_candidates=1000,
                 unclip_ratio=2.0,
                 use_dilation=False,
                 score_mode="fast",
                 box_type='quad',
                 **kwargs):
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio
        self.min_size = 3
        self.score_mode = score_mode
        self.box_type = box_type
        assert score_mode in [
            "slow", "fast"
        ], "Score mode must be in [slow, fast] but got: {}".format(score_mode)

        self.dilation_kernel = None if not use_dilation else np.array(
            [[1, 1], [1, 1]])

    def polygons_from_bitmap(self, pred, _bitmap, dest_width, dest_height):
        """
        _bitmap: single map with shape (1, H, W),
            whose values are binarized as {0, 1}
        """

        bitmap = _bitmap
        height, width = bitmap.shape

        boxes = []
        scores = []

        contours, _ = cv2.findContours((bitmap * 255).astype(np.uint8),
                                       cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours[:self.max_candidates]:
            epsilon = 0.002 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            points = approx.reshape((-1, 2))
            if points.shape[0] < 4:
                continue

            score = self.box_score_fast(pred, points.reshape(-1, 2))
            if self.box_thresh > score:
                continue

            if points.shape[0] > 2:
                box = self.unclip(points, self.unclip_ratio)
                if len(box) > 1:
                    continue
            else:
                continue
            box = box.reshape(-1, 2)

            _, sside = self.get_mini_boxes(box.reshape((-1, 1, 2)))
            if sside < self.min_size + 2:
                continue

            box = np.array(box)
            box[:, 0] = np.clip(
                np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(
                np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes.append(box.tolist())
            scores.append(score)
        return boxes, scores

    def boxes_from_bitmap(self, pred, _bitmap, dest_width, dest_height):
        """
        _bitmap: single map with shape (1, H, W),
                whose values are binarized as {0, 1}
        """

        bitmap = _bitmap
        height, width = bitmap.shape

        outs = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST,
                                cv2.CHAIN_APPROX_SIMPLE)
        if len(outs) == 3:
            img, contours, _ = outs[0], outs[1], outs[2]
        elif len(outs) == 2:
            contours, _ = outs[0], outs[1]

        num_contours = min(len(contours), self.max_candidates)

        boxes = []
        scores = []
        for index in range(num_contours):
            contour = contours[index]
            points, sside = self.get_mini_boxes(contour)
            if sside < self.min_size:
                continue
            points = np.array(points)
            if self.score_mode == "fast":
                score = self.box_score_fast(pred, points.reshape(-1, 2))
            else:
                score = self.box_score_slow(pred, contour)
            if self.box_thresh > score:
                continue

            box = self.unclip(points, self.unclip_ratio).reshape(-1, 1, 2)
            box, sside = self.get_mini_boxes(box)
            if sside < self.min_size + 2:
                continue
            box = np.array(box)

            box[:, 0] = np.clip(
                np.round(box[:, 0] / width * dest_width), 0, dest_width)
            # 检测框普遍偏下，上移2个像素
            box[:, 1] = np.clip(
                np.round(box[:, 1] / height * dest_height - 2), 0, dest_height)
            boxes.append(box.astype("int32"))
            scores.append(score)
        return np.array(boxes, dtype="int32"), scores

    def unclip(self, box, unclip_ratio):
        poly = Polygon(box)
        # 根据多边形的面积和 unclip_ratio 计算偏移距离
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded

    def get_mini_boxes(self, contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = [
            points[index_1], points[index_2], points[index_3], points[index_4]
        ]
        return box, min(bounding_box[1])

    def box_score_fast(self, bitmap, _box):
        """
        box_score_fast: use bbox mean score as the mean score
        """
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype("int32"), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype("int32"), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype("int32"), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype("int32"), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype("int32"), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

    def box_score_slow(self, bitmap, contour):
        '''
        box_score_slow: use polyon mean score as the mean score
        '''
        h, w = bitmap.shape[:2]
        contour = contour.copy()
        contour = np.reshape(contour, (-1, 2))

        xmin = np.clip(np.min(contour[:, 0]), 0, w - 1)
        xmax = np.clip(np.max(contour[:, 0]), 0, w - 1)
        ymin = np.clip(np.min(contour[:, 1]), 0, h - 1)
        ymax = np.clip(np.max(contour[:, 1]), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)

        contour[:, 0] = contour[:, 0] - xmin
        contour[:, 1] = contour[:, 1] - ymin

        cv2.fillPoly(mask, contour.reshape(1, -1, 2).astype("int32"), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

    def __call__(self, outs_dict, shape_list):
        pred = outs_dict['maps']
        if isinstance(pred, paddle.Tensor):
            pred = pred.numpy()
        pred = pred[:, 0, :, :]
        segmentation = pred > self.thresh

        boxes_batch = []
        for batch_index in range(pred.shape[0]):
            src_h, src_w, ratio_h, ratio_w = shape_list[batch_index]
            if self.dilation_kernel is not None:
                mask = cv2.dilate(
                    np.array(segmentation[batch_index]).astype(np.uint8),
                    self.dilation_kernel)
            else:
                mask = segmentation[batch_index]
            if self.box_type == 'poly':
                boxes, scores = self.polygons_from_bitmap(pred[batch_index],
                                                          mask, src_w, src_h)
            elif self.box_type == 'quad':
                boxes, scores = self.boxes_from_bitmap(pred[batch_index], mask,
                                                       src_w, src_h)
                # todo: 合并相连的文本框
                boxes = merge_union_boxes(boxes)
            else:
                raise ValueError("box_type can only be one of ['quad', 'poly']")

            boxes_batch.append({'points': boxes})
        return boxes_batch


class DistillationDBPostProcess(object):
    def __init__(self,
                 model_name=None,
                 key=None,
                 thresh=0.3,
                 box_thresh=0.6,
                 max_candidates=1000,
                 unclip_ratio=1.5,
                 use_dilation=False,
                 score_mode="fast",
                 box_type='quad',
                 **kwargs):
        self.model_name = model_name if model_name is not None else ["student"]
        self.key = key
        self.post_process = DBPostProcess(
            thresh=thresh,
            box_thresh=box_thresh,
            max_candidates=max_candidates,
            unclip_ratio=unclip_ratio,
            use_dilation=use_dilation,
            score_mode=score_mode,
            box_type=box_type)

    def __call__(self, predicts, shape_list):
        results = {}
        for k in self.model_name:
            results[k] = self.post_process(predicts[k], shape_list=shape_list)
        return results
