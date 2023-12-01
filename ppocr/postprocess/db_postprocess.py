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


def box2rect(box):
    """
    Box坐标    [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    转Rect坐标  [x1, y1, x2, y2]
    """
    assert np.shape(box) == (4, 2)

    np_box = np.reshape(box, -1)
    x1, x2 = min(np_box[::2]), max(np_box[::2])
    y1, y2 = min(np_box[1::2]), max(np_box[1::2])

    return [x1, y1, x2, y2]


def box2roi(box):
    """
    Box坐标   [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    转ROI坐标  [x1, y1, w, h]
    """
    assert np.shape(box) == (4, 2)

    np_box = np.reshape(box, -1)
    x1, x2 = min(np_box[::2]), max(np_box[::2])
    y1, y2 = min(np_box[1::2]), max(np_box[1::2])

    return [x1, y1, x2 - x1, y2 - y1]


def rect2box(rect):
    """
    Rect坐标  [x1, y1, x2, y2]
    转Box坐标  [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    """
    assert np.shape(rect) == (4,)

    x1, y1, x2, y2 = rect
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def get_slope(box):
    assert np.shape(box) == (4, 2)

    np_box = np.reshape(box, -1)

    c_x1, c_x2 = (np_box[0] + np_box[2]) / 2, (np_box[4] + np_box[6]) / 2
    c_y1, c_y2 = (np_box[1] + np_box[7]) / 2, (np_box[3] + np_box[5]) / 2

    delta_x, delta_y = c_x2 - c_x1, c_y2 - c_y1

    w, h = (np_box[2] + np_box[4] - np_box[0] - np_box[6]) / 2, (np_box[5] + np_box[7] - np_box[1] - np_box[3]) / 2

    # 水平斜率：deltaY / width
    slope_h = float('inf') if w == 0 else delta_y / w

    # 垂直斜率: deltaX / height
    slope_v = float('inf') if h == 0 else delta_x / h

    return slope_h, slope_v


def predict_y(src_pt, dest_x, slope_h=0):
    delta_x = dest_x - src_pt[0]
    return src_pt[1] if delta_x == 0 else src_pt[1] + delta_x * slope_h


def is_union_horizontal(bbox_1, bbox_2, threshold=0.7):
    """判断是否水平相连"""
    roi1 = box2roi(bbox_1)
    roi2 = box2roi(bbox_2)

    # 计算两个矩形的交集RECT坐标
    i_left = max(roi1[0], roi2[0])
    i_top = max(roi1[1], roi2[1])
    i_right = min(roi1[0] + roi1[2], roi2[0] + roi2[2])
    i_bottom = min(roi1[1] + roi1[3], roi2[1] + roi2[3])

    # 计算交集面积
    i_area = cal_area([i_left, i_top, i_right, i_bottom])

    if i_area == 0:
        # 没有交集
        return False

    # 使用重合部分垂直方向重叠率判断是否水平相连
    # 引入斜率，减少倾角大时的误判概率
    slope = get_slope(bbox_1)[0] if roi1[2] > roi2[2] else get_slope(bbox_2)[0]

    # 根据斜率，计算在重叠区域，box1 和 box2的 top 和 bottom
    v_top1, v_bottom1 = predict_y(bbox_1[0], i_left, slope), predict_y(bbox_1[2], i_right, slope)
    v_top2, v_bottom2 = predict_y(bbox_2[0], i_left, slope), predict_y(bbox_2[2], i_right, slope)

    v_area1 = cal_area([i_left, v_top1, i_right, v_bottom1])
    v_area2 = cal_area([i_left, v_top2, i_right, v_bottom2])

    # 垂直方向重叠率
    iov = i_area / min(v_area1, v_area2)
    return iov > threshold


def merge_union_boxes(boxes, debug=False):
    """
    将相连Box合并起来
    :param boxes 需要合并的边框，Shape（-1，4，2）
    """
    size = len(boxes)
    if size < 2:
        # 少于2个，直接返回
        return boxes

    def validate_box(box):
        """检查边框是否符合要求"""
        w1, w2 = box[1][0] - box[0][0], box[2][0] - box[3][0]
        h1, h2 = box[3][1] - box[0][1], box[2][1] - box[1][1]

        if max(w1, w2) / min(w1, w2) > 1.3 or max(h1, h2) / min(h1, h2) > 1.3:
            # 当Box对边的长度比超过1.3时，舍弃该Box
            return False

        # if abs(get_slope(box)[0]) > 0.5:
        #     # 舍弃倾斜的Box
        #     return False

        return True

    def merge_boxes(bbox1, bbox2, threshold=10):
        """
        计算合并后的Box坐标
        :param bbox1 Box1坐标
        :param bbox2 Box2坐标
        :param threshold 垂直方向上的偏差阈值
        """
        # 计算两个边框的水平斜率
        slope1 = get_slope(bbox1)[0]
        slope2 = get_slope(bbox2)[0]

        # 计算合并后的四个顶点的X，Y坐标
        x1, y1 = bbox1[0] if bbox1[0][0] < bbox2[0][0] else bbox2[0]
        x2, y2 = bbox1[1] if bbox1[1][0] > bbox2[1][0] else bbox2[1]
        x3, y3 = bbox1[2] if bbox1[2][0] > bbox2[2][0] else bbox2[2]
        x4, y4 = bbox1[3] if bbox1[3][0] < bbox2[3][0] else bbox2[3]

        # 通过每个Box的斜率预测四个顶点的Y坐标
        if threshold >= abs(box1[0][0] - box2[0][0]) >= 0:
            y1 = min(predict_y(bbox1[0], x1, slope1), predict_y(bbox2[0], x1, slope2))
        else:
            pred_y1 = min(predict_y(bbox1[0], x1, slope1), predict_y(bbox2[0], x1, slope2))
            if abs(pred_y1 - y1) < threshold:
                y1 = min(y1, pred_y1)

        if threshold >= abs(box1[1][0] - box2[1][0]) >= 0:
            y2 = min(predict_y(bbox1[1], x2, slope1), predict_y(bbox2[1], x2, slope2))
        else:
            pred_y2 = min(predict_y(bbox1[1], x2, slope1), predict_y(bbox2[1], x2, slope2))
            if abs(pred_y2 - y2) < threshold:
                y2 = min(y2, pred_y2)

        if threshold >= abs(box1[2][0] - box2[2][0]) >= 0:
            y3 = max(predict_y(bbox1[2], x3, slope1), predict_y(bbox2[2], x3, slope2))
        else:
            pred_y3 = max(predict_y(bbox1[2], x3, slope1), predict_y(bbox2[2], x3, slope2))
            if abs(pred_y3 - y3) < threshold:
                y3 = max(y3, pred_y3)

        if threshold >= abs(box1[2][0] - box2[2][0]) >= 0:
            y4 = max(predict_y(bbox1[3], x4, slope1), predict_y(bbox2[3], x4, slope2))
        else:
            pred_y4 = max(predict_y(bbox1[3], x4, slope1), predict_y(bbox2[3], x4, slope2))
            if abs(pred_y4 - y4) < threshold:
                y4 = max(y4, pred_y4)

        return [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]

    # 存放合并后的Box集合
    merged_boxes = []
    # 存放已处理的Box序号
    done_indexes = []

    boxes = list(filter(validate_box, boxes))

    for i, box1 in enumerate(boxes):
        if i in done_indexes:
            # 已处理，跳过
            continue
        elif i == len(boxes) - 1:
            # 最后一个
            merged_boxes.append(copy.deepcopy(box1))
            break

        # 将相连的边界框合并成新的边界框
        merged_box = copy.deepcopy(box1)

        for j in range(i + 1, len(boxes)):
            # 与其他边界框进行比较处理
            box2 = boxes[j]
            if is_union_horizontal(merged_box, box2):
                # 相连合并成新边界框
                done_indexes.append(j)
                merged_box = merge_boxes(merged_box, box2)

        for j, box2 in enumerate(merged_boxes):
            # 与已处理的边界框进行比较处理，避免遗漏
            if is_union_horizontal(merged_box, box2):
                # 和已处理的边界框相连
                done_indexes.append(i)
                merged_boxes[j] = merge_boxes(merged_box, box2)

        if i not in done_indexes:
            # 将新增合并边界框添加到集合中
            done_indexes.append(i)
            merged_boxes.append(merged_box)

    return merged_boxes


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
                # boxes = merge_union_boxes(boxes)
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
