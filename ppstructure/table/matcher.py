# copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from ppstructure.table.table_master_match import deal_eb_token, deal_bb


def distance(box_1, box_2):
    # 计算中心距离
    x1, y1, x2, y2 = box_1
    x3, y3, x4, y4 = box_2

    dist_x = abs((x1 + x2) / 2 - (x3 + x4) / 2)
    dist_y = abs((y1 + y2) / 2 - (y3 + y4) / 2)

    return dist_x + dist_y

    # dis = abs(x3 - x1) + abs(y3 - y1) + abs(x4 - x2) + abs(y4 - y2)
    # dis_2 = abs(x3 - x1) + abs(y3 - y1)
    # dis_3 = abs(x4 - x2) + abs(y4 - y2)
    # return dis + min(dis_2, dis_3)


def compute_iou(rec1, rec2):
    """
    计算交并比（IoU）
    :param rec1: (x0, y0, x1, y1), which reflects
            (left, top, right, bottom)
    :param rec2: (x0, y0, x1, y1)
    :return: scala value of IoU
    """
    # 提取两个边界框的坐标
    x1_0, y1_0, x1_1, y1_1 = rec1
    x2_0, y2_0, x2_1, y2_1 = rec2

    # 计算两个边界框的面积
    s1 = (x1_1 - x1_0) * (y1_1 - y1_0)
    s2 = (x2_1 - x2_0) * (y2_1 - y2_0)
    sum_area = s1 + s2

    # 计算交集矩形的坐标
    left_line = max(x1_0, x2_0)
    right_line = min(x1_1, x2_1)
    top_line = max(y1_0, y2_0)
    bottom_line = min(y1_1, y2_1)

    # 判断是否存在交集
    if left_line >= right_line or top_line >= bottom_line:
        # 没有交集
        return 0.0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        # 计算rec1有多少区域和rec2重合
        return (intersect / s1) * 1.0


class TableMatch:
    def __init__(self, filter_ocr_result=False, use_master=False):
        self.filter_ocr_result = filter_ocr_result
        self.use_master = use_master

    def __call__(self, structure_res, dt_boxes, rec_res):
        pred_structures, pred_bboxes = structure_res
        if self.filter_ocr_result:
            dt_boxes, rec_res = self._filter_ocr_result(pred_bboxes, dt_boxes,
                                                        rec_res)
        matched_index = self.match_result(dt_boxes, pred_bboxes)
        if self.use_master:
            pred_html, pred = self.get_pred_html_master(pred_structures,
                                                        matched_index, rec_res)
        else:
            pred_html, pred = self.get_pred_html(pred_structures, matched_index,
                                                 rec_res)
        return pred_html

    def match_result(self, dt_boxes, cell_bboxes):
        matched = {}
        # 判断每个文本框所属的单元格
        for i, gt_box in enumerate(dt_boxes):
            # 遍历文本框
            distances = []
            for j, cell_box in enumerate(cell_bboxes):
                # 遍历单元格
                if len(cell_box) == 8:
                    # 格式转换：xyxyxyxy --> xyxy
                    cell_box = [
                        np.min(cell_box[0::2]), np.min(cell_box[1::2]),
                        np.max(cell_box[0::2]), np.max(cell_box[1::2])
                    ]
                distances.append((distance(gt_box, cell_box),
                                  1. - compute_iou(gt_box, cell_box)
                                  ))  # compute iou and l1 distance

            sorted_distances = distances.copy()
            # select det box by iou and l1 distance
            sorted_distances = sorted(
                sorted_distances, key=lambda item: (item[1], item[0]))

            # 如果一个文本库部分覆盖到多个单元格，且覆盖面相近时，优先判断距离
            print(i, sorted_distances[0], sorted_distances[1])
            prop_cell_idx = distances.index(sorted_distances[0])
            if len(sorted_distances) > 1 \
                    and abs(sorted_distances[0][1] - sorted_distances[1][1]) < 0.1:
                if sorted_distances[0][0] > sorted_distances[1][0]:
                    prop_cell_idx = distances.index(sorted_distances[1])

            print(f'{i} >> {prop_cell_idx}')
            if prop_cell_idx not in matched.keys():
                matched[prop_cell_idx] = [i]
            else:
                matched[prop_cell_idx].append(i)
        return matched

    def get_pred_html(self, pred_structures, matched_index, ocr_contents):
        end_html = []
        td_index = 0
        for tag in pred_structures:
            if '</td>' in tag:
                if '<td></td>' == tag:
                    end_html.extend('<td>')
                if td_index in matched_index.keys():
                    b_with = False
                    if '<b>' in ocr_contents[matched_index[td_index][
                        0]] and len(matched_index[td_index]) > 1:
                        b_with = True
                        end_html.extend('<b>')
                    for i, td_index_index in enumerate(matched_index[td_index]):
                        content = ocr_contents[td_index_index][0]
                        if len(matched_index[td_index]) > 1:
                            if len(content) == 0:
                                continue
                            if content[0] == ' ':
                                content = content[1:]
                            if '<b>' in content:
                                content = content[3:]
                            if '</b>' in content:
                                content = content[:-4]
                            if len(content) == 0:
                                continue
                            if i != len(matched_index[
                                            td_index]) - 1 and ' ' != content[-1]:
                                content += ' '
                        end_html.extend(content)
                    if b_with:
                        end_html.extend('</b>')
                if '<td></td>' == tag:
                    end_html.append('</td>')
                else:
                    end_html.append(tag)
                td_index += 1
            else:
                end_html.append(tag)
        return ''.join(end_html), end_html

    def get_pred_html_master(self, pred_structures, matched_index,
                             ocr_contents):
        end_html = []
        td_index = 0
        for token in pred_structures:
            if '</td>' in token:
                txt = ''
                b_with = False
                if td_index in matched_index.keys():
                    if '<b>' in ocr_contents[matched_index[td_index][
                        0]] and len(matched_index[td_index]) > 1:
                        b_with = True
                    for i, td_index_index in enumerate(matched_index[td_index]):
                        content = ocr_contents[td_index_index][0]
                        if len(matched_index[td_index]) > 1:
                            if len(content) == 0:
                                continue
                            if content[0] == ' ':
                                content = content[1:]
                            if '<b>' in content:
                                content = content[3:]
                            if '</b>' in content:
                                content = content[:-4]
                            if len(content) == 0:
                                continue
                            if i != len(matched_index[
                                            td_index]) - 1 and ' ' != content[-1]:
                                content += ' '
                        txt += content
                if b_with:
                    txt = '<b>{}</b>'.format(txt)
                if '<td></td>' == token:
                    token = '<td>{}</td>'.format(txt)
                else:
                    token = '{}</td>'.format(txt)
                td_index += 1
            token = deal_eb_token(token)
            end_html.append(token)
        html = ''.join(end_html)
        html = deal_bb(html)
        return html, end_html

    def _filter_ocr_result(self, pred_bboxes, dt_boxes, rec_res):
        y1 = pred_bboxes[:, 1::2].min()
        new_dt_boxes = []
        new_rec_res = []

        for box, rec in zip(dt_boxes, rec_res):
            if np.max(box[1::2]) < y1:
                continue
            new_dt_boxes.append(box)
            new_rec_res.append(rec)
        return new_dt_boxes, new_rec_res
