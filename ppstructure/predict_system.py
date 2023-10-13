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

import os
import sys
import subprocess

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'
import cv2
import json
import numpy as np
import time
import logging
from copy import deepcopy

from ppocr.utils.utility import get_image_file_list, check_and_read
from ppocr.utils.logging import get_logger
from ppocr.utils.visual import draw_ser_results, draw_re_results
from tools.infer.predict_system import TextSystem
from ppstructure.layout.predict_layout import LayoutPredictor
from ppstructure.table.predict_table import TableSystem, to_excel
from ppstructure.utility import parse_args, draw_structure_result

logger = get_logger()


class StructureSystem(object):
    def __init__(self, args):
        self.mode = args.mode
        self.recovery = args.recovery
        # 保存参数信息
        self.args = args

        self.image_orientation_predictor = None
        if args.image_orientation:
            import paddleclas
            self.image_orientation_predictor = paddleclas.PaddleClas(
                model_name="text_image_orientation")

        if self.mode == 'structure':
            if not args.show_log:
                logger.setLevel(logging.INFO)
            if args.layout == False and args.ocr == True:
                args.ocr = False
                logger.warning(
                    "When args.layout is false, args.ocr is automatically set to false"
                )
            # args.drop_score = 0
            # init model
            self.layout_predictor = None
            self.text_system = None
            self.table_system = None
            if args.layout:
                # 版面分析器
                self.layout_predictor = LayoutPredictor(args)
                if args.ocr:
                    # 文本识别
                    self.text_system = TextSystem(args)
            if args.table:
                if self.text_system is not None:
                    self.table_system = TableSystem(
                        args, self.text_system.text_detector,
                        self.text_system.text_recognizer)
                else:
                    self.table_system = TableSystem(args)

        elif self.mode == 'kie':
            from ppstructure.kie.predict_kie_token_ser_re import SerRePredictor
            self.kie_predictor = SerRePredictor(args)

    def __call__(self, img,
                 return_ocr_result_in_table=False,
                 wl_region_types: list[str] = None,
                 bl_region_types: list[str] = None, img_idx=0):
        """
            文档分析+识别
            @param return_ocr_result_in_table: 是否需要返回表格OCR结果
            @param wl_region_types: 需要特别处理的版面类型
            @param bl_region_types: 不需要特别处理的版面类型（优先级高于 wl_region_types）
        """
        time_dict = {
            'image_orientation': 0,
            'layout': 0,
            'table': 0,
            'table_match': 0,
            'det': 0,
            'rec': 0,
            'kie': 0,
            'all': 0
        }
        start = time.time()
        if self.image_orientation_predictor is not None:
            # 方向检测+矫正
            tic = time.time()
            cls_result = self.image_orientation_predictor.predict(
                input_data=img)
            cls_res = next(cls_result)
            angle = cls_res[0]['label_names'][0]
            cv_rotate_code = {
                '90': cv2.ROTATE_90_COUNTERCLOCKWISE,
                '180': cv2.ROTATE_180,
                '270': cv2.ROTATE_90_CLOCKWISE
            }
            if angle in cv_rotate_code:
                img = cv2.rotate(img, cv_rotate_code[angle])
            toc = time.time()
            time_dict['image_orientation'] = toc - tic

        if self.mode == 'structure':
            ori_im = img.copy()
            # 图片的高和宽
            h, w = ori_im.shape[:2]
            if self.layout_predictor is not None:
                # 进行版面推理分析
                layout_res, elapse = self.layout_predictor(img)
                time_dict['layout'] += elapse
            else:
                # 不进行版面分析，整个页面当作表格处理
                layout_res = [dict(bbox=None, label='table')]
            res_list = []

            other_img = img.copy()
            has_other = True
            for region in layout_res:
                region_type = region['label']
                if bl_region_types is not None and region_type in bl_region_types:
                    # 跳过不需要特别处理的区域
                    continue

                if wl_region_types is not None and region_type not in wl_region_types:
                    # 跳过需要特别处理的区域
                    continue

                res = ''
                if region['bbox'] is not None:
                    x1, y1, x2, y2 = region['bbox']
                    # 为了后续更好的识别，向外延伸5个单位
                    x1, y1, x2, y2 = max(0, int(x1) - 5), max(0, int(y1) - 5), min(w, int(x2)) + 5, min(h, int(y2) + 5)
                    roi_img = ori_im[y1:y2, x1:x2, :]
                else:
                    has_other = False
                    x1, y1, x2, y2 = 0, 0, w, h
                    roi_img = ori_im

                # 从图片中移除已识别区域
                other_img[y1:y2, x1:x2, :] = 255

                if region_type == 'table':
                    # 表格处理
                    if self.table_system is not None:
                        res, table_time_dict = self.table_system(
                            roi_img, return_ocr_result_in_table)

                        # 统计检测+时间耗时
                        time_dict['table'] += table_time_dict['table']
                        time_dict['table_match'] += table_time_dict['match']
                        time_dict['det'] += table_time_dict['det']
                        time_dict['rec'] += table_time_dict['rec']
                else:
                    if self.text_system is not None:
                        if self.recovery:
                            # 恢复版面，除了需要识别的区域，其他区域置白
                            wht_im = np.ones(ori_im.shape, dtype=ori_im.dtype) * 255
                            wht_im[y1:y2, x1:x2, :] = roi_img
                            filter_boxes, filter_rec_res, ocr_time_dict = self.text_system(
                                wht_im)
                        else:
                            filter_boxes, filter_rec_res, ocr_time_dict = self.text_system(
                                roi_img)

                        # 统计检测+时间耗时
                        time_dict['det'] += ocr_time_dict['det']
                        time_dict['rec'] += ocr_time_dict['rec']

                        res = postprocess_rec_res(filter_boxes, filter_rec_res, self.recovery, (x1, y1))

                res_list.append({
                    'type': region['label'].lower(),
                    'bbox': [x1, y1, x2, y2],
                    'img': roi_img,
                    'res': res,
                    'img_idx': img_idx
                })

            if has_other:
                # 对版面分析未识别到的区域进行OCR识别
                filter_boxes, filter_rec_res, ocr_time_dict = self.text_system(
                    other_img)

                # 统计检测+时间耗时
                time_dict['det'] += ocr_time_dict['det']
                time_dict['rec'] += ocr_time_dict['rec']

                res = postprocess_rec_res(filter_boxes, filter_rec_res, self.recovery, (0, 0))
                res_list.append({
                    'type': 'other',
                    'bbox': [0, 0, w, h],
                    'img': other_img,
                    'res': res,
                    'img_idx': img_idx
                })

            end = time.time()
            time_dict['all'] = end - start
            return res_list, time_dict
        elif self.mode == 'kie':
            re_res, elapse = self.kie_predictor(img)
            time_dict['kie'] = elapse
            time_dict['all'] = elapse
            return re_res[0], time_dict
        return None, None


def postprocess_rec_res(filter_boxes, filter_rec_res, recovery=False, offset=(0, 0)):
    # remove style char,
    # when using the recognition model trained on the PubtabNet dataset,
    # it will recognize the text format in the table, such as <b>
    style_token = [
        '<strike>', '<strike>', '<sup>', '</sub>', '<b>',
        '</b>', '<sub>', '</sup>', '<overline>',
        '</overline>', '<underline>', '</underline>', '<i>',
        '</i>'
    ]
    res = []
    for box, rec_res in zip(filter_boxes, filter_rec_res):
        rec_str, rec_conf = rec_res[0], rec_res[1]
        for token in style_token:
            if token in rec_str:
                rec_str = rec_str.replace(token, '')
        if not recovery:
            box += [offset[0], offset[1]]
        res.append({
            'text': rec_str,
            'confidence': float(rec_conf),
            'text_region': box.tolist(),
            # 返回结果添加字符位置信息
            'char_pos': rec_res[2] if len(rec_res) == 3 else []
        })

    return res


def save_structure_res(res, save_folder, img_name, img_idx=0):
    excel_save_folder = os.path.join(save_folder, img_name)
    os.makedirs(excel_save_folder, exist_ok=True)
    res_cp = deepcopy(res)
    # save res
    with open(
            os.path.join(excel_save_folder, 'res_{}.txt'.format(img_idx)),
            'w',
            encoding='utf8') as f:
        for region in res_cp:
            roi_img = region.pop('img')
            f.write('{}\n'.format(json.dumps(region)))

            if region['type'].lower() == 'table' and len(region[
                                                             'res']) > 0 and 'html' in region['res']:
                excel_path = os.path.join(
                    excel_save_folder,
                    '{}_{}.xlsx'.format(region['bbox'], img_idx))
                to_excel(region['res']['html'], excel_path)
            elif region['type'].lower() == 'figure':
                img_path = os.path.join(
                    excel_save_folder,
                    '{}_{}.jpg'.format(region['bbox'], img_idx))
                cv2.imwrite(img_path, roi_img)


def main(args):
    image_file_list = get_image_file_list(args.image_dir)
    image_file_list = image_file_list
    image_file_list = image_file_list[args.process_id::args.total_process_num]

    if not args.use_pdf2docx_api:
        structure_sys = StructureSystem(args)
        save_folder = os.path.join(args.output, structure_sys.mode)
        os.makedirs(save_folder, exist_ok=True)
    img_num = len(image_file_list)

    for i, image_file in enumerate(image_file_list):
        logger.info("[{}/{}] {}".format(i, img_num, image_file))
        img, flag_gif, flag_pdf = check_and_read(image_file)
        img_name = os.path.basename(image_file).split('.')[0]

        if args.recovery and args.use_pdf2docx_api and flag_pdf:
            from pdf2docx.converter import Converter
            os.makedirs(args.output, exist_ok=True)
            docx_file = os.path.join(args.output,
                                     '{}_api.docx'.format(img_name))
            cv = Converter(image_file)
            cv.convert(docx_file)
            cv.close()
            logger.info('docx save to {}'.format(docx_file))
            continue

        if not flag_gif and not flag_pdf:
            img = cv2.imread(image_file)

        if not flag_pdf:
            if img is None:
                logger.error("error in loading image:{}".format(image_file))
                continue
            imgs = [img]
        else:
            imgs = img

        all_res = []
        for index, img in enumerate(imgs):
            res, time_dict = structure_sys(img, img_idx=index)
            img_save_path = os.path.join(save_folder, img_name,
                                         'show_{}.jpg'.format(index))
            os.makedirs(os.path.join(save_folder, img_name), exist_ok=True)
            if structure_sys.mode == 'structure' and res != []:
                draw_img = draw_structure_result(img, res, args.vis_font_path)
                save_structure_res(res, save_folder, img_name, index)
            elif structure_sys.mode == 'kie':
                if structure_sys.kie_predictor.predictor is not None:
                    draw_img = draw_re_results(
                        img, res, font_path=args.vis_font_path)
                else:
                    draw_img = draw_ser_results(
                        img, res, font_path=args.vis_font_path)

                with open(
                        os.path.join(save_folder, img_name,
                                     'res_{}_kie.txt'.format(index)),
                        'w',
                        encoding='utf8') as f:
                    res_str = '{}\t{}\n'.format(
                        image_file,
                        json.dumps(
                            {
                                "ocr_info": res
                            }, ensure_ascii=False))
                    f.write(res_str)
            if res != []:
                cv2.imwrite(img_save_path, draw_img)
                logger.info('result save to {}'.format(img_save_path))
            if args.recovery and res != []:
                from ppstructure.recovery.recovery_to_doc import sorted_layout_boxes, convert_info_docx
                h, w, _ = img.shape
                res = sorted_layout_boxes(res, w)
                all_res += res

        if args.recovery and all_res != []:
            try:
                convert_info_docx(img, all_res, save_folder, img_name)
            except Exception as ex:
                logger.error("error in layout recovery image:{}, err msg: {}".
                             format(image_file, ex))
                continue
        logger.info("Predict time : {:.3f}s".format(time_dict['all']))


if __name__ == "__main__":
    args = parse_args()
    if args.use_mp:
        p_list = []
        total_process_num = args.total_process_num
        for process_id in range(total_process_num):
            cmd = [sys.executable, "-u"] + sys.argv + [
                "--process_id={}".format(process_id),
                "--use_mp={}".format(False)
            ]
            p = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stdout)
            p_list.append(p)
        for p in p_list:
            p.wait()
    else:
        main(args)
