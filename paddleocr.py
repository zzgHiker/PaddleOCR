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
import importlib

__dir__ = os.path.dirname(__file__)

import time

import uuid

import paddle
from PIL.ImageDraw import ImageDraw

sys.path.append(os.path.join(__dir__, ''))

import cv2
import logging
import numpy as np
from pathlib import Path
import base64
from io import BytesIO
from PIL import Image


def _import_file(module_name, file_path, make_importable=False):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if make_importable:
        sys.modules[module_name] = module
    return module


tools = _import_file(
    'tools', os.path.join(__dir__, 'tools/__init__.py'), make_importable=True)
ppocr = importlib.import_module('ppocr', 'paddleocr')
ppstructure = importlib.import_module('ppstructure', 'paddleocr')
from ppocr.utils.logging import get_logger
from tools.infer import predict_system
from ppocr.utils.utility import check_and_read, get_image_file_list, alpha_to_color, binarize_img
from ppocr.utils.network import maybe_download, download_with_progressbar, is_link, confirm_model_dir_url
from ppocr.utils.visual import draw_ser_results, draw_re_results
from tools.infer.utility import draw_ocr, str2bool, check_gpu
from ppstructure.utility import init_args, draw_structure_result
from ppstructure.predict_system import StructureSystem, save_structure_res, to_excel, save_kie_res, save_crop_image

logger = get_logger()
__all__ = [
    'PaddleOCR', 'PPStructure', 'draw_ocr', 'draw_structure_result',
    'save_structure_res', 'download_with_progressbar', 'to_excel'
]

SUPPORT_DET_MODEL = ['DB']
SUPPORT_REC_MODEL = ['CRNN', 'SVTR_LCNet']
BASE_DIR = os.path.expanduser("~/.paddleocr/")

DEFAULT_OCR_MODEL_VERSION = 'PP-OCRv4'
SUPPORT_OCR_MODEL_VERSION = ['PP-OCR', 'PP-OCRv2', 'PP-OCRv3', 'PP-OCRv4']
DEFAULT_STRUCTURE_MODEL_VERSION = 'PP-StructureV2'
SUPPORT_STRUCTURE_MODEL_VERSION = ['PP-Structure', 'PP-StructureV2']
MODEL_URLS = {
    'OCR': {
        'PP-OCRv4': {
            'det': {
                'ch': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_infer.tar',
                },
                'en': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar',
                },
                'ml': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/Multilingual_PP-OCRv3_det_infer.tar'
                }
            },
            'rec': {
                'ch': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_infer.tar',
                    'dict_path': './ppocr/utils/ppocr_keys_v1.txt'
                },
                'en': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_rec_infer.tar',
                    'dict_path': './ppocr/utils/en_dict.txt'
                },
                'korean': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/PP-OCRv4/multilingual/korean_PP-OCRv4_rec_infer.tar',
                    'dict_path': './ppocr/utils/dict/korean_dict.txt'
                },
                'japan': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/PP-OCRv4/multilingual/japan_PP-OCRv4_rec_infer.tar',
                    'dict_path': './ppocr/utils/dict/japan_dict.txt'
                },
                'chinese_cht': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/chinese_cht_PP-OCRv3_rec_infer.tar',
                    'dict_path': './ppocr/utils/dict/chinese_cht_dict.txt'
                },
                'ta': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/PP-OCRv4/multilingual/ta_PP-OCRv4_rec_infer.tar',
                    'dict_path': './ppocr/utils/dict/ta_dict.txt'
                },
                'te': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/PP-OCRv4/multilingual/te_PP-OCRv4_rec_infer.tar',
                    'dict_path': './ppocr/utils/dict/te_dict.txt'
                },
                'ka': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/PP-OCRv4/multilingual/ka_PP-OCRv4_rec_infer.tar',
                    'dict_path': './ppocr/utils/dict/ka_dict.txt'
                },
                'latin': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/latin_PP-OCRv3_rec_infer.tar',
                    'dict_path': './ppocr/utils/dict/latin_dict.txt'
                },
                'arabic': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/PP-OCRv4/multilingual/arabic_PP-OCRv4_rec_infer.tar',
                    'dict_path': './ppocr/utils/dict/arabic_dict.txt'
                },
                'cyrillic': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/cyrillic_PP-OCRv3_rec_infer.tar',
                    'dict_path': './ppocr/utils/dict/cyrillic_dict.txt'
                },
                'devanagari': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/PP-OCRv4/multilingual/devanagari_PP-OCRv4_rec_infer.tar',
                    'dict_path': './ppocr/utils/dict/devanagari_dict.txt'
                },
            },
            'cls': {
                'ch': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar',
                }
            },
        },
        'PP-OCRv3': {
            'det': {
                'ch': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar',
                },
                'en': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar',
                },
                'ml': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/Multilingual_PP-OCRv3_det_infer.tar'
                }
            },
            'rec': {
                'ch': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar',
                    'dict_path': './ppocr/utils/ppocr_keys_v1.txt'
                },
                'en': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_infer.tar',
                    'dict_path': './ppocr/utils/en_dict.txt'
                },
                'korean': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/korean_PP-OCRv3_rec_infer.tar',
                    'dict_path': './ppocr/utils/dict/korean_dict.txt'
                },
                'japan': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/japan_PP-OCRv3_rec_infer.tar',
                    'dict_path': './ppocr/utils/dict/japan_dict.txt'
                },
                'chinese_cht': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/chinese_cht_PP-OCRv3_rec_infer.tar',
                    'dict_path': './ppocr/utils/dict/chinese_cht_dict.txt'
                },
                'ta': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/ta_PP-OCRv3_rec_infer.tar',
                    'dict_path': './ppocr/utils/dict/ta_dict.txt'
                },
                'te': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/te_PP-OCRv3_rec_infer.tar',
                    'dict_path': './ppocr/utils/dict/te_dict.txt'
                },
                'ka': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/ka_PP-OCRv3_rec_infer.tar',
                    'dict_path': './ppocr/utils/dict/ka_dict.txt'
                },
                'latin': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/latin_PP-OCRv3_rec_infer.tar',
                    'dict_path': './ppocr/utils/dict/latin_dict.txt'
                },
                'arabic': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/arabic_PP-OCRv3_rec_infer.tar',
                    'dict_path': './ppocr/utils/dict/arabic_dict.txt'
                },
                'cyrillic': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/cyrillic_PP-OCRv3_rec_infer.tar',
                    'dict_path': './ppocr/utils/dict/cyrillic_dict.txt'
                },
                'devanagari': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/devanagari_PP-OCRv3_rec_infer.tar',
                    'dict_path': './ppocr/utils/dict/devanagari_dict.txt'
                },
            },
            'cls': {
                'ch': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar',
                }
            },
        },
        'PP-OCRv2': {
            'det': {
                'ch': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_det_infer.tar',
                },
            },
            'rec': {
                'ch': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_rec_infer.tar',
                    'dict_path': './ppocr/utils/ppocr_keys_v1.txt'
                }
            },
            'cls': {
                'ch': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar',
                }
            },
        },
        'PP-OCR': {
            'det': {
                'ch': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_det_infer.tar',
                },
                'en': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/en_ppocr_mobile_v2.0_det_infer.tar',
                },
                'structure': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/dygraph_v2.0/table/en_ppocr_mobile_v2.0_table_det_infer.tar'
                }
            },
            'rec': {
                'ch': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_infer.tar',
                    'dict_path': './ppocr/utils/ppocr_keys_v1.txt'
                },
                'en': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/en_number_mobile_v2.0_rec_infer.tar',
                    'dict_path': './ppocr/utils/en_dict.txt'
                },
                'french': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/french_mobile_v2.0_rec_infer.tar',
                    'dict_path': './ppocr/utils/dict/french_dict.txt'
                },
                'german': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/german_mobile_v2.0_rec_infer.tar',
                    'dict_path': './ppocr/utils/dict/german_dict.txt'
                },
                'korean': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/korean_mobile_v2.0_rec_infer.tar',
                    'dict_path': './ppocr/utils/dict/korean_dict.txt'
                },
                'japan': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/japan_mobile_v2.0_rec_infer.tar',
                    'dict_path': './ppocr/utils/dict/japan_dict.txt'
                },
                'chinese_cht': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/chinese_cht_mobile_v2.0_rec_infer.tar',
                    'dict_path': './ppocr/utils/dict/chinese_cht_dict.txt'
                },
                'ta': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/ta_mobile_v2.0_rec_infer.tar',
                    'dict_path': './ppocr/utils/dict/ta_dict.txt'
                },
                'te': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/te_mobile_v2.0_rec_infer.tar',
                    'dict_path': './ppocr/utils/dict/te_dict.txt'
                },
                'ka': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/ka_mobile_v2.0_rec_infer.tar',
                    'dict_path': './ppocr/utils/dict/ka_dict.txt'
                },
                'latin': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/latin_ppocr_mobile_v2.0_rec_infer.tar',
                    'dict_path': './ppocr/utils/dict/latin_dict.txt'
                },
                'arabic': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/arabic_ppocr_mobile_v2.0_rec_infer.tar',
                    'dict_path': './ppocr/utils/dict/arabic_dict.txt'
                },
                'cyrillic': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/cyrillic_ppocr_mobile_v2.0_rec_infer.tar',
                    'dict_path': './ppocr/utils/dict/cyrillic_dict.txt'
                },
                'devanagari': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/devanagari_ppocr_mobile_v2.0_rec_infer.tar',
                    'dict_path': './ppocr/utils/dict/devanagari_dict.txt'
                },
                'structure': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/dygraph_v2.0/table/en_ppocr_mobile_v2.0_table_rec_infer.tar',
                    'dict_path': 'ppocr/utils/dict/table_dict.txt'
                }
            },
            'cls': {
                'ch': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar',
                }
            },
        }
    },
    'STRUCTURE': {
        'PP-Structure': {
            'table': {
                'en': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/dygraph_v2.0/table/en_ppocr_mobile_v2.0_table_structure_infer.tar',
                    'dict_path': 'ppocr/utils/dict/table_structure_dict.txt'
                }
            }
        },
        'PP-StructureV2': {
            'table': {
                'en': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/ppstructure/models/slanet/en_ppstructure_mobile_v2.0_SLANet_infer.tar',
                    'dict_path': 'ppocr/utils/dict/table_structure_dict.txt'
                },
                'ch': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/ppstructure/models/slanet/ch_ppstructure_mobile_v2.0_SLANet_infer.tar',
                    'dict_path': 'ppocr/utils/dict/table_structure_dict_ch.txt'
                }
            },
            'layout': {
                'en': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/ppstructure/models/layout/picodet_lcnet_x1_0_fgd_layout_infer.tar',
                    'dict_path':
                        'ppocr/utils/dict/layout_dict/layout_publaynet_dict.txt'
                },
                'ch': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/ppstructure/models/layout/picodet_lcnet_x1_0_fgd_layout_cdla_infer.tar',
                    'dict_path':
                        'ppocr/utils/dict/layout_dict/layout_cdla_dict.txt'
                }
            }
        }
    }
}


def parse_args(mMain=True):
    import argparse
    parser = init_args()
    parser.add_help = mMain
    parser.add_argument("--lang", type=str, default='ch')
    parser.add_argument("--det", type=str2bool, default=True)
    parser.add_argument("--rec", type=str2bool, default=True)
    parser.add_argument("--type", type=str, default='ocr')
    parser.add_argument(
        "--ocr_version",
        type=str,
        choices=SUPPORT_OCR_MODEL_VERSION,
        default='PP-OCRv4',
        help='OCR Model version, the current model support list is as follows: '
             '1. PP-OCRv4/v3 Support Chinese and English detection and recognition model, and direction classifier model'
             '2. PP-OCRv2 Support Chinese detection and recognition model. '
             '3. PP-OCR support Chinese detection, recognition and direction classifier and multilingual recognition model.'
    )
    parser.add_argument(
        "--structure_version",
        type=str,
        choices=SUPPORT_STRUCTURE_MODEL_VERSION,
        default='PP-StructureV2',
        help='Model version, the current model support list is as follows:'
             ' 1. PP-Structure Support en table structure model.'
             ' 2. PP-StructureV2 Support ch and en table structure model.')

    for action in parser._actions:
        if action.dest in [
            'rec_char_dict_path', 'table_char_dict_path', 'layout_dict_path'
        ]:
            action.default = None
    if mMain:
        return parser.parse_args()
    else:
        inference_args_dict = {}
        for action in parser._actions:
            inference_args_dict[action.dest] = action.default
        return argparse.Namespace(**inference_args_dict)


def parse_lang(lang):
    latin_lang = [
        'af', 'az', 'bs', 'cs', 'cy', 'da', 'de', 'es', 'et', 'fr', 'ga', 'hr',
        'hu', 'id', 'is', 'it', 'ku', 'la', 'lt', 'lv', 'mi', 'ms', 'mt', 'nl',
        'no', 'oc', 'pi', 'pl', 'pt', 'ro', 'rs_latin', 'sk', 'sl', 'sq', 'sv',
        'sw', 'tl', 'tr', 'uz', 'vi', 'french', 'german'
    ]
    arabic_lang = ['ar', 'fa', 'ug', 'ur']
    cyrillic_lang = [
        'ru', 'rs_cyrillic', 'be', 'bg', 'uk', 'mn', 'abq', 'ady', 'kbd', 'ava',
        'dar', 'inh', 'che', 'lbe', 'lez', 'tab'
    ]
    devanagari_lang = [
        'hi', 'mr', 'ne', 'bh', 'mai', 'ang', 'bho', 'mah', 'sck', 'new', 'gom',
        'sa', 'bgc'
    ]
    if lang in latin_lang:
        lang = "latin"
    elif lang in arabic_lang:
        lang = "arabic"
    elif lang in cyrillic_lang:
        lang = "cyrillic"
    elif lang in devanagari_lang:
        lang = "devanagari"
    assert lang in MODEL_URLS['OCR'][DEFAULT_OCR_MODEL_VERSION][
        'rec'], 'param lang must in {}, but got {}'.format(
        MODEL_URLS['OCR'][DEFAULT_OCR_MODEL_VERSION]['rec'].keys(), lang)
    if lang == "ch":
        det_lang = "ch"
    elif lang == 'structure':
        det_lang = 'structure'
    elif lang in ["en", "latin"]:
        det_lang = "en"
    else:
        det_lang = "ml"
    return lang, det_lang


def get_model_config(type, version, model_type, lang):
    if type == 'OCR':
        DEFAULT_MODEL_VERSION = DEFAULT_OCR_MODEL_VERSION
    elif type == 'STRUCTURE':
        DEFAULT_MODEL_VERSION = DEFAULT_STRUCTURE_MODEL_VERSION
    else:
        raise NotImplementedError

    model_urls = MODEL_URLS[type]
    if version not in model_urls:
        version = DEFAULT_MODEL_VERSION
    if model_type not in model_urls[version]:
        if model_type in model_urls[DEFAULT_MODEL_VERSION]:
            version = DEFAULT_MODEL_VERSION
        else:
            logger.error('{} models is not support, we only support {}'.format(
                model_type, model_urls[DEFAULT_MODEL_VERSION].keys()))
            sys.exit(-1)

    if lang not in model_urls[version][model_type]:
        if lang in model_urls[DEFAULT_MODEL_VERSION][model_type]:
            version = DEFAULT_MODEL_VERSION
        else:
            logger.error(
                'lang {} is not support, we only support {} for {} models'.
                format(lang, model_urls[DEFAULT_MODEL_VERSION][model_type].keys(
                ), model_type))
            sys.exit(-1)
    return model_urls[version][model_type][lang]


def img_decode(content: bytes):
    np_arr = np.frombuffer(content, dtype=np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)


def check_img(img):
    if isinstance(img, bytes):
        img = img_decode(img)
    if isinstance(img, str):
        # download net image
        if is_link(img):
            download_with_progressbar(img, 'tmp.jpg')
            img = 'tmp.jpg'
        image_file = img
        img, flag_gif, flag_pdf = check_and_read(image_file)
        if not flag_gif and not flag_pdf:
            with open(image_file, 'rb') as f:
                img_str = f.read()
                img = img_decode(img_str)
            if img is None:
                try:
                    buf = BytesIO()
                    image = BytesIO(img_str)
                    im = Image.open(image)
                    rgb = im.convert('RGB')
                    rgb.save(buf, 'jpeg')
                    buf.seek(0)
                    image_bytes = buf.read()
                    data_base64 = str(base64.b64encode(image_bytes),
                                      encoding="utf-8")
                    image_decode = base64.b64decode(data_base64)
                    img_array = np.frombuffer(image_decode, np.uint8)
                    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                except:
                    logger.error("error in loading image:{}".format(image_file))
                    return None
        if img is None:
            logger.error("error in loading image:{}".format(image_file))
            return None
    if isinstance(img, np.ndarray) and len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    return img


class PaddleOCR(predict_system.TextSystem):
    def __init__(self, **kwargs):
        """
        paddleocr package
        args:
            **kwargs: other params show in paddleocr --help
        """
        params = parse_args(mMain=False)
        params.__dict__.update(**kwargs)
        assert params.ocr_version in SUPPORT_OCR_MODEL_VERSION, "ocr_version must in {}, but get {}".format(
            SUPPORT_OCR_MODEL_VERSION, params.ocr_version)
        params.use_gpu = check_gpu(params.use_gpu)

        if not params.show_log:
            logger.setLevel(logging.INFO)
        self.use_angle_cls = params.use_angle_cls
        lang, det_lang = parse_lang(params.lang)

        # init model dir
        # 检测模型配置
        det_model_config = get_model_config('OCR', params.ocr_version, 'det',
                                            det_lang)
        params.det_model_dir, det_url = confirm_model_dir_url(
            params.det_model_dir,
            os.path.join(BASE_DIR, 'whl', 'det', det_lang),
            det_model_config['url'])

        # 识别模型配置
        rec_model_config = get_model_config('OCR', params.ocr_version, 'rec',
                                            lang)
        params.rec_model_dir, rec_url = confirm_model_dir_url(
            params.rec_model_dir,
            os.path.join(BASE_DIR, 'whl', 'rec', lang), rec_model_config['url'])

        # 分类模型配置
        cls_model_config = get_model_config('OCR', params.ocr_version, 'cls',
                                            'ch')
        params.cls_model_dir, cls_url = confirm_model_dir_url(
            params.cls_model_dir,
            os.path.join(BASE_DIR, 'whl', 'cls'), cls_model_config['url'])

        if params.ocr_version in ['PP-OCRv3', 'PP-OCRv4']:
            params.rec_image_shape = "3, 48, 320"
        else:
            params.rec_image_shape = "3, 32, 320"

        # download model if using paddle infer
        # 下载必要的模型文件
        if not params.use_onnx:
            maybe_download(params.det_model_dir, det_url)
            maybe_download(params.rec_model_dir, rec_url)
            maybe_download(params.cls_model_dir, cls_url)

        if params.det_algorithm not in SUPPORT_DET_MODEL:
            logger.error('det_algorithm must in {}'.format(SUPPORT_DET_MODEL))
            sys.exit(0)

        if params.rec_algorithm not in SUPPORT_REC_MODEL:
            logger.error('rec_algorithm must in {}'.format(SUPPORT_REC_MODEL))
            sys.exit(0)

        if params.rec_char_dict_path is None:
            params.rec_char_dict_path = str(
                Path(__file__).parent / rec_model_config['dict_path'])

        logger.debug(params)
        # init det_model and rec_model
        super().__init__(params)
        self.page_num = params.page_num

    def ocr(self,
            img,
            det=True,
            rec=True,
            cls=True,
            bin=False,
            inv=False,
            alpha_color=(255, 255, 255)):
        """
        OCR with PaddleOCR
        args：
            img: img for OCR, support ndarray, img_path and list or ndarray
            det: use text detection or not. If False, only rec will be exec. Default is True
            rec: use text recognition or not. If False, only det will be exec. Default is True
            cls: use angle classifier or not. Default is True. If True, the text with rotation of 180 degrees can be recognized. If no text is rotated by 180 degrees, use cls=False to get better performance. Text with rotation of 90 or 270 degrees can be recognized even if cls=False.
            bin: binarize image to black and white. Default is False.
            inv: invert image colors. Default is False.
            alpha_color: set RGB color Tuple for transparent parts replacement. Default is pure white.
        """
        assert isinstance(img, (np.ndarray, list, str, bytes))
        if isinstance(img, list) and det is True:
            # 检测算法仅支持单一图片
            logger.error('When input a list of images, det must be false')
            exit(0)
        if cls is True and self.use_angle_cls is False:
            logger.warning(
                'Since the angle classifier is not initialized, it will not be used during the forward process'
            )

        # 检查图片
        img = check_img(img)
        # for infer pdf file
        if isinstance(img, list):
            if self.page_num > len(img) or self.page_num == 0:
                self.page_num = len(img)
            imgs = img[:self.page_num]
        else:
            imgs = [img]

        def preprocess_image(_image):
            # 预处理图片
            _image = alpha_to_color(_image, alpha_color)
            if inv:
                # 反转
                _image = cv2.bitwise_not(_image)
            if bin:
                # 二值化
                _image = binarize_img(_image)
            return _image

        if det and rec:
            # 检测+识别
            ocr_res = []
            for idx, img in enumerate(imgs):
                img = preprocess_image(img)
                dt_boxes, rec_res, _ = self.__call__(img, cls)
                if not dt_boxes and not rec_res:
                    ocr_res.append(None)
                    continue
                tmp_res = [[box.tolist(), res]
                           for box, res in zip(dt_boxes, rec_res)]
                ocr_res.append(tmp_res)
            return ocr_res
        elif det and not rec:
            # 检测
            ocr_res = []
            for idx, img in enumerate(imgs):
                img = preprocess_image(img)
                dt_boxes, elapse = self.text_detector(img)
                if dt_boxes is None:
                    ocr_res.append(None)
                    continue
                tmp_res = [box.tolist() for box in dt_boxes]
                ocr_res.append(tmp_res)
            return ocr_res
        else:
            ocr_res = []
            cls_res = []
            for idx, img in enumerate(imgs):
                if not isinstance(img, list):
                    img = preprocess_image(img)
                    img = [img]
                if self.use_angle_cls and cls:
                    img, cls_res_tmp, elapse = self.text_classifier(img)
                    if not rec:
                        cls_res.append(cls_res_tmp)
                rec_res, elapse = self.text_recognizer(img)
                ocr_res.append(rec_res)
            if not rec:
                return cls_res
            return ocr_res


class PPStructure(StructureSystem):
    def __init__(self, **kwargs):
        params = parse_args(mMain=False)
        params.__dict__.update(**kwargs)
        assert params.structure_version in SUPPORT_STRUCTURE_MODEL_VERSION, "structure_version must in {}, but get {}".format(
            SUPPORT_STRUCTURE_MODEL_VERSION, params.structure_version)
        params.use_gpu = check_gpu(params.use_gpu)

        if params.mode is None:
            # 默认模式，支持其他模式：kie
            params.mode = 'structure'

        if not params.show_log:
            logger.setLevel(logging.INFO)
        lang, det_lang = parse_lang(params.lang)
        if lang == 'ch':
            table_lang = 'ch'
        else:
            table_lang = 'en'
        if params.structure_version == 'PP-Structure':
            params.merge_no_span_structure = False

        # init model dir
        det_model_config = get_model_config('OCR', params.ocr_version, 'det',
                                            det_lang)
        params.det_model_dir, det_url = confirm_model_dir_url(
            params.det_model_dir,
            os.path.join(BASE_DIR, 'whl', 'det', det_lang),
            det_model_config['url'])
        rec_model_config = get_model_config('OCR', params.ocr_version, 'rec',
                                            lang)
        params.rec_model_dir, rec_url = confirm_model_dir_url(
            params.rec_model_dir,
            os.path.join(BASE_DIR, 'whl', 'rec', lang), rec_model_config['url'])
        table_model_config = get_model_config(
            'STRUCTURE', params.structure_version, 'table', table_lang)
        params.table_model_dir, table_url = confirm_model_dir_url(
            params.table_model_dir,
            os.path.join(BASE_DIR, 'whl', 'table'), table_model_config['url'])
        layout_model_config = get_model_config(
            'STRUCTURE', params.structure_version, 'layout', lang)
        params.layout_model_dir, layout_url = confirm_model_dir_url(
            params.layout_model_dir,
            os.path.join(BASE_DIR, 'whl', 'layout'), layout_model_config['url'])
        # download model
        maybe_download(params.det_model_dir, det_url)
        maybe_download(params.rec_model_dir, rec_url)
        maybe_download(params.table_model_dir, table_url)
        maybe_download(params.layout_model_dir, layout_url)

        if params.rec_char_dict_path is None:
            params.rec_char_dict_path = str(
                Path(__file__).parent / rec_model_config['dict_path'])
        if params.table_char_dict_path is None:
            params.table_char_dict_path = str(
                Path(__file__).parent / table_model_config['dict_path'])
        if params.layout_dict_path is None:
            params.layout_dict_path = str(
                Path(__file__).parent / layout_model_config['dict_path'])
        logger.debug(params)
        super().__init__(params)

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
        img = check_img(img)
        res, _ = super().__call__(
            img, return_ocr_result_in_table, wl_region_types, bl_region_types, img_idx)
        return res


def main():
    # for cmd
    args = parse_args(mMain=True)
    image_dir = args.image_dir
    if is_link(image_dir):
        download_with_progressbar(image_dir, 'tmp.jpg')
        image_file_list = ['tmp.jpg']
    else:
        image_file_list = get_image_file_list(args.image_dir)
    if len(image_file_list) == 0:
        logger.error('no images find in {}'.format(args.image_dir))
        return
    if args.type == 'ocr':
        engine = PaddleOCR(**(args.__dict__))
    elif args.type == 'structure':
        engine = PPStructure(**(args.__dict__))
    else:
        raise NotImplementedError

    for img_path in image_file_list:
        img_name = os.path.basename(img_path).split('.')[0]
        logger.info('{}{}{}'.format('*' * 10, img_path, '*' * 10))
        if args.type == 'ocr':
            result = engine.ocr(img_path,
                                det=args.det,
                                rec=args.rec,
                                cls=args.use_angle_cls,
                                bin=args.binarize,
                                inv=args.invert,
                                alpha_color=args.alphacolor)
            if result is not None:
                pil_img = Image.open(img_path)
                drawer = ImageDraw(pil_img)
                for idx in range(len(result)):
                    res = result[idx]
                    for line in res:
                        box, (text, conf, char_pos) = line
                        box = np.asarray(box)
                        drawer.rectangle((box[0][0], box[0][1], box[2][0], box[2][1]), outline='red')
                        # 计算文本框的尺寸（宽，高）
                        w = int(
                            max(
                                np.linalg.norm(box[0] - box[1]),
                                np.linalg.norm(box[2] - box[3])))
                        h = int(
                            max(
                                np.linalg.norm(box[0] - box[3]),
                                np.linalg.norm(box[1] - box[2])))

                        left, top = np.min(box[:, 0]), np.min(box[:, 1])

                        # 考虑到文本框存在倾斜，计算每个字符的中心位置
                        if w > h:
                            # 水平方向，计算垂直中心位置 center_y
                            arr_c_y = np.linspace(np.mean([box[0][1], box[3][1]]), np.mean([box[1][1], box[2][1]]),
                                                  len(char_pos))
                        else:
                            # 垂直方向，计算水平中心位置 center_x
                            arr_c_x = np.linspace(np.mean([box[0][0], box[1][0]]), np.mean([box[2][0], box[3][0]]),
                                                  len(char_pos))

                        for i, p in enumerate(char_pos):
                            center_x = left + p if w > h else int(arr_c_x[i])
                            center_y = top + p if h > w else int(arr_c_y[i])

                            radius = 2
                            drawer.ellipse((center_x - radius, center_y - radius, center_x + radius, center_y + radius),
                                           fill="red")
                        logger.info(line)

                pil_img.show()
        elif args.type == 'structure' and args.mode == 'structure':
            img, flag_gif, flag_pdf = check_and_read(img_path)
            if not flag_gif and not flag_pdf:
                img = Image.open(img_path)
                img = np.array(img)

            if args.recovery and args.use_pdf2docx_api and flag_pdf:
                from pdf2docx.converter import Converter
                docx_file = os.path.join(args.output,
                                         '{}.docx'.format(img_name))
                cv = Converter(img_path)
                cv.convert(docx_file)
                cv.close()
                logger.info('docx save to {}'.format(docx_file))
                continue

            if not flag_pdf:
                if img is None:
                    logger.error("error in loading image:{}".format(img_path))
                    continue
                img_paths = [[img_path, img]]
            else:
                img_paths = []
                for index, pdf_img in enumerate(img):
                    os.makedirs(
                        os.path.join(args.output, img_name), exist_ok=True)
                    pdf_img_path = os.path.join(
                        args.output, img_name,
                        img_name + '_' + str(index) + '.jpg')
                    cv2.imwrite(pdf_img_path, pdf_img)
                    img_paths.append([pdf_img_path, pdf_img])

            all_res = []
            for index, (new_img_path, img) in enumerate(img_paths):
                logger.info('processing {}/{} page:'.format(index + 1,
                                                            len(img_paths)))
                # 表格识别结果返回ocr结果
                result = engine(img, return_ocr_result_in_table=True,
                                wl_region_types=['table'],
                                img_idx=index)
                save_structure_res(result, args.output, img_name, index)

                # 可视化识别结果
                pil_img = Image.open(new_img_path)
                drawer = ImageDraw(pil_img)

                from ppstructure.table.matcher import TableMatch
                matcher = TableMatch(filter_ocr_result=True)
                for i, region in enumerate(result):
                    region_bbox = region["bbox"]
                    drawer.rectangle(region_bbox, outline="black")
                    drawer.text((region_bbox[2], region_bbox[1]), region["type"], fill="blue")

                    if region["type"] == "table":
                        cell_bbox = region["res"]["cell_bbox"]
                        rec_bbox = region["res"]["boxes"]
                        rec_res = region["res"]["rec_res"]

                        print('html', region["res"]['html'])

                        # 截取表格区域并保存
                        save_crop_image(img, region_bbox,
                                        os.path.join(args.output, 'table'),
                                        "{}_{}.jpg".format(img_name, time.time_ns()))

                        # 文本框配备到对应的表格单元格
                        matched_res = matcher.match_result(rec_bbox, cell_bbox)

                        for j, bbox in enumerate(cell_bbox):
                            # 绘制单元格边框
                            bbox = np.array(bbox).reshape((-1, 2))
                            drawer.polygon([(x, y) for x, y in bbox], outline="blue")
                            drawer.text(bbox[0], f'cell-{j}', fill="blue")

                            if j not in matched_res:
                                # 单元格中没有内容
                                continue

                            for text_bbox, text, rec_idx in [(rec_bbox[idx], rec_res[idx], idx) for idx in
                                                             matched_res[j]]:
                                text_pts = np.array(text_bbox).reshape((-1, 2))
                                drawer.rectangle(tuple(text_pts.flatten()), outline="red")
                                drawer.text(tuple(text_pts[0]), f"{i}_{rec_idx}_{j}", fill="red")
                                print(f"cell {i}_{rec_idx}_{j}", text[0], text[1])
                    else:
                        for j, rec_res in enumerate(region["res"]):
                            text, conf, bbox = rec_res["text"], rec_res["confidence"], rec_res["text_region"]
                            drawer.polygon([(x, y) for x, y in bbox], outline="red")
                            drawer.text((bbox[0][0], bbox[0][1]), f"{i}_{j}", fill="red")
                            print(f"plain {i}_{j}", text, conf)

                pil_img.show()

                if args.recovery and result != []:
                    from copy import deepcopy
                    from ppstructure.recovery.recovery_to_doc import sorted_layout_boxes
                    h, w, _ = img.shape
                    result_cp = deepcopy(result)
                    result_sorted = sorted_layout_boxes(result_cp, w)
                    all_res += result_sorted
            if args.recovery and all_res != []:
                # 将识别结果进行版面恢复，保存为Word档案
                try:
                    from ppstructure.recovery.recovery_to_doc import convert_info_docx
                    convert_info_docx(img, all_res, args.output, img_name)
                except Exception as ex:
                    logger.error(
                        "error in layout recovery image:{}, err msg: {}".format(
                            img_name, ex))
                    continue

            for item in all_res:
                item.pop('img')
                item.pop('res')
                logger.info(item)
            logger.info('result save to {}'.format(args.output))

        elif args.type == 'structure' and args.mode == 'kie':
            # KIE（SER / SER + RE）
            result = engine(img_path)
            img = cv2.imread(img_path)

            if engine.kie_predictor.predictor is not None:
                img_save_path = os.path.join(args.output,
                                             '{}_ser_re.jpg'.format(img_name))
                draw_img = draw_re_results(
                    img, result, font_path=args.vis_font_path)
            else:
                img_save_path = os.path.join(args.output,
                                             '{}_ser.jpg'.format(img_name))
                draw_img = draw_ser_results(
                    img, result, font_path=args.vis_font_path)

            if result:
                cv2.imwrite(img_save_path, draw_img)
                logger.info('result save to {}'.format(img_save_path))

                save_kie_res(result, args.output, img_name, 0)

            # 可视化识别结果
            pil_img = Image.fromarray(draw_img)
            pil_img.show()


if __name__ == '__main__':
    main()
