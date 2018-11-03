# -*- coding: utf-8 -*-
# File: eval.py

import tqdm
import os
from collections import namedtuple
from contextlib import ExitStack
import numpy as np
import cv2

from tensorpack.utils.utils import get_tqdm_kwargs

#from pycocotools.coco import COCO
#from pycocotools.cocoeval import COCOeval
#import pycocotools.mask as cocomask

from coco import COCOMeta
from common import CustomResize, clip_boxes
from config import config as cfg
import pickle

import _init_paths
from _utils import *
from BoundingBox import BoundingBox
from BoundingBoxes import BoundingBoxes
from Evaluator import Evaluator
from VOC_parse import getBBoxes, _findNode
import xml.etree.ElementTree as ET

DetectionResult = namedtuple(
    'DetectionResult',
    ['box', 'score', 'class_id', 'mask'])
"""
box: 4 float
score: float
class_id: int, 1~NUM_CLASS
mask: None, or a binary image of the original image shape
"""
with open('mapper.pickle', 'rb') as f:
    mapper = pickle.load(f)

def fill_full_mask(box, mask, shape):
    """
    Args:
        box: 4 float
        mask: MxM floats
        shape: h,w
    """
    # int() is floor
    # box fpcoor=0.0 -> intcoor=0.0
    x0, y0 = list(map(int, box[:2] + 0.5))
    # box fpcoor=h -> intcoor=h-1, inclusive
    x1, y1 = list(map(int, box[2:] - 0.5))    # inclusive
    x1 = max(x0, x1)    # require at least 1x1
    y1 = max(y0, y1)

    w = x1 + 1 - x0
    h = y1 + 1 - y0

    # rounding errors could happen here, because masks were not originally computed for this shape.
    # but it's hard to do better, because the network does not know the "original" scale
    mask = (cv2.resize(mask, (w, h)) > 0.5).astype('uint8')
    ret = np.zeros(shape, dtype='uint8')
    ret[y0:y1 + 1, x0:x1 + 1] = mask
    return ret


def detect_one_image(img, model_func):
    """
    Run detection on one image, using the TF callable.
    This function should handle the preprocessing internally.

    Args:
        img: an image
        model_func: a callable from TF model,
            takes image and returns (boxes, probs, labels, [masks])

    Returns:
        [DetectionResult]
    """

    orig_shape = img.shape[:2]
    resizer = CustomResize(cfg.PREPROC.SHORT_EDGE_SIZE, cfg.PREPROC.MAX_SIZE)
    resized_img = resizer.augment(img)
    scale = np.sqrt(resized_img.shape[0] * 1.0 / img.shape[0] * resized_img.shape[1] / img.shape[1])
    boxes, probs, labels, *masks = model_func(resized_img)
    boxes = boxes / scale
    # boxes are already clipped inside the graph, but after the floating point scaling, this may not be true any more.
    boxes = clip_boxes(boxes, orig_shape)

    if masks:
        # has mask
        full_masks = [fill_full_mask(box, mask, orig_shape)
                      for box, mask in zip(boxes, masks[0])]
        masks = full_masks
    else:
        # fill with none
        masks = [None] * len(boxes)

    results = [DetectionResult(*args) for args in zip(boxes, probs, labels, masks)]
    return results


def eval_coco(df, detect_func, tqdm_bar=None):
    """
    Args:
        df: a DataFlow which produces (image, image_id)
        detect_func: a callable, takes [image] and returns [DetectionResult]
        tqdm_bar: a tqdm object to be shared among multiple evaluation instances. If None,
            will create a new one.

    Returns:
        list of dict, to be dumped to COCO json format
    """
    df.reset_state()
    all_results = []
    # tqdm is not quite thread-safe: https://github.com/tqdm/tqdm/issues/323
    with ExitStack() as stack:
        if tqdm_bar is None:
            tqdm_bar = stack.enter_context(
                tqdm.tqdm(total=df.size(), **get_tqdm_kwargs()))
        for img, img_id in df.get_data():
            results = detect_func(img)
            for r in results:
                box = r.box
                cat_id = mapper.get(r.class_id, 'background')#COCOMeta.class_id_to_category_id[r.class_id]
                # this two lines are coco format(xywh)
                # ignore
                # box[2] -= box[0]
                # box[3] -= box[1]

                res = {
                    'image_id': img_id,
                    'category_id': cat_id,
                    'bbox': list(map(lambda x: round(float(x), 2), box)),
                    'score': round(float(r.score), 3),
                }

                # also append segmentation to results
                if r.mask is not None:
                    rle = cocomask.encode(
                        np.array(r.mask[:, :, None], order='F'))[0]
                    rle['counts'] = rle['counts'].decode('ascii')
                    res['segmentation'] = rle
                all_results.append(res)
            tqdm_bar.update(1)
    return all_results


# https://github.com/pdollar/coco/blob/master/PythonAPI/pycocoEvalDemo.ipynb
def print_evaluation_scores(json_file):
    ret = {}
    assert cfg.DATA.BASEDIR and os.path.isdir(cfg.DATA.BASEDIR)
    annofile = os.path.join(
        cfg.DATA.BASEDIR, 'annotations',
        'instances_{}.json'.format(cfg.DATA.VAL))
    coco = COCO(annofile)
    cocoDt = coco.loadRes(json_file)
    cocoEval = COCOeval(coco, cocoDt, 'bbox')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    fields = ['IoU=0.5:0.95', 'IoU=0.5', 'IoU=0.75', 'small', 'medium', 'large']
    for k in range(6):
        ret['mAP(bbox)/' + fields[k]] = cocoEval.stats[k]

    if cfg.MODE_MASK:
        cocoEval = COCOeval(coco, cocoDt, 'segm')
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        for k in range(6):
            ret['mAP(segm)/' + fields[k]] = cocoEval.stats[k]
    return ret

def parseBBoxes(element):
    truncated = _findNode(element, 'truncated', parse=int)
    difficult = _findNode(element, 'difficult', parse=int)
    class_name = _findNode(element, 'name').text

    box = np.zeros((1, 4))
    bndbox = _findNode(element, 'bndbox')
    box[0, 0] = _findNode(bndbox, 'xmin', 'bndbox.xmin', parse=float) - 1
    box[0, 1] = _findNode(bndbox, 'ymin', 'bndbox.ymin', parse=float) - 1
    box[0, 2] = _findNode(bndbox, 'xmax', 'bndbox.xmax', parse=float) - 1
    box[0, 3] = _findNode(bndbox, 'ymax', 'bndbox.ymax', parse=float) - 1
    return box, class_name, difficult

def createEvalBBoxes(imgNames):
    gtBBoxes = BoundingBoxes()
    for name in imgNames:
        path = os.path.join('../VOCdevkit/VOC2007/Annotations', name+'.xml')
        root = ET.parse(path).getroot()
        height = _findNode(root.find('size'), 'height', parse=int)
        width = _findNode(root.find('size'), 'width', parse=int)
        for element in root.iter('object'):
            box, class_name, difficult = parseBBoxes(element)
            if difficult:
                continue
            gtBBoxes.addBoundingBox(BoundingBox(imageName=name, classId=class_name,
                                     x=box[0, 0], y=box[0, 1], w=box[0, 2], h=box[0, 3], typeCoordinates=CoordinatesType.Absolute,
                                     bbType=BBType.GroundTruth, format=BBFormat.XYX2Y2, imgSize=(width,height)))
    with open('validationBBoxes.pickle', 'wb') as f:
        pickle.dump(gtBBoxes, f)
    print('create EvalBBoxes over!')

def print_evaluation_scores_voc(output_file):
    import os
    import copy
    if not os.path.exists('./validationBBoxes.pickle'):
        filelist = []
        for file in output_file:
            fileName = file['image_id']
            if not fileName in filelist:
                filelist.append(fileName)
        createEvalBBoxes(filelist)
    with open('validationBBoxes.pickle', 'rb') as f:
        gtBBoxes = pickle.load(f)
    # create eval bboxes
    evalBBoxes = copy.deepcopy(gtBBoxes)
    for detBBoxes in output_file:
        box = np.round(detBBoxes['bbox'])
        evalBBoxes.addBoundingBox(BoundingBox(imageName=detBBoxes['image_id'], classId=detBBoxes['category_id'], \
                                              classConfidence=detBBoxes['score'], \
                                              x=box[0], y=box[1], \
                                              w=box[2], h=box[3], \
                                              typeCoordinates=CoordinatesType.Absolute,\
                                              bbType=BBType.Detected, format=BBFormat.XYX2Y2, \
                                              ))
    evaluator = Evaluator()
    metricsPerClass = evaluator.GetPascalVOCMetrics(evalBBoxes, IOUThreshold=0.5)
    print("Average precision values per class:\n")
    # Loop through classes to obtain their metrics
    # reconstruct for monitors
    ret = {}
    for mc in metricsPerClass:
        # Get metric values per each class
        c = mc['class']
        precision = mc['precision']
        recall = mc['recall']
        average_precision = mc['AP']
        ipre = mc['interpolated precision']
        irec = mc['interpolated recall']
        # Print AP per class
        ret[c] = average_precision
        print('%s: %f' % (c, average_precision))
    return ret

if __name__ == '__main__':
    import json
    with open('output1.json', 'r') as f:
        all_results = json.load(f)
    print_evaluation_scores_voc(all_results)