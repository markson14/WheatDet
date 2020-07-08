# A modification version from chainercv repository.
# (See https://github.com/chainer/chainercv/blob/master/chainercv/evaluations/eval_detection_voc.py)
from __future__ import division

import os
from collections import defaultdict
import numpy as np
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.utils.visualize import vis_image
from PIL import Image

tag = 1


def display_image(pred_bbox_l, gt_boxlist, gt_bbox_l, pred_score_l):
    impath = gt_boxlist.get_field("ID")
    img = Image.open(impath)
    print(impath)
    pred_bbox_l = pred_bbox_l[pred_score_l > 0.7]
    pil_image = vis_image(img, pred_bbox_l, mode=2, line_width=3)
    pil_image = vis_image(pil_image, gt_bbox_l, mode=0, line_width=3)
    # print("pred box_list:\n", pred_bbox_l.astype(np.float16))
    # print("pred box_list score:\n", pred_score_l.astype(np.float16))
    # print("gt box_list:\n", gt_bbox_l)
    width, height = pil_image.size
    print(f'image size: ({width}, {height})')
    if 'Hisence' in impath:
        newsize = (width // 5, height // 5)
    else:
        newsize = (width, height)
    pil_image.resize(newsize).save('./test_pred.png')


def coco_metric(ap, iouThr, result_dic, iou_thresh_list):
    iStr = ' {:<18} {:<6} @[ IoU={:<9} | area={:>4s}] = {:0.4f}\n'
    if ap == 0:
        titleStr = 'IoU'
        typeStr = '(IoU)'
    elif ap == 1:
        titleStr = 'Average Precision'
        typeStr = '(AP)'
    elif ap == 2:
        titleStr = 'Recall'
        typeStr = '(R)'
    elif ap == 3:
        titleStr = 'F1 Score'
        typeStr = '(F1)'
    elif ap == 4:
        titleStr = 'F2 Score'
        typeStr = '(F2)'
    elif ap == 5:
        titleStr = 'Precision'
        typeStr = '(P)'

    iouStr = '{:0.2f}:{:0.2f}'.format(iou_thresh_list[0], iou_thresh_list[-1]) \
        if iouThr is None else '{:0.2f}'.format(iouThr)

    if not iouThr:
        if ap == 1:
            ap_list = []
            for i, v in result_dic.items():
                ap_list.append(v['map'])
            result = np.mean(ap_list)
        elif ap == 0:
            iou_list = []
            for i, v in result_dic.items():
                iou_list.append(v['iou'])
            result = np.mean(iou_list)
        elif ap == 2:
            recall_list = []
            for i, v in result_dic.items():
                recall_list.append(v['recall'])
            result = np.nanmean(recall_list)
        elif ap == 3:
            f1_list = []
            for i, v in result_dic.items():
                f1_list.append(v['f1'])
            result = np.nanmean(f1_list)
        elif ap == 4:
            f2_list = []
            for i, v in result_dic.items():
                f2_list.append(v['f2'])
            result = np.nanmean(f2_list)
        elif ap == 5:
            prec_list = []
            for i, v in result_dic.items():
                prec_list.append(v['precision'])
            result = np.nanmean(prec_list)
    else:
        if ap == 1:
            result = result_dic[iouThr]['map']
        elif ap == 0:
            result = np.mean(result_dic[iouThr]['iou'])
        elif ap == 2:
            result = np.nanmean(result_dic[iouThr]['recall'])
        elif ap == 3:
            result = np.nanmean(result_dic[iouThr]['f1'])
        elif ap == 4:
            result = np.nanmean(result_dic[iouThr]['f2'])
        elif ap == 5:
            result = np.nanmean(result_dic[iouThr]['precision'])

    return iStr.format(titleStr, typeStr, iouStr, 'all', result)


def do_voc_evaluation(dataset, predictions, output_folder, logger):
    # TODO need to make the use_07_metric format available
    # for the user to choose

    # AP@IOU defines here!!!!
    iou_thresh_list = [i / 100 for i in range(50, 100, 5)]

    pred_boxlists = []
    gt_boxlists = []
    for image_id, prediction in enumerate(predictions):
        img_info = dataset.get_img_info(image_id)
        if len(prediction) == 0:
            continue
        image_width = img_info["width"]
        image_height = img_info["height"]
        prediction = prediction.resize((image_width, image_height))
        pred_boxlists.append(prediction)
        gt_boxlist = dataset.get_groundtruth(image_id)
        gt_boxlists.append(gt_boxlist)

    result_list = []
    result_dic = {}
    result_str = ""
    mAP = []
    mmIoU = []

    global tag
    tag = 1

    for iou_thresh in iou_thresh_list:
        result = eval_detection_voc(
            pred_boxlists=pred_boxlists,
            gt_boxlists=gt_boxlists,
            iou_thresh=iou_thresh,
            use_07_metric=True,
        )
        # mAP.append(result["map"])
        # mmIoU.append(np.mean(result["iou"]))
        # if iou_thresh == 0.5 or iou_thresh == 0.75:
        #     result_str += "\nAverage Precision@{}: {:.4f}\n".format(iou_thresh, result["map"])
        #     result_str += "mIoU: {:.4f}\n".format(np.mean(result["iou"]))
        #     for i, ap in enumerate(result["ap"]):
        #         if i == 0:  # skip background
        #             continue
        #         result_str += "{}: {:.4f}\n".format(
        #             dataset.map_class_id_to_class_name(i), ap
        #         )
        # result_list.append(result)
        result_dic[iou_thresh] = result
    # result_str += "\nmAP: {:.4f}\n".format(np.mean(mAP))
    # result_str += "mAIoU: {:.4f}\n".format(np.mean(mmIoU))

    result_str += coco_metric(1, 0.5, result_dic, iou_thresh_list)
    result_str += coco_metric(1, 0.75, result_dic, iou_thresh_list)
    result_str += coco_metric(1, None, result_dic, iou_thresh_list)
    result_str += "\n"
    result_str += coco_metric(0, None, result_dic, iou_thresh_list)
    result_str += coco_metric(2, None, result_dic, iou_thresh_list)
    result_str += coco_metric(5, None, result_dic, iou_thresh_list)
    result_str += coco_metric(3, None, result_dic, iou_thresh_list)
    result_str += coco_metric(4, None, result_dic, iou_thresh_list)

    logger.info('\n' + result_str)
    if output_folder:
        with open(os.path.join(output_folder, "result.txt"), "w") as fid:
            fid.write(result_str)
    # return result_list


def f_score(recall, precision, beta=1):
    return (1+beta**2) * recall * precision / (beta**2 * precision + recall)


def eval_detection_voc(pred_boxlists, gt_boxlists, iou_thresh=0.75, use_07_metric=False):
    """Evaluate on voc dataset.
    Args:
        pred_boxlists(list[BoxList]): pred boxlist, has labels and scores fields.
        gt_boxlists(list[BoxList]): ground truth boxlist, has labels field.
        iou_thresh: iou thresh
        use_07_metric: boolean
    Returns:
        dict represents the results
    """
    assert len(gt_boxlists) == len(
        pred_boxlists
    ), "Length of gt and pred lists need to be same."
    prec, rec, res_iou, r, p, f1, f2 = calc_detection_voc_prec_rec(
        pred_boxlists=pred_boxlists, gt_boxlists=gt_boxlists, iou_thresh=iou_thresh
    )
    ap = calc_detection_voc_ap(prec, rec, use_07_metric=use_07_metric)
    return {"ap": ap, "map": np.nanmean(ap),
            "iou": res_iou, "recall": r, "precision": p,
            "f1": f1, "f2": f2}


def calc_detection_voc_prec_rec(gt_boxlists, pred_boxlists, iou_thresh=0.5):
    """Calculate precision and recall based on evaluation code of PASCAL VOC.
    This function calculates precision and recall of
    predicted bounding boxes obtained from a dataset which has :math:`N`
    images.
    The code is based on the evaluation code used in PASCAL VOC Challenge.
   """
    n_pos = defaultdict(int)
    score = defaultdict(list)
    match = defaultdict(list)
    res_iou = list()
    for gt_boxlist, pred_boxlist in zip(gt_boxlists, pred_boxlists):
        pred_bbox = pred_boxlist.bbox.numpy()
        pred_label = pred_boxlist.get_field("labels").numpy()
        pred_score = pred_boxlist.get_field("scores").numpy()
        gt_bbox = gt_boxlist.bbox.numpy()
        gt_label = gt_boxlist.get_field("labels").numpy()
        gt_difficult = gt_boxlist.get_field("difficult").numpy()

        for l in np.unique(np.concatenate((pred_label, gt_label)).astype(int)):
            pred_mask_l = pred_label == l
            pred_bbox_l = pred_bbox[pred_mask_l]
            pred_score_l = pred_score[pred_mask_l]
            # sort by score
            order = pred_score_l.argsort()[::-1]
            pred_bbox_l = pred_bbox_l[order]
            pred_score_l = pred_score_l[order]

            gt_mask_l = gt_label == l
            gt_bbox_l = gt_bbox[gt_mask_l]
            gt_difficult_l = gt_difficult[gt_mask_l]

            n_pos[l] += np.logical_not(gt_difficult_l).sum()
            score[l].extend(pred_score_l)

            if len(pred_bbox_l) == 0:
                continue
            if len(gt_bbox_l) == 0:
                match[l].extend((0,) * pred_bbox_l.shape[0])
                continue

            # VOC evaluation follows integer typed bounding boxes.
            pred_bbox_l = pred_bbox_l.copy()
            pred_bbox_l[:, 2:] += 1
            gt_bbox_l = gt_bbox_l.copy()
            gt_bbox_l[:, 2:] += 1
            iou = boxlist_iou(
                BoxList(pred_bbox_l, gt_boxlist.size),
                BoxList(gt_bbox_l, gt_boxlist.size),
            ).numpy()
            f_iou = iou.max(axis=0)
            res_iou.extend(f_iou)
            
            # display intermediate result
            global tag
            if tag == 1:
                # display_image(pred_bbox_l, gt_boxlist, gt_bbox_l, pred_score_l)
                tag = 0
                print("Done save intermediate result")
                
            gt_index = iou.argmax(axis=1)
            # set -1 if there is no matching ground truth
            gt_index[iou.max(axis=1) < iou_thresh] = -1
            del iou

            selec = np.zeros(gt_bbox_l.shape[0], dtype=bool)
            for gt_idx in gt_index:
                if gt_idx >= 0:
                    if gt_difficult_l[gt_idx]:
                        match[l].append(-1)
                    else:
                        if not selec[gt_idx]:
                            match[l].append(1)
                        else:
                            match[l].append(0)
                    selec[gt_idx] = True
                else:
                    match[l].append(0)

    n_fg_class = max(n_pos.keys()) + 1
    prec = [None] * n_fg_class
    rec = [None] * n_fg_class

    p = [np.nan] * n_fg_class
    r = [np.nan] * n_fg_class
    f1 = [np.nan] * n_fg_class
    f2 = [np.nan] * n_fg_class

    for l in n_pos.keys():
        score_l = np.array(score[l])
        match_l = np.array(match[l], dtype=np.int8)

        order = score_l.argsort()[::-1]
        match_l = match_l[order]

        tp = np.cumsum(match_l == 1)
        fp = np.cumsum(match_l == 0)

        # If an element of fp + tp is 0,
        # the corresponding element of prec[l] is nan.
        prec[l] = tp / (fp + tp)
        # If n_pos[l] is 0, rec[l] is None.
        if n_pos[l] > 0:
            rec[l] = tp / n_pos[l]

        # Overall true positive and false positive
        tp_i = np.sum(match_l == 1)
        fp_i = np.sum(match_l == 0)

        r[l] = tp_i / n_pos[l]
        p[l] = tp_i / (fp_i + tp_i)

        f1[l] = f_score(r[l], p[l], 1)
        f2[l] = f_score(r[l], p[l], 2)

    return prec, rec, res_iou, r, p, f1, f2


def calc_detection_voc_ap(prec, rec, use_07_metric=False):
    """Calculate average precisions based on evaluation code of PASCAL VOC.
    This function calculates average precisions
    from given precisions and recalls.
    The code is based on the evaluation code used in PASCAL VOC Challenge.
    Args:
        prec (list of numpy.array): A list of arrays.
            :obj:`prec[l]` indicates precision for class :math:`l`.
            If :obj:`prec[l]` is :obj:`None`, this function returns
            :obj:`numpy.nan` for class :math:`l`.
        rec (list of numpy.array): A list of arrays.
            :obj:`rec[l]` indicates recall for class :math:`l`.
            If :obj:`rec[l]` is :obj:`None`, this function returns
            :obj:`numpy.nan` for class :math:`l`.
        use_07_metric (bool): Whether to use PASCAL VOC 2007 evaluation metric
            for calculating average precision. The default value is
            :obj:`False`.
    Returns:
        ~numpy.ndarray:
        This function returns an array of average precisions.
        The :math:`l`-th value corresponds to the average precision
        for class :math:`l`. If :obj:`prec[l]` or :obj:`rec[l]` is
        :obj:`None`, the corresponding value is set to :obj:`numpy.nan`.
    """

    n_fg_class = len(prec)
    ap = np.empty(n_fg_class)
    for l in range(n_fg_class):
        if prec[l] is None or rec[l] is None:
            ap[l] = np.nan
            continue

        if use_07_metric:
            # 11 point metric
            ap[l] = 0
            for t in np.arange(0.0, 1.1, 0.1):
                if np.sum(rec[l] >= t) == 0:
                    p = 0
                else:
                    p = np.max(np.nan_to_num(prec[l])[rec[l] >= t])
                ap[l] += p / 11
        else:
            # correct AP calculation
            # first append sentinel values at the end
            mpre = np.concatenate(([0], np.nan_to_num(prec[l]), [0]))
            mrec = np.concatenate(([0], rec[l], [1]))

            mpre = np.maximum.accumulate(mpre[::-1])[::-1]

            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]

            # and sum (\Delta recall) * prec
            ap[l] = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap
