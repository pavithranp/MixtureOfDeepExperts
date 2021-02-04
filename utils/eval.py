import numpy as np
from math import sqrt, isnan

def read_gt():
    return eval(open('docs/gt_dict.txt', 'r').read())
def intersection_dist(px, py, qx, qy, rx, ry, dx, dy):
    l = (dy * (rx - qx) + dx * (qy - ry)) / (dy * (px - qx) + dx * (qy - py))
    m = (py * (rx - qx) + px * (qy - ry) + qx * ry - qy * rx) / (dy * (px - qx) + dx * (qy - py))
    if isnan(l) or isnan(m):
        return 0.0
    elif 0 <= l <= 1 and m > 0:
        return m / sqrt(dx * dx + dy * dy) * 2
    else:
        return 0.0

def frange(start, stop, step):
    x = start
    while x < stop:
        yield x
        x += step

def compute_mAP(precisionList, recallList):
    # compute average precision
    ap = 0
    p = 0
    precs = np.array(precisionList)
    recs = np.array(recallList)
    for t in frange(0, 1.0, 0.1):
        p = 0
        if any(precs[np.where(recs >= t)]):
            p = max(precs[np.where(recs >= t)])
        ap = ap + p / 11
    return ap

def voc_ap(rec, prec, use_07_metric=False):
    """Compute VOC AP given precision and recall. If use_07_metric is true, uses
    the VOC 07 11-point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def compute_mAP(precisionList, recallList):
    # compute average precision
    ap = 0
    p = 0
    precs = np.array(precisionList)
    recs = np.array(recallList)
    for t in frange(0, 1.0, 0.1):
        p = 0
        if any(precs[np.where(recs >= t)]):
            p = max(precs[np.where(recs >= t)])
        ap = ap + p / 11
    return ap

def f1_score(precision, recall):
    return 2 * ((precision* recall)/(precision+recall))
