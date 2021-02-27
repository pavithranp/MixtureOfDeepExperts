import numpy as np
from math import sqrt, isnan
import math,operator
SOFTMAX_THRESHOLD = 0.5

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

def evaluate(sorted_dict, groundtruth, number_of_groundtruth_boxes, threshold=0.6):
    print('IoU-Threshold,\t{0}'.format(threshold))
    print('Softmax-Threshold,\t{0}'.format(SOFTMAX_THRESHOLD))
    tp = 0
    fp = 0
    fn = 0
    precisionList = []
    recallList = []
    true_positive_difference = []
    iou_true_positives = []

    for indice, obj in enumerate(sorted_dict):
        if len(obj) == 6:
            score, img_name, xmin, ymin, xmax, ymax = obj
        elif len(obj) == 8:
            score, img_name, xmin, ymin, xmax, ymax, gating_factor_1, gating_factor_2 = obj
        else:
            print('len(obj) not known')
            break

        # testBB = [int(xmin)/2, int(ymin)/2, int(xmax)/2, int(ymax)/2]
        testBB = [int(xmin), int(ymin), int(xmax), int(ymax)]
        if img_name in groundtruth:
            # get the groundtruth boxes
            gtBB = groundtruth[img_name]
            # val_old = -1
            for i in range(0, len(gtBB)):
                val = bb_intersection_over_union(gtBB[i], testBB)
                # val_old = interUnio(gtBB[i], testBB)

                # if val != val_old:
                #     val_diff.append(val-val_old)
                if val >= threshold:
                    tp = tp + 1
                    true_positive_difference.append(
                        np.array(gtBB[i]) - np.array(testBB)
                    )
                    iou_true_positives.append(val)

                    # remove the box as we have found it
                    del gtBB[i]
                    break

            else:  # for loop fell through
                fp = fp + 1
            if (tp + fp) == 0:  # if first matched person is occluded
                print('hack, fixme, first matched person is occluded')
                continue
        else:
            fp = fp + 1
        precisionList.append(float(tp) / float((tp + fp)))
        recallList.append(float(tp) / float(number_of_groundtruth_boxes))

    print("tp,\t{0}".format(tp))
    print("fp,\t{0}".format(fp))
    print("fn,\t{0}".format(fn))
    print("precision,\t{0}".format(precisionList[-1]))
    print("recall,\t{0}".format(recallList[-1]))
    print("f1 score,\t{0}".format(f1_score(precisionList[-1], recallList[-1])))
    print('avg iou tp,\t{0}'.format(np.mean(iou_true_positives)))


    eer_x = None
    eer_y = None
    for i in range(0, len(precisionList) - 1):
        dist = intersection_dist(recallList[i], precisionList[i] + 0.000001, recallList[i + 1],
                                     precisionList[i + 1], 0.0, 0.0, 1.0, 1.0)
        if dist:
            eer_x = math.sin(np.pi / 4) * dist
            eer_y = math.cos(np.pi / 4) * dist
            print('EER:, {0}, {1}'.format(eer_x, eer_y))
            break
    else:
        print('EER:, computation did not find an intersection')

    ap_voc_2007 = voc_ap(np.asarray(recallList), np.asarray(precisionList), True)
    ap_voc_2010 = voc_ap(np.asarray(recallList), np.asarray(precisionList), False)
    m_ap = compute_mAP(precisionList, recallList)
    print('Medium average Precision (Legacy),\t{0}'.format(m_ap))
    # Recall at AP value, we get the index of the element
    recall_map_voc_2007 = min(range(len(precisionList)), key=lambda i: abs(precisionList[i] - ap_voc_2007))
    recall_map_voc_2010 = min(range(len(precisionList)), key=lambda i: abs(precisionList[i] - ap_voc_2010))
    print('Medium average Precision (VOC2007),\t{0}'.format(ap_voc_2007))
    print('Recall at maP (VOC2007),\t{0}'.format(recallList[recall_map_voc_2007]))
    print('Medium average Precision (VOC2010),\t{0}'.format(ap_voc_2010))
    print('Recall at maP (VOC2010),\t{0}'.format(recallList[recall_map_voc_2010]))


def readAndSortBBs(prediction,gt,sortedListTestBB):
    # global numOfGTBBs
    groundtruth_boxes = gt
    lines_boxes = prediction
    imgsNoPerson = []

    for res in lines_boxes.items():
        imgName = res[0]
        bboxes = res[1][0]
        # gating_factor_1, gating_factor_2 = res[2][0], res[2][1]

        # imgName = splittedLine[0][:-16] # for rgb -4, else 16, 0 for rcnn
        if imgName not in groundtruth_boxes:
            print('No person annotation in: ', imgName)
            imgsNoPerson.append(imgName)

        if len(bboxes) == 0:
            # noDetectionList.append(imgName)
            break
        for i in range(0, len(bboxes)):
            tmp_softmax_value = float(res[1][1][i])
            if tmp_softmax_value > SOFTMAX_THRESHOLD:
                tmp_entry = [str(0.0), imgName, bboxes[i][0], bboxes[i][1], bboxes[i][2], bboxes[i][3]]
                # This normalization is only required as we compute our bounding boxes on the full-hd resolution
                # tmp_entry[2] = str(int(int(tmp_entry[2]) / 2.))
                # tmp_entry[3] = str(int(int(tmp_entry[3]) / 2.))
                # tmp_entry[4] = str(int(int(tmp_entry[4]) / 2.))
                # tmp_entry[5] = str(int(int(tmp_entry[5]) / 2.))

                # tmp_entry += [res[2][0], res[2][1]]

                if tmp_softmax_value >= 0.01:
                    tmp_entry[0] = tmp_softmax_value
                sortedListTestBB.append(tuple(tmp_entry))

    return sorted(sortedListTestBB, key=operator.itemgetter(0), reverse=True), imgsNoPerson

def get_dicts(d,output):
    # bboxes = output['instances'].

    bboxes = [[y for y in x] for x in output['instances'].pred_boxes.tensor.cpu().numpy()]
    scores = [x for x in output['instances'].scores.cpu().numpy()]
    return [d,bboxes,scores]