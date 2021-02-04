# from tqdm import tqdm
import math,os
from dataloader.single_loader import SingleDataset
import detectron2.data.transforms as T
import operator
import cv2
import numpy as np

SOFTMAX_THRESHOLD = 0.5
from network.GatingNetwork import  GatingNetwork
from detectron2.config import get_cfg
from detectron2 import model_zoo
import torch ,pickle
from utils.eval import bb_intersection_over_union, f1_score, voc_ap, compute_mAP, intersection_dist,read_gt


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


def readAndSortBBs(prediction,gt):
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

def image_process(path1,path2,cfg):
    aug = T.ResizeShortestEdge([cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST)
    image1 = cv2.imread(path1)
    image1 = aug.get_transform(image1).apply_image(image1)
    image1 = torch.as_tensor(image1.astype("float32").transpose(2, 0, 1))
    # image1 = cv2.resize(image1,(1920,1080))
    image2 = cv2.imread(path2)
    # image2 = cv2.resize(image2, (1920,1080))
    height, width = image2.shape[:2]

    image2 = aug.get_transform(image2).apply_image(image2)
    image2 = torch.as_tensor(image2.astype("float32").transpose(2, 0, 1))
    # image1 = torch.as_tensor(image1.transpose(2, 0, 1), dtype=torch.float32)
    # image2 = torch.as_tensor(image2.transpose(2, 0, 1), dtype=torch.float32)

    return [{"rgb_image": image1,"depth_image": image2, "height": height, "width": width}]



if __name__ =="__main__":
    sortedListTestBB = []
    val_dataset = SingleDataset(root='/mnt/AAB281B7B2818911/datasets/InOutDoorPeopleRGBD', dataset='DepthJetQhd',
                                set='val', grad=True)
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_C4_3x.yaml"))
    cfg.DATASETS.TRAIN = ("InOutDoorDepth_train",)
    print("InOutDoorDepth_train")
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_C4_3x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 5000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

    # cfg.MODEL.WEIGHTS = "output_RGB/model_final.pth"

    # model = build_model(cfg)
    cfg2 = cfg.clone()
    cfg.MODEL.WEIGHTS = "output/model_0029999.pth"
    cfg2.MODEL.WEIGHTS = "output/model_0019999.pth"

    predictions = {}
    if not os.path.isfile("gating_pred.pkl"):
        for d in val_dataset.files:
        #     input = image_process(os.path.join(val_dataset.root, 'Images', d + '.png'), os.path.join(val_dataset.root, 'Images', d + '.png'), cfg)
            input = image_process(os.path.join(val_dataset.root, 'ImagesQ_hd', d + '.png'),os.path.join(val_dataset.root, 'DepthJetQhd', d + '.png'), cfg2)
            with torch.no_grad():
                try:
                    gn = GatingNetwork(cfg, cfg2)
                    output = gn(input)
                    print('done')
                    pred = get_dicts(d, output[0])
                    del output
                    torch.cuda.empty_cache()
                    predictions[pred[0]] = [pred[1], pred[2]]
                except:
                    print(d)
                    continue
        with open("gating_pred.pkl", "wb")  as out:
            pickle.dump(predictions, out)
    else:
        with open("gating_pred.pkl", "rb")  as out:
            predictions = pickle.load(out)

    # print(outputs)
    groundtruth_boxes = read_gt()
    number_of_groundtruth_boxes = sum(
        [len(groundtruth_boxes[x]) for x in groundtruth_boxes.keys() if x in predictions.keys()])
    j, k = readAndSortBBs(predictions, groundtruth_boxes)
    evaluate(j, groundtruth_boxes, number_of_groundtruth_boxes, threshold=0.6)

    SOFTMAX_THRESHOLD = 0.6
        # torch.cuda.empty_cache()


