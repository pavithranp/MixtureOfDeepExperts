# from tqdm import tqdm
import cv2 ,os
import math,torch
import numpy as np
import operator
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from dataloader.single_loader import SingleDataset
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from utils.eval import readAndSortBBs, evaluate, read_gt, get_dicts
import pickle
import argparse

def parse_arg():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--modality_path', default= "DepthJetQhd",
                        help='path to RGB model')
    parser.add_argument('--data', default='/mnt/AAB281B7B2818911/datasets/InOutDoorPeopleRGBD',
                        help='data directory to save models')
    parser.add_argument('--batch_size',  default=2,
                        help='batch size for dataloader')
    parser.add_argument('--workers',  default=4,
                        help='no of workers for dataloader')
    parser.add_argument('--iterations', default=10000,
                        help='no. of iterations for training')
    parser.add_argument('--out_dir', default='output/',
                        help='output directory to save models')
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_arg()
    val_dataset = SingleDataset(root=args.data,dataset=args.modality_path,
                                    set='test',grad=True)
    sortedListTestBB =[]
    cfg = get_cfg()
    model = "COCO-Detection/faster_rcnn_R_50_C4_3x.yaml"
    cfg.merge_from_file(model_zoo.get_config_file(model))
    cfg.DATASETS.TRAIN = ("InOutDoorDepth_train",)
    print("InOutDoorDepth_train")
    cfg.DATASETS.TEST = ()
    cfg.OUTPUT_DIR = args.out_dir
    cfg.DATALOADER.NUM_WORKERS = args.workers
    cfg.MODEL.WEIGHTS = 'output/model_0024999.pth'
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = args.iterations  # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 50  # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 3000
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 3000
    cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 500
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 500
    model = build_model(cfg)
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)
    model.eval()
    outputs = []
    predictions = {}
    # dataset_dicts = x.load_annotations()
    print("validating with:",len(val_dataset.files))
    with torch.no_grad():
        for d in val_dataset.files:
            im = cv2.imread(os.path.join(val_dataset.root,val_dataset.dataset_path,d+'.png'))
            image = torch.as_tensor(im.transpose(2, 0, 1), dtype=torch.float32)
            output = model([{'image':image,'height': 540,'width':960 }])
            pred = get_dicts(d,output[0])
            predictions[pred[0]]= [pred[1],pred[2]]
    # print(outputs)
    groundtruth_boxes = read_gt()
    number_of_groundtruth_boxes = sum([len(groundtruth_boxes[x]) for x in groundtruth_boxes.keys() if x in predictions.keys()])
    j,k = readAndSortBBs(predictions,groundtruth_boxes,sortedListTestBB)
    evaluate(j, groundtruth_boxes, number_of_groundtruth_boxes, threshold=0.6)

    SOFTMAX_THRESHOLD = 0.6







