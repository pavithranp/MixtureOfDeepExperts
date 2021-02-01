import random

# import some common detectron2 utilities
# from detectron2.utils.visualizer import Visualizer
from dataloader.single_loader import SingleDataset
# from tqdm import tqdm
import cv2 ,os
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from dataloader.dataset import Dataset

if __name__ == "__main__":

    val_dataset = SingleDataset(root='/mnt/AAB281B7B2818911/datasets/InOutDoorPeopleRGBD',dataset='ImagesQ_hd',
                                    set='val',grad=True)

    args = "DepthJetQhd/"
    x = Dataset(args)
    cfg = get_cfg()
    model = "COCO-Detection/faster_rcnn_R_50_C4_3x.yaml"
    cfg.merge_from_file(model_zoo.get_config_file(model))
    cfg.DATASETS.TRAIN = ("InOutDoorDepth_train",)
    print("InOutDoorDepth_train")
    cfg.DATASETS.TEST = ()
    cfg.OUTPUT_DIR = 'output/'
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = 'output/model_final.pth'
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 10000  # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 50  # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 3000
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 3000
    cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 500
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 500
    predictor = DefaultPredictor(cfg)

    outputs = []
    dataset_dicts = x.load_annotations()
    for d in val_dataset.files:
        im = cv2.imread(os.path.join(val_dataset.root,val_dataset.dataset_path,d+'.png'))
        output = predictor(im)
        outputs.append(output)


# def evaluate(test_loader, model):
#     """
#     Evaluate.
#
#     :param test_loader: DataLoader for test data
#     :param model: model
#     """
#
#     # Make sure it's in eval mode
#     model.eval()
#
#     # Lists to store detected and true boxes, labels, scores
#     det_boxes = list()
#     det_labels = list()
#     det_scores = list()
#     true_boxes = list()
#     true_labels = list()
#     true_difficulties = list()  # it is necessary to know which objects are 'difficult', see 'calculate_mAP' in utils.py
#
#     with torch.no_grad():
#         # Batches
#         for i, (images, boxes, labels, difficulties) in enumerate(tqdm(test_loader, desc='Evaluating')):
#             images = images.to(device)  # (N, 3, 300, 300)
#
#             # Forward prop.
#             predicted_locs, predicted_scores = model(images)
#
#             # Detect objects in SSD output
#             det_boxes_batch, det_labels_batch, det_scores_batch = model.detect_objects(predicted_locs, predicted_scores,
#                                                                                        min_score=0.01, max_overlap=0.45,
#                                                                                        top_k=200)
#             # Evaluation MUST be at min_score=0.01, max_overlap=0.45, top_k=200 for fair comparision with the paper's results and other repos
#
#             # Store this batch's results for mAP calculation
#             boxes = [b.to(device) for b in boxes]
#             labels = [l.to(device) for l in labels]
#             difficulties = [d.to(device) for d in difficulties]
#
#             det_boxes.extend(det_boxes_batch)
#             det_labels.extend(det_labels_batch)
#             det_scores.extend(det_scores_batch)
#             true_boxes.extend(boxes)
#             true_labels.extend(labels)
#             true_difficulties.extend(difficulties)
#
#         # Calculate mAP
#         APs, mAP = calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties)
#
#     # Print AP for each class
#     pp.pprint(APs)
#
#     print('\nMean Average Precision (mAP): %.3f' % mAP)


# if __name__ == '__main__':
#     evaluate(test_loader, model)
