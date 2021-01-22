# from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
# from detectron2.config import get_cfg
# from detectron2.utils.visualizer import Visualizer
# from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.modeling import build_model
import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
# import random
from detectron2 import model_zoo
from detectron2.config import get_cfg
import torch ,cv2
from torch import nn
import numpy as np
# PATH = '../model/faster-RCNN_FPN.pth'
height, width = 480,640
im = cv2.imread("../docs/input.jpg")
# x = torch.tensor(np.reshape(im,(1,3,im.shape[0],im.shape[1]))).cuda()
x = torch.randn(1,3,height,width).cuda()
input = [{'image':x}]
model = "COCO-Detection/faster_rcnn_R_50_C4_3x.yaml"
# model = torch.load(PATH)
cfg = get_cfg()
from detectron2.structures import ImageList
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file(model))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)
aug = T.ResizeShortestEdge([cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST )
# pred = DefaultPredictor(cfg)
# out =pred(im)
# print(out)
model = build_model(cfg)
cfg1= cfg.clone()
image = cv2.imread("../docs/input.jpg")
height, width = image.shape[:2]
image = aug.get_transform(image).apply_image(image)
image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
inputs = [{"image": image, "height": height, "width": width}]
model.eval()
checkpointer = DetectionCheckpointer(model)
checkpointer.load(cfg1.MODEL.WEIGHTS)
# with torch.no_grad():
#     out = model(inputs)[0]
# print(out)

with torch.no_grad():
    model.proposal_generator.training = False
    images = model.preprocess_image(inputs)  # don't forget to preprocess
    features = model.backbone(images.tensor)  # set of cnn features
    proposals, _ = model.proposal_generator(images, features, None)  # RPN

    features_ = [features[f] for f in model.roi_heads.in_features]
    box_features = model.roi_heads.pooler(features_, [x.proposal_boxes for x in proposals])
    box_features = model.roi_heads.res5(box_features)  # features of all 1k candidates
    # predictions = model.roi_heads.box_predictor(box_features)
    x = box_features.mean(dim=[2, 3])  # features
    scores = model.roi_heads.box_predictor.cls_score(x)
    proposal_deltas = model.roi_heads.box_predictor.bbox_pred(x)
    predictions = scores, proposal_deltas
    pred_instances, _ = model.roi_heads.box_predictor.inference(predictions, proposals)
    results = model.roi_heads.forward_with_given_boxes(features, pred_instances)

    # output boxes, masks, scores, etc
    pred_instances = model._postprocess(pred_instances, inputs, images.image_sizes)  # scale box to orig size
    # features of the proposed boxes
    # feats = box_features[pred_inds]
    # model.eval()
    print(pred_instances)
# out = model(input)

# print(out)
class MyModelA(nn.Module):
    def __init__(self):
        super(MyModelA, self).__init__()
        self.fc1 = nn.Linear(20, 10)
        self.fc2 = nn.Linear(10, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class MyModelB(nn.Module):
    def __init__(self):
        super(MyModelB, self).__init__()
        self.fc1 = nn.Linear(10, 2)

    def forward(self, x):
        x = self.fc1(x)
        return x


class MyEnsemble(nn.Module):
    def __init__(self, modelA, modelB,model):
        super(MyEnsemble, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.faster_rcnn = model


    def forward(self, x1, x2):
        head1 = self.modelA(x1)
        head2 = self.modelB(self.modelA.fc1(x1))
        features = model.backbone(x2)
        proposals = model.proposal_generator(x2,features,None)
        proposal_boxes = [x.proposal_boxes for x in proposals]
        box_features = model.roi_heads._shared_roi_transform(
            [features[f] for f in self.in_features], proposal_boxes
        )
        x =  box_features.mean(dim=[2, 3]) #features
        scores = model.roi_heads.box_predictor.cls_score(x)
        proposal_deltas = model.roi_heads.box_predictor.bbox_pred(x)
        predictions = scores,proposal_deltas
        pred_instances, _ = model.roi_heads.box_predictor.inference(predictions, proposals)
        results = model.roi_heads.forward_with_given_boxes(features, pred_instances)
        x = torch.cat((head1, head2,results), dim=1)
        return x

a = MyModelA()
b = MyModelB()
combined = MyEnsemble(a,b,model)
# print(combined)