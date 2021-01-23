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

# with torch.no_grad():
#     out = model(inputs)[0]
# print(out)
class FRCNN_ROIHeads(nn.Module):
    def __init__(self,cfg):
        super(FRCNN_ROIHeads, self).__init__()
        self.cfg = cfg
        self.model = build_model(cfg)

    def forward(self,inputs):
        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(self.cfg.MODEL.WEIGHTS)

        self.model.eval()
        with torch.no_grad():
            # return self.model(inputs)[0]
            self.model.proposal_generator.training = False
            images = self.model.preprocess_image(inputs)  # don't forget to preprocess
            features = self.model.backbone(images.tensor)  # set of cnn features
            proposals, _ = self.model.proposal_generator(images, features, None)  # RPN

            features_ = [features[f] for f in self.model.roi_heads.in_features]
            # box_features = self.model.roi_heads.box_pooler(features_, [x.proposal_boxes for x in proposals])
            # x = self.model.roi_heads.box_head(box_features)  # features of all 1k candidates
            box_features = self.model.roi_heads.pooler(features_, [x.proposal_boxes for x in proposals])
            box_features1 = self.model.roi_heads.res5(box_features)  # features of all 1k candidates

            # predictions = model.roi_heads.box_predictor(box_features)
            x = box_features1.mean(dim=[2, 3])  # features
            if x.dim() > 2:
                x = torch.flatten(x, start_dim=1)
            scores = self.model.roi_heads.box_predictor.cls_score(x)
            proposal_deltas = self.model.roi_heads.box_predictor.bbox_pred(x)
            return box_features,scores, proposal_deltas, proposals, features

class FRCNN_OutputLayer(nn.Module):
    def __init__(self,cfg):
        super(FRCNN_OutputLayer, self).__init__()
        self.model = build_model(cfg)
        self.model.eval()
    def forward(self,scores, proposal_deltas, proposals, features,images):
        with torch.no_grad():
            predictions = scores, proposal_deltas
            pred_instances, _ = self.model.roi_heads.box_predictor.inference(predictions, proposals)
            pred_instances = self.model.roi_heads.forward_with_given_boxes(features, pred_instances)
            return self.model._postprocess(pred_instances, inputs, images.image_sizes)



class GatingNetwork(nn.Module):
    def __init__(self, cfg_modelA, cfg_modelB):
        super(GatingNetwork, self).__init__()
        num_class = 2
        num_experts = 2
        self.RGBDetector = FRCNN_ROIHeads(cfg_modelA)
        self.DepthDetector = FRCNN_ROIHeads(cfg_modelB)
        self.gatingLayer1 = nn.Linear(2048,500)
        self.RGBGating = nn.Linear(500,num_class)
        self.DepthGating = nn.Linear(500,num_class)
        self.output = FRCNN_OutputLayer(cfg_modelA)


    def forward(self, x1, x2):
            RGB_box_features, RGB_scores, RGB_proposal_deltas, RGB_proposals, RGBfeatures = self.RGBDetector(x1)
            Depth_box_features, Depth_scores, _, _, _ = self.DepthDetector(x2)
            inputs = torch.cat([RGB_box_features,Depth_box_features ], dim=1)
            # Depth_box_features, Depth_scores, Depth_proposal_deltas, Depth_proposals, Depthfeatures = self.DepthDetector(x2)
            x = self.gatingLayer1(inputs)
            RGB = self.RGBGating(x)
            Depth = self.DepthGating(x)
            self.weighted_scores = nn.Softmax(RGB*RGB_scores + Depth*Depth_scores)
            return self.output(self.weighted_scores,RGB_proposal_deltas, RGB_proposals, RGBfeatures)

if __name__ == '__main':
    height, width = 480, 640
    im = cv2.imread("../docs/input.jpg")
    # x = torch.tensor(np.reshape(im,(1,3,im.shape[0],im.shape[1]))).cuda()
    x = torch.randn(1, 3, height, width).cuda()
    input = [{'image': x}]
    model = "COCO-Detection/faster_rcnn_R_50_C4_3x.yaml"
    # model = torch.load(PATH)
    cfg = get_cfg()
    from detectron2.structures import ImageList

    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file(model))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)
    aug = T.ResizeShortestEdge([cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST)
    pred = DefaultPredictor(cfg)
    # out =pred(im)
    # print(out)
    model = build_model(cfg)
    image = cv2.imread("../docs/image.jpg")
    height, width = image.shape[:2]
    image = aug.get_transform(image).apply_image(image)
    image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
    inputs = [{"image": image, "height": height, "width": width}]
    model.eval()

    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)
    a = build_model(cfg)
    b = build_model(cfg)
    combined = GatingNetwork(a,b,model)
# print(combined)