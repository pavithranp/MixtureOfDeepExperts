# from detectron2 import model_zoo
# from detectron2.engine import DefaultPredictor
# from detectron2.config import get_cfg
# from detectron2.utils.visualizer import Visualizer
# from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.modeling import build_model
# import random
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling import build_backbone
from detectron2.structures import ImageList
import torch
# PATH = '../model/faster-RCNN.pth'
x = torch.randn(1,3,480,640).cuda()
input = [{'image':x}]
# model = torch.load(PATH)
cfg = get_cfg()
from detectron2.structures import ImageList
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_C4_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_C4_3x.yaml")

model = build_model(cfg)
model.eval()
image = ImageList (x,[(480,640)])
features = model.backbone(image.tensor)

# proposals, _ = model.proposal_generator(image, features)
# instances, _ = model.roi_heads(image, features, proposals)
# mask_features = [features[f] for f in model.roi_heads.in_features]
# proposals, _ = model.proposal_generator(x.cuda(), features)
# instances, _ = model.roi_heads(input, features, proposals)
# backbone = build_backbone(cfg)
# pooler = ROIPooler(...)
# rois = pooler(features, [0, 0, 0, 0])
# print(features)