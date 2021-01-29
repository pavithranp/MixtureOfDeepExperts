# from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
# from detectron2.config import get_cfg
# from detectron2.utils.visualizer import Visualizer
# from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.modeling import build_model
import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
# import random
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.structures import ImageList
import torch ,cv2
from torch import nn
import numpy as np
from network import custom_proposal_generator
# PATH = '../model/faster-RCNN_FPN.pth'

# with torch.no_grad():
#     out = model(inputs)[0]
# print(out)
class Gating_ROIHeads(nn.Module):
    def __init__(self,cfg1,cfg2):

        super(Gating_ROIHeads, self).__init__()
        self.model3 = build_model(cfg2)
        cfg1.MODEL.PROPOSAL_GENERATOR.NAME = 'CustomRPN'
        cfg2.MODEL.PROPOSAL_GENERATOR.NAME = 'CustomRPN'
        cfg1.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 3000
        cfg2.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 3000
        cfg1.MODEL.RPN.PRE_NMS_TOPK_TEST = 3000
        cfg2.MODEL.RPN.PRE_NMS_TOPK_TEST = 3000
        cfg1.MODEL.RPN.POST_NMS_TOPK_TRAIN = 500
        cfg2.MODEL.RPN.POST_NMS_TOPK_TRAIN = 500
        cfg1.MODEL.RPN.POST_NMS_TOPK_TEST = 500
        cfg2.MODEL.RPN.POST_NMS_TOPK_TEST = 500
        self.cfg1 = cfg1
        self.model1 = build_model(cfg1)
        self.cfg2 = cfg2
        self.model2 = build_model(cfg2)
        self.training = False

    def forward(self,x1):
        # out =self.model3(x1[0])
        checkpointer1 = DetectionCheckpointer(self.model1)
        checkpointer1.load(self.cfg1.MODEL.WEIGHTS)
        checkpointer2 = DetectionCheckpointer(self.model2)
        checkpointer2.load(self.cfg2.MODEL.WEIGHTS)

        self.model1.eval()
        self.model2.eval()
        # self.model3.eval()
        self.model3.training =True
        # for d in x1:
        #     d['image'] = d.pop('rgb_image')
        # out = self.model3(x1)[0]

        with torch.no_grad():
            # return self.model(inputs)[0]
            self.model1.proposal_generator.training = False
            targets = self.get_targets(x1)
            images1 = self.preprocess_image(x1,'rgb_image')  # don't forget to preprocess
            features1 = self.model1.backbone(images1.tensor)  # set of cnn features
            proposals1, _ = self.model1.proposal_generator(images1, features1, None)  # RPN


            self.model2.proposal_generator.training = False
            images2 = self.preprocess_image(x1,'depth_image')  # don't forget to preprocess
            features2 = self.model2.backbone(images2.tensor)  # set of cnn features
            proposals2, _ = self.model2.proposal_generator(images2, features2, None)  # RPN
            topklogits = torch.cat((proposals1[0],proposals2[0]),dim=1)
            topkboxes = torch.cat((proposals1[1],proposals2[1]),dim=1)
            levels = torch.cat((proposals1[2],proposals2[2]),dim=0)
            proposals = self.model1.proposal_generator.merge_proposals(image_sizes= images2.image_sizes,
                                            nms_thresh= self.model1.proposal_generator.nms_thresh,
                                            post_nms_topk= self.model1.proposal_generator.pre_nms_topk[self.training],
                                            min_box_size= self.model1.proposal_generator.min_box_size,
                                            training= self.model1.proposal_generator.training,
                                            topk_proposals=topkboxes,topk_scores=topklogits,level_ids=levels)
            if self.training:
                targets = self.get_targets(x1)
                assert targets
                proposals = self.model1.roi_heads.label_and_sample_proposals(proposals, targets)
            del targets
            features1_ = [features1[f] for f in self.model1.roi_heads.in_features]
            features2_ = [features2[f] for f in self.model2.roi_heads.in_features]

            box_features1 = self.model1.roi_heads._shared_roi_transform(features1_, [x.proposal_boxes for x in proposals])

            x = box_features1.mean(dim=[2, 3])  # features
            if x.dim() > 2:
                x = torch.flatten(x, start_dim=1)
            scores1 = self.model1.roi_heads.box_predictor.cls_score(x)
            proposal_deltas1 = self.model1.roi_heads.box_predictor.bbox_pred(x)

            box_features2 = self.model2.roi_heads._shared_roi_transform(features2_, [x.proposal_boxes for x in proposals])

            x = box_features2.mean(dim=[2, 3])  # features
            if x.dim() > 2:
                x = torch.flatten(x, start_dim=1)
            scores2 = self.model2.roi_heads.box_predictor.cls_score(x)
            proposal_deltas2 = self.model2.roi_heads.box_predictor.bbox_pred(x)

            return box_features1,box_features2,scores1,scores2, proposal_deltas1, proposals ,features1_,images1

    def preprocess_image(self, batched_inputs,string):
        """
        Normalize, pad and batch the input images.
        """
        images = [x[string].to(self.model1.device) for x in batched_inputs]
        images = [(x - self.model1.pixel_mean) / self.model1.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.model1.backbone.size_divisibility)
        return images

    def get_targets(self,batches_inputs):
        return [x['instances'].to(self.model1.device) for x in batches_inputs]

class Gating_OutputLayer(nn.Module):
    def __init__(self,cfg):
        super(Gating_OutputLayer, self).__init__()
        self.model = build_model(cfg)
        # self.model.eval()
        self.training = False
    def forward(self,scores, proposal_deltas, proposals, features,images,batch_inputs):

        predictions = scores, proposal_deltas
        if self.training:
            del features
            losses = self.model.roi_heads.box_predictor.losses(predictions, proposals)
            return losses
        pred_instances, _ = self.model.roi_heads.box_predictor.inference(predictions, proposals)
        pred_instances = self.model.roi_heads.forward_with_given_boxes(features, pred_instances)
        return self.model._postprocess(pred_instances, batch_inputs, images.image_sizes)



class GatingNetwork(nn.Module):
    def __init__(self, cfg_modelA, cfg_modelB):
        super(GatingNetwork, self).__init__()
        num_class = 2
        num_experts = 2
        self.Detector = Gating_ROIHeads(cfg_modelA,cfg_modelB)
        self.weight = Parameter(torch.Tensor(2, 4096).cuda())
        self.bias = Parameter(torch.Tensor(2).cuda())

        self.output = Gating_OutputLayer(cfg_modelA)

    def set_training(self,cond):
        self.Detector.training = cond
        self.output.training = cond

    def forward(self, x1):
        RGB_box_features, Depth_box_features, RGBscores, DepthScores, RGB_proposal_deltas, RGB_proposals ,RGBfeatures,image = self.Detector(x1)
        inputs = torch.cat([RGB_box_features,Depth_box_features ], dim=1)
        input = inputs.mean(dim=[2, 3])
        x = F.linear(input,self.weight,self.bias)
        scores = x*RGBscores + (1-x)*DepthScores

        return self.output(scores,RGB_proposal_deltas, RGB_proposals, RGBfeatures,image,x1)
