from network.GatingNetwork import  GatingNetwork
from detectron2.config import get_cfg
from detectron2 import model_zoo
import os, torch,cv2, copy
import detectron2.data.transforms as T
from dataloader.torch_dataloader import RGBD
from detectron2.data import detection_utils as utils
from detectron2.utils.events import EventStorage


def mapper(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
    # can use other ways to read image
    rgb_image = utils.read_image(dataset_dict["rgb_file_name"], format="BGR")
    rgb_image = torch.tensor(rgb_image.transpose(2, 0, 1), requires_grad=True)
    depth_image = utils.read_image(dataset_dict["depth_file_name"], format="BGR")
    depth_image = torch.from_numpy(depth_image.transpose(2, 0, 1).copy(), requires_grad=True).cuda()
    annos = [
        utils.transform_instance_annotations(annotation, [], rgb_image.shape[1:])
        for annotation in dataset_dict.pop("annotations")
    ]
    return {
       # create the format that the model expects
       "rgb_image": rgb_image,
        "depth_image": depth_image,
        "path": dataset_dict["rgb_file_name"], # for debugging
        "width":960,
        "height" :540,
       "instances": utils.annotations_to_instances(annos, rgb_image.shape[1:])
    }
if __name__=="__main__":
    dataset_name ="RGBD"
    cfg = get_cfg()
    model = "COCO-Detection/faster_rcnn_R_50_C4_3x.yaml"
    cfg.merge_from_file(model_zoo.get_config_file(model))
    cfg.DATASETS.TRAIN = (dataset_name +"_train",)
    print("InOutDoorDepth_train")
    cfg.DATASETS.TEST = ()
    cfg.OUTPUT_DIR = dataset_name
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR

    cfg.SOLVER.MAX_ITER = 10000  # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 1  # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg_depth = cfg.clone()
    cfg.MODEL.WEIGHTS = "output/model_final.pth"
    cfg_depth.MODEL.WEIGHTS = "DepthJetQhd/model_final.pth"
    gn = GatingNetwork(cfg, cfg_depth)
    # train only linear
    for param in gn.parameters():
        param.requires_grad = True
    gn.set_training(True)
    params = [gn.weight]+  [gn.bias]
    optimizer = torch.optim.Adam(params)
    dataset = RGBD('/mnt/AAB281B7B2818911/datasets/InOutDoorPeopleRGBD')

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=20, shuffle=True, num_workers=4,collate_fn =dataset.collate_fn)

    it = iter(dataloader)
    # g = next(it)
    for epoch in range(1,10):
        for step in range(cfg.SOLVER.MAX_ITER):
            inputs = next(it)
            with EventStorage() as storage:
                losses = gn(inputs)
            optimizer.zero_grad()
            loss = losses['loss_cls']
            loss.backward()
            optimizer.step()
        print('Checkpoint')
        save_name = os.path.join(cfg.OUTPUT_DIR, 'Gatingnet_{}.pth'.format( epoch))
        gn_param = {"weights":gn.weight, "bias":gn.bias, "epoch":epoch}
        torch.save(gn_param, save_name)
    print('done')
    exit(0)
