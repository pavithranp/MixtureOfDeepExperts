from network.GatingNetwork import  GatingNetwork
from detectron2.config import get_cfg
from detectron2 import model_zoo
import os, torch,cv2, copy, errno
import detectron2.data.transforms as T
from dataloader.torch_dataloader import RGBD
from detectron2.data import detection_utils as utils
from detectron2.utils.events import EventStorage
import argparse


def parse_arg():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--model1', default= "output/model_final.pth",
                        help='path to RGB model')
    parser.add_argument('--model2', default="output/model_0019999.pth",
                        help='path to Depth model')
    parser.add_argument('--batch_size',  default=20,
                        help='batch size for dataloader')
    parser.add_argument('--no_of_workers',  default=4,
                        help='no of workers for dataloader')
    parser.add_argument('--data',  default='/mnt/AAB281B7B2818911/datasets/InOutDoorPeopleRGBD',
                        help='path to InOutDoorData')
    parser.add_argument('--epoch', default=10,
                        help='no. of epochs for training')
    parser.add_argument('--out_dir', default='RGBD',
                        help='output directory to save models')
    return parser.parse_args()

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


    args = parse_arg()
    cfg = get_cfg()
    model = "COCO-Detection/faster_rcnn_R_50_C4_3x.yaml"
    cfg.merge_from_file(model_zoo.get_config_file(model))
    cfg.DATASETS.TRAIN = (args.out_dir +"_train",)
    print("InOutDoorDepth_train")
    cfg.DATASETS.TEST = ()
    cfg.OUTPUT_DIR = args.out_dir
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR

    cfg.SOLVER.MAX_ITER = 10000  # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 1  # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg_depth = cfg.clone()
    cfg.MODEL.WEIGHTS = args.model1
    cfg_depth.MODEL.WEIGHTS = "DepthJetQhd/model_final.pth"
    gn = GatingNetwork(cfg, cfg_depth)
    # train only linear
    for param in gn.parameters():
        param.requires_grad = True
    gn.set_training(True)
    params = [gn.weight]+  [gn.bias]
    optimizer = torch.optim.Adam(params)
    if os.path.exists(args.data):
        dataset = RGBD(args.data)
    else:
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), args.data)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,collate_fn =dataset.collate_fn)

    it = iter(dataloader)
    for epoch in range(1,args.epoch):
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
