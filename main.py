import torch
import yaml
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
import numpy as np

from data_preprocess.datapreprocess import Compose, ToTensor, Resize
from dataset.NTHUSCDet_dataset import NTHUSCDet_Dataset

from model.fireresidual_detection import FireResidualDetection
from model.detection_header import Uncertainty_header

import cv2

def arguments():
    parser = argparse.ArgumentParser("Training a segmentation modules")
    parser.add_argument("--dataset", type=str, default="NTHUSCDet", help="dataset")
    parser.add_argument("-C", '--config_path', type=str, default="config/default_config.yaml", help='')
    parser.add_argument("-E", '--Epochs', type=int, default=30, help='Epochs')

    args = parser.parse_args()

    print(args)

    return args

def main(args):
    model = FireResidualDetection(3).cuda()
    optim, criteria = optim_and_criteria(model)

    train_dataset = DatasetLoad(args)
    train_dataset[0]
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, drop_last=True)
    #uncertainty_module = Uncertainty_header()
    #uncertainty = uncertainty_spatial(model, _input, uncertainty_module)
    for epoch in range(args.Epochs):
        running_loss = Train(model, train_dataloader, optim, criteria)
        print("[Epochs][{}]\tloss:\t{}".format(epoch, running_loss))
        if epoch > 10:
            Test(model, train_dataloader)

def optim_and_criteria(model):
    optim = torch.optim.Adam(model.parameters(), lr=3e-3)#, weight_decay=1e-6)
    criteria = criteria_seg#torch.nn.CrossEntropy()
    return optim, criteria 

def Train(model, dataloader, optim, criteria):
    running_loss = []
    for index, (img, gt) in enumerate(dataloader):
        
        img, gt = img.float().cuda(), gt.long().cuda()
        
        segment_list = model(img)
        loss = criteria_seg(segment_list, gt)
        optim.zero_grad()
        loss.backward()
        optim.step()
        # print(loss.item())
        running_loss.append(loss.item())
    return np.mean(running_loss)
def Test(model, dataloader):
    with torch.no_grad():
        for img, gt in dataloader:
            b, c, w, h = gt.shape
            cv2.imshow('img', img.data.cpu().numpy()[0,...].transpose(1, 2, 0))
            cv2.imshow('gt', gt.data.cpu().numpy()[0,...].transpose(1, 2, 0))
            pred_list = model(img)
            pred = torch.mean(torch.cat([torch.nn.functional.interpolate(_pred, size=(w,h)) for _pred in pred_list], dim=1), dim=1)
            
            cv2.imshow('pred', pred.data.cpu().numpy()[0, ...])
            cv2.waitKey()
        
        
    return 
def Active_modules(model):
    import pdb;pdb.set_trace()

def criteria_seg(_input, _target):
    
    for index, _input_l in enumerate(_input):
        # if index >1 : break
        b, c, w, h = _input_l.shape
        
        _target_resize = torch.nn.functional.interpolate(_target.float(), size=(w, h)).long().squeeze()
        if index != 0:
            loss += nn.CrossEntropyLoss()(_input_l, _target_resize)
        else:
            loss = nn.CrossEntropyLoss()(_input_l, _target_resize)

        # _target_resize = torch.nn.functional.interpolate(_target.float(), size=(w, h)).float()
        # loss += nn.MSELoss()(_input_l, _target_resize)
        
    return loss

    # n_class = 2
    # weights = torch.FloatTensor(n_class).fill_(1.0)
    # import pdb;pdb.set_trace()
    # log_soft_max = nn.LogSoftmax()
    # nll_loss_2d = nn.NLLLoss2d(weights)
    # return nll_loss_2d(log_soft_max(_input), _target)

def DatasetLoad(args):
    with open(args.config_path, 'r') as F:
        config = yaml.load(F)
    Transform = Compose([
        Resize((640, 480)),
        ToTensor()
    ]) 
    if args.dataset == "NTHUSCDet":
        dataset = NTHUSCDet_Dataset(config['dataset_path']['NTHUSCDet']['train_img_path'], config['dataset_path']['NTHUSCDet']['train_gt_path'], \
                                    transform=Transform)
    elif args.dataset == "SmartHome":
        # dataset = #
        pass
    else:
        raise NotImplementedError

    return dataset

if __name__ == "__main__":
    args = arguments()
    main(args)
