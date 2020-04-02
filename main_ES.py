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

from utils.Dataset_warp import SubSampleModule, CrossValidationModule
import cv2

def arguments():
    parser = argparse.ArgumentParser("Training a segmentation modules")
    parser.add_argument("--dataset", type=str, default="NTHUSCDet", help="dataset")
    parser.add_argument("-C", '--config_path', type=str, default="config/default_config.yaml", help='')
    parser.add_argument("-E", '--Epochs', type=int, default=130, help='Epochs')

    args = parser.parse_args()

    print(args)

    return args

def main(args):
    model = FireResidualDetection(3).cuda()
    optim, criteria = optim_and_criteria(model)

    train_dataset = DatasetLoad(args)
    #SSM = SubSampleModule(train_dataset)
    #unlabeled, labeled = SSM.query_sample_indexes()
    # Update indexes
    #labeled_dataset = SSM(new_index)
    labeled_dataset = train_dataset
    k = 6
    CV_managment = CrossValidationModule(k, labeled_dataset)

    #uncertainty_module = Uncertainty_header()
    #uncertainty = uncertainty_spatial(model, _input, uncertainty_module)

    val_loss_list = []
    _temp_val_loss = []
    for epoch in range(args.Epochs):
        train_ds, val_ds = CV_managment()
        train_dataloader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=4, drop_last=True, pin_memory=True)
        val_dataloader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=4, drop_last=False, pin_memory=True)
        running_loss = Train(model, train_dataloader, optim, criteria)
        #print("[Epochs][{}]\tloss:\t{}".format(epoch, running_loss))
        val_loss = Test(model, val_dataloader, criteria)
        print("[Epochs][{}]\tloss:\t{}".format(epoch, val_loss))
        _temp_val_loss.append(val_loss)

        if epoch%k == 0 and epoch!=0:
            val_loss_list.append(np.mean(_temp_val_loss))
            _temp_val_loss = []

        if np.sum(np.diff(val_loss_list[-5:]) > 0) >= 3: break
        #if epoch >10:
        #    Vis_result(model, train_dataloader)

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
        running_loss.append(loss.item())
    return np.mean(running_loss)

def Test(model, dataloader, criteria):
    running_loss = []
    with torch.no_grad():
        for index, (img, gt) in enumerate(dataloader):
            img, gt = img.float().cuda(), gt.long().cuda()
            segment_list = model(img)
            loss = criteria_seg(segment_list, gt)
            running_loss.append(loss.item())
    return np.mean(running_loss)

def Vis_result(model, dataloader):
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
