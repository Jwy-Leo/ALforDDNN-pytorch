from modules.data_preprocess.datapreprocess import Compose, ToTensor, Resize
from modules.dataset.NTHUSCDet_dataset import NTHUSCDet_Dataset
from modules.dataset.SmartHome_NTHU_like_dataloader import SmartHome_Dataset

def DatasetLoad(args):
    Transform = Compose([
        Resize((640, 480)),
        ToTensor()
    ])
    config = args.config
    if "NTHUSCDet" in args.dataset:
        dataset = NTHUSCDet_Dataset(config['dataset_path'][args.dataset]['train_img_path'], \
                                    config['dataset_path'][args.dataset]['train_gt_path'], \
                                    transform=Transform)
    elif args.dataset == "SmartHome":
        dataset = SmartHome_Dataset(config['dataset_path']['SmartHome']['train_img_path'], \
                                    config['dataset_path']['SmartHome']['train_gt_path'], \
                                    transform=Transform)
    else:
        raise NotImplementedError

    return dataset
