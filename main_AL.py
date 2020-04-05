import os
import yaml
import torch
import numpy as np
import argparse
from torch.utils.data import DataLoader

from procedure_func import *

from modules.model.fireresidual_detection import FireResidualDetection
from modules.model.detection_header import Uncertainty_header, uncertainty_spatial
from modules.utils.Dataset_warp import SubSampleModule, CrossValidationModule
import scipy.ndimage as sp_im
import cv2
import time

use_temporal_coherence=True
apply_selection_trick=True
trick_flags = None
START_TRAINING_TIME = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())

def arguments():
    parser = argparse.ArgumentParser("Training a segmentation modules")
    parser.add_argument("--dataset", type=str, default="NTHUSCDet", help="dataset")
    parser.add_argument("-C", '--config_path', type=str, default="config/default_config.yaml", help='')
    parser.add_argument("-E", '--Epochs', type=int, default=320, help='Epochs')

    args = parser.parse_args()

    with open(args.config_path, 'r') as F:
        config = yaml.load(F)
    assert args.dataset in config['dataset_path'].keys(), "We only provide {} ".format(config['dataset_path'].keys())
    args.config = config
    print(args)

    return args

def main(args):
    global trick_flags
    # Config parameters
    BatchSize = args.config['training_parameters']['batch_size']
    NUM_workers = args.config['training_parameters']['num_workers']

    # Training Inintial
    model = FireResidualDetection(3).cuda()
    optim, criteria = optim_and_criteria(args, model)
    train_dataset = DatasetLoad(args)
    active_loader = DataLoader(train_dataset, shuffle=False, batch_size=50, pin_memory=True, num_workers=NUM_workers)

    postfix = [args.dataset, START_TRAINING_TIME]
    postfix = "_".join(postfix)

    # Active learning phase 
    SSM = SubSampleModule(train_dataset)
    labeled, unlabeled = SSM.query_sample_indexes()
    labeled, unlabeled = labeled.tolist(), unlabeled.tolist()
    selection_times = 10
    selection_num = len(unlabeled) // selection_times
    query_item = Query_item_list_gen(unlabeled, selection_times, mode='uniform')
    for i in range(0, selection_times):
        labeled, unlabeled = SSM.query_sample_indexes()
        labeled, unlabeled = labeled.tolist(), unlabeled.tolist()
        if i != 0:
            folder_name = os.path.join("../experiments/", postfix, 'checkpoints', 'query_iter_{:.2f}'.format(float(i)/float(selection_times)))
            mf, subfs, files = next(iter(os.walk(folder_name)))
            files = sorted(files, key=lambda k : float(str.split(k, '-')[-1][:-4]))
            for _file in files[1:]:
                os.remove(os.path.join(folder_name, _file))
            model.load_state_dict(torch.load(os.path.join(folder_name,files[0])))

        filepath = os.path.join("../experiments/", postfix, "selection", "{}_{}".format('spatial_mi', START_TRAINING_TIME), "query_iter_{:.1f}.txt".format(i*0.1+0.1))
        # query_index = np.random.choice(unlabeled, size=selection_num, replace=False)
         
        uncertainty_module = Uncertainty_header()
        uncertainties = []
        for images, gt in active_loader:
            with torch.no_grad():
                images = images.float().cuda()
                feature_map_uncertainty = uncertainty_spatial(model, images, uncertainty_module)
                uncertainty_values = np.mean(sp_im.maximum_filter(feature_map_uncertainty.data.cpu().numpy(), size = (1, 30, 30))[..., ::30, ::30], axis=(1, 2) )
                uncertainties.append(uncertainty_values)
        uncertainties = np.hstack(uncertainties)

        if use_temporal_coherence:
            filt = np.exp(-np.power(np.linspace(-5, 5, 11, dtype=np.float32), 2) / (2*3**2))
            filt = filt / filt.sum()
            uncertain_smooth = np.convolve(uncertainties, filt, 'same')
        uncertain = uncertain_smooth
        inds_sorted = np.argsort(uncertain)[::-1]

        data_table = torch.zeros((1,len(train_dataset))).scatter_(1, torch.LongTensor(labeled)[None,:], 1).squeeze().long().data.cpu().numpy()
        if trick_flags is not None:
            flags = trick_flags
        else:
            flags = np.zeros((uncertain.shape[0],))
            
        budget = query_item[i]
        count = 0
        list_selected_inds = []
        if np.sum(flags==False) > budget:
            for _tt in range(3):
                flags_selected = np.zeros((uncertain.shape[0],))
                _ii = 0
                while count < budget and _ii < uncertain.shape[0]:
                    if flags[inds_sorted[_ii]] == 1 or (flags_selected[inds_sorted[_ii]] == 1 and _tt <= 1):
                        _ii += 1
                        continue
                    list_selected_inds.append(_ii)

                    flags[inds_sorted[_ii]] = True
                    if apply_selection_trick:
                        flags_selected[inds_sorted[_ii] - 15:inds_sorted[_ii] + 15] = 1
                        flags[inds_sorted[_ii] - 2:inds_sorted[_ii] + 3] = 1
                    else:
                        flags_selected[inds_sorted[_ii]] = 1
                        flags[inds_sorted[_ii]] = 1

                    assert len(flags) == uncertain.shape[0]
                    #data_table.append(self._ds_active_unlabeled.data_table[inds_sorted[_ii]])
                    #ped_count += len(data_table[-1][1])
                    count += 1
            trick_flags = flags
            query_index = list_selected_inds
        else:
            unlabeled_np = np.array(unlabeled)
            unlabeled_uncert = uncertain[unlabeled_np]
            selection_index = np.argsort(unlabeled_uncert)[::-1][:budget]
            query_index = unlabeled_np[selection_index].to_list()
        
        # Update indexes
        recode_selection_information(filepath, train_dataset, query_index)
        labeled_dataset = SSM(query_index) #labeled_dataset = train_dataset

        # Cross Validationset module initial
        k = args.config['early_stop']['cross_validation_params']['k']
        CV_managment = CrossValidationModule(k, labeled_dataset)

        # Optimize the model
        val_loss_list = []
        _temp_val_loss = []
        for epoch in range(args.Epochs):
            train_ds, val_ds = CV_managment()
            train_dataloader = DataLoader(train_ds, batch_size=BatchSize, shuffle=True, num_workers=NUM_workers, drop_last=True, pin_memory=True)
            val_dataloader = DataLoader(val_ds, batch_size=BatchSize, shuffle=False, num_workers=NUM_workers, drop_last=True, pin_memory=True)

            running_loss = Train(model, train_dataloader, optim, criteria)
            val_loss = Test(model, val_dataloader, criteria)
            print("[Epochs][{}]\tloss:\t{}".format(epoch, val_loss))
            _temp_val_loss.append(val_loss)

            if epoch%k == 0 and epoch!=0:
                mean_val_loss = np.mean(_temp_val_loss)
                val_loss_list.append(mean_val_loss)
                folder_name = os.path.join("../experiments/", postfix, 'checkpoints', 'query_iter_{:.2f}'.format(float(i+1)/float(selection_times)))
                if not os.path.exists(folder_name):
                    os.makedirs(folder_name)
                model_path = os.path.join(folder_name, "Epoch-{}-Loss-{}.pth".format(epoch, mean_val_loss))
                torch.save(model.state_dict(), model_path)
                _temp_val_loss = []

            observe_window = args.config['early_stop']['observe_window']
            if np.sum(np.diff(val_loss_list[-observe_window:]) > 0) >= (observe_window//2+1): break
            #if epoch >10:
            #    Vis_result(model, train_dataloader)
    folder_name = os.path.join("../experiments/", postfix, 'checkpoints', 'query_iter_{:.2f}'.format(float(i+1)/float(selection_times)))
    mf, subfs, files = next(iter(os.walk(folder_name)))
    files = sorted(files, key=lambda k : float(str.split(k, '-')[-1][:-4]))
    for _file in files[1:]:
        os.remove(os.path.join(folder_name, _file))

def recode_selection_information(filepath, train_dataset, query_index):
    if not os.path.exists(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath))
    with open(filepath, 'w') as F:
        for _s_index in query_index:
            video_name, frame_id = train_dataset.mapping_video(_s_index)
            F.write("{}, {}\n".format(video_name, frame_id))

def Query_item_list_gen(unlabeled, Query_iteration, mode='uniform'):
    def search_the_ratio(sample_num, base, query_iteration):
        _grid_search_ratio = 0.03
        _seudo_r = 0
        while True:
            expect_num = base * (_seudo_r ** query_iteration - 1) / (_seudo_r -1)
            if sample_num < expect_num:    break
            _seudo_r += _grid_search_ratio
        return _seudo_r - _grid_search_ratio
    if mode == "uniform":
        query_item = len(unlabeled) // Query_iteration
        query_item = [query_item for i in range(Query_iteration)]
    elif mode == 'incremental':
        if len(unlabeled) < (2 ** Query_iteration -1):
            base = 1
            r = search_the_ratio(len(unlabeled), base, Query_iteration)
        else:
            r = 2
            base = int(len(unlabeled) / ( r ** (Query_iteration) - 1))
        query_item = [ int(base * (r ** _iter)) for _iter in range(Query_iteration)]
        bias = (len(unlabeled) - sum(query_item)) // Query_iteration
        _residual_bias = (len(unlabeled) - sum(query_item)) - bias * Query_iteration
        query_item = [item + bias for item in query_item]
        query_item[0] += _residual_bias
    else:
        raise NotImplementedError("We only provide two evaluation protocol: uniform and incremental")

    return query_item 

if __name__ == "__main__":
    args = arguments()
    main(args)
