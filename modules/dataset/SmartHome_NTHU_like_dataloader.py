import os
import cv2
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
import time

class SmartHome_Dataset(Dataset):
    def __init__(self, dataset_image_folder_path, dataset_label_json_path,\
                 transform=None, downpurning_ratio = 1.0):
        super(SmartHome_Dataset,self).__init__()
        self.class_names = ["BackGround", "Person"]

        self.downpurning_ratio = downpurning_ratio
        # Labels
        with open(dataset_label_json_path, 'r') as F:
            labels = json.load(F)
        self.labels = labels
        self.video_names = sorted(list(labels.keys()))

        # Images
        self.video_image_paths = {}
        self.video_lengths = {}
        self.video_index_mapping ={}
        self.accumulate_len = []
        total_count = 0
        mf, _folders, _files = next(iter(os.walk(dataset_image_folder_path)))
        for _folder in self.video_names:
            sub_video_img_folder_path = os.path.join(mf, _folder)
            mf2, _, _files2 = next(iter(os.walk(sub_video_img_folder_path)))
            _files2_tuple = [("".join(item), int("".join([char for char in item[0] if str.isdigit(char)])) ) for item in map(os.path.splitext, _files2)]
            _files2 = sorted(_files2_tuple, key = lambda x : x[1])

            downsample_stride = int( 1 / self.downpurning_ratio)
            _files2 = [_files2[i] for i in range(0, len(_files2), downsample_stride)]

            total_count += len(_files2)
            self.video_lengths[_folder] = len(_files2)
            self.accumulate_len.append(total_count)
            if _folder not in self.video_image_paths.keys():
                self.video_image_paths[_folder] = {}            
            if _folder not in self.video_index_mapping.keys():
                self.video_index_mapping[_folder] = []
            for _file, index in _files2:
                self.video_image_paths[_folder][str(index)] = os.path.join(mf2, _file)
                self.video_index_mapping[_folder].append(str(index))
            
        self.dataset_len = total_count
        self.transform = transform
        self.TENSOR_IN_MEMORY = True
        if self.TENSOR_IN_MEMORY:
            self.tensor_memory_buffer = []
            for i in range(self.dataset_len):
                video_name, frame_name = self.mapping_video(i)
                # 0.020
                img = self._load_image(video_name, frame_name)
                zero_img = np.zeros(tuple(img.shape[0:2]) +(1,))
                gt_img = self._load_gt(video_name, frame_name, zero_img)
                self.tensor_memory_buffer.append((img, gt_img))

        # classes_sample = np.concatenate([data[1][1] for data in map(self.get_annotation, range(len(self)) )], axis=0)
        # self.weighted_samples = np.histogram(classes_sample, range(0, max(classes_sample)+2))[0]
        # print(self.weighted_samples)
        # self.weighted_samples[0] = np.sum(self.weighted_samples) * 3
        # import pdb;pdb.set_trace()
        # self.weighted_samples = np.linalg.norm(self.weighted_samples) / (self.weighted_samples + 1e-8)
        # self.weighted_samples = self.weighted_samples / np.sum(self.weighted_samples)
        # self.weighted_samples = np.exp(self.weighted_samples*3) / np.sum(np.exp(self.weighted_samples*3))

        
    def __getitem__(self, index):
        if not self.TENSOR_IN_MEMORY:
            video_name, frame_name = self.mapping_video(index)
            img = self._load_image(video_name, frame_name)
            zero_img = np.zeros(tuple(img.shape[0:2]) +(1,))
            gt_img = self._load_gt(video_name, frame_name, zero_img)
            gt_img = gt_img.copy()
        else:
            img, gt_img = self.tensor_memory_buffer[index]
            img, gt_img = img.copy(), gt_img.copy()

        if self.transform:
            img, gt_img = self.transform(img, gt_img)

        return img, gt_img
    def __len__(self):
        return self.dataset_len
    def get_image(self, index):
        video_name, frame_name = self.mapping_video(index)
        return self._load_image(video_name, frame_name)
    def get_annotation(self, index):
        video_name, frame_name = self.mapping_video(index)
        gt = self._load_gt(video_name, frame_name)
        difficult = np.zeros_like(classes)
        return index, gt
    def _load_image(self, video_name, frame_name):
        img_path = self.video_image_paths[video_name][frame_name]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        return img

    def _load_gt(self, video_name, frame_name, zero_img):
        try:
            gt_lists = self.labels[video_name][frame_name]
        except:
            import pdb;pdb.set_trace()

        gtl = np.array(gt_lists)
        bboxes = gtl[:, :4]
        bboxes[:, 2] += bboxes[:, 0]
        bboxes[:, 3] += bboxes[:, 1]
        #labels = np.vectorize(self.class_mapping_table.get)(gtl[:,4])
        labels = gtl[:,4]
        non_dont_care_mask = labels!=-1
        bboxes, labels = bboxes[non_dont_care_mask,:].astype(np.float32), labels[non_dont_care_mask].astype(np.int64)
        for box in bboxes:
            box = box.astype(int)
            zero_img[box[1]:box[3], box[0]:box[2], 0] = 1
        gt_img = zero_img
        return gt_img
    def mapping_video(self, index):
        for pointer_location, max_bounded in enumerate(self.accumulate_len):
            if pointer_location ==0 :
                if index < max_bounded:
                    i = pointer_location
                    baseline = 0
                    break
            else:
                if ( index >= self.accumulate_len[pointer_location-1] ) and ( index < max_bounded ):
                    i = pointer_location
                    baseline = self.accumulate_len[pointer_location-1]
                    break
        video_name, frame_number = self.video_names[i], index - baseline
        frame_name = self.video_index_mapping[video_name][frame_number]
        return video_name, frame_name

if __name__ == "__main__":
    main()
