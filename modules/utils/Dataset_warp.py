import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from sklearn.model_selection import KFold
def main():
    dataset = TestDataset()
    SSM = SubSampleModule(dataset)
    #index = np.array([i for i in range(len(dataset))])
    #Cumulate_Dataset = None
    for i in range(10):
        unlabeled_index = [SSM[i] for i in range(len(SSM))]
        _index = np.random.choice(unlabeled_index, 10, replace=False)
        Cumulate_Dataset = SSM(_index)
        CVM = CrossValidationModule(5, Cumulate_Dataset)
        tr_ddd = []
        tv_ddd = []
        td = []
        for i in range(6):
            train_set, val_set = CVM()
            #print(len(train_set), len(val_set))
            tr_dataloader = DataLoader(train_set, batch_size=10, drop_last=False)
            val_dataloader = DataLoader(val_set, batch_size=10, drop_last=False)
            for _d in tr_dataloader:
                tr_ddd.append(_d)
                td.append(_d)
            for _d in val_dataloader:
                tv_ddd.append(_d)
                td.append(_d)
        print("Total data : {}, unique data : {}".format(len(torch.cat(td)), len(set(torch.cat(td).data.numpy()))))
        print("Training_freq : {}".format(np.histogram(torch.cat(tr_ddd).numpy(), [i for i in range(100)])))
        #import pdb;pdb.set_trace()

class SubSampleModule(object):
    def __init__(self, dataset):
        self.dataset = dataset
        self.cumulate_dataset = None
        self.indexes = np.array([i for i in range(len(dataset))])
        self.cumulate_indexes = np.array([])
    def __call__(self, new_index):
        new_index = np.array(new_index)
        self._update_index(new_index)
        concat_dataset = self._dataset_subset_selection(new_index)
        return concat_dataset
    def __getitem__(self, index):
        return self.indexes[index]
    def __len__(self):
        return len(self.indexes) 
    def _update_index(self, new_index):
        new_index = np.array(new_index)
        self.cumulate_indexes = np.hstack([self.cumulate_indexes, new_index])
        self.indexes = np.setdiff1d(self.indexes, new_index)
    def _dataset_subset_selection(self, new_index):        
        self.cumulate_dataset = Subset(self.dataset, new_index) + self.cumulate_dataset if self.cumulate_dataset is not None else Subset(self.dataset, new_index)
        return self.cumulate_dataset
    def query_sample_indexes(self):
        return self.cumulate_indexes, self.indexes

class CrossValidationModule(object):
    def __init__(self, k, dataset):
        self.k = k
        self.dataset = dataset
        self.index_n = np.random.permutation(len(self.dataset))
        stride = len(self.dataset)//self.k
        self.group = np.array([self.index_n[i * stride : (i+1) * stride] if i!=self.k-1 else self.index_n[i * stride : len(dataset)] for i in range(self.k)])
        self.count = 0
    def __call__(self):
        residual_group = np.setdiff1d(np.array([ i for i in range(self.k)]), np.array([self.count%self.k]))
        residual_index = np.hstack(self.group[residual_group])

        train_set = Subset(self.dataset, residual_index)
        validation_set = Subset(self.dataset, self.index_n[self.group[self.count%self.k]])

        self.count += 1

        return train_set, validation_set
    def direct_initial(self, index):
        self.index_n = index
        #np.random.permutation(len(self.dataset))
        stride = len(self.dataset)//self.k
        self.group = np.array([self.index_n[i * stride : (i+1) * stride] if i!=self.k-1 else self.index_n[i * stride : len(self.dataset)] for i in range(self.k)])
        

class TestDataset(Dataset):
    def __init__(self):
        self.index = [i for i in range(100)]
    def __getitem__(self, index):
        return self.index[index]
    def __len__(self):
        return len(self.index)

if __name__ == "__main__":
    main()
