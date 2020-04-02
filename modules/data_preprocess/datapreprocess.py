import cv2
import numpy as np

class Compose(object):
    def __init__(self, module_list):
        assert isinstance(module_list, list), "the transfrom doesn't a modules"
        self.ml = module_list
    def __call__(self, img, gt):
        for _func in self.ml:
            img, gt = _func(img, gt)
        return img, gt

class Resize(object):
    def __init__(self, size):
        self.size = size
    def __call__(self, img, gt):
        img, gt = cv2.resize(img, self.size), cv2.resize(gt, self.size)
        return img, gt


class ToTensor(object):
    def __init__(self):
        pass
    def __call__(self, img, gt):
        # To transform to 0-1
        if img.dtype == np.uint8:
            img = img.astype(float)
            img /= 255
        if gt.dtype == np.uint8:
            gt = gt.astype(float)
            gt /= 255

        # transpose to (2, 1, 0)
        
        if len(gt.shape) == 2:
            gt = gt[None,...]
            # img, gt = img.transpose(2, 0, 1), gt.transpose(2, 0, 1)
        img = img.transpose(2, 0, 1)
        
        return img, gt