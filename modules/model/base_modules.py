import torch
import torch.nn as nn
import torch.nn.functional as F
# activate_fn = nn.ELU()
# normalize_fn = nn.BatchNorm2d()

class Downsample(nn.Module):
    def __init__(self, inplane, outplane, k_size_conv=3, k_size_pool=2, stride=2, keep_prob = 1):
        super(Downsample, self).__init__()
        self.conv = nn.Conv2d(inplane, outplane, k_size_conv, stride, padding = k_size_conv//2)#, padding_mode="SAME")
        self.batchnorm = nn.BatchNorm2d(outplane)
        self.max_pool = nn.MaxPool2d(k_size_pool, stride=stride)
        self.dropout_layer = nn.Dropout2d(1-keep_prob)
        self.activation_fn = nn.ELU()

    def forward(self, x):
        conv_f = self.activation_fn(self.batchnorm(self.conv(x)))
        pool_f = self.max_pool(x)
        out_f = torch.cat([conv_f, pool_f], dim=1)
        output = self.dropout_layer(out_f)
        return output

class Fire_vertical(nn.Module):
    def __init__(self, inplane, firstplane, secondplane, dilation=1, k_size=3, keep_prob=1):
        super(Fire_vertical, self).__init__()
        self.conv1 = nn.Conv2d(inplane, firstplane, [3, 1], padding=[1, 0])#, padding_mode="SAME")
        self.batchnorm1 = nn.BatchNorm2d(firstplane)
        self.wide_conv1 = nn.Conv2d(firstplane, secondplane, k_size, dilation=dilation, padding=k_size //2 * dilation)#, padding_mode="SAME")
        self.wbatchnorm1 = nn.BatchNorm2d(secondplane)
        self.wide_conv2 = nn.Conv2d(firstplane, secondplane, 1)#, padding_mode="SAME")
        self.wbatchnorm2 = nn.BatchNorm2d(secondplane)
        self.dropout_layer = nn.Dropout2d(1-keep_prob)
        self.activation_fn = nn.ELU()
    def forward(self, x):
        feature = self.activation_fn(self.batchnorm1(self.conv1(x)))
        x1 = self.activation_fn(self.wbatchnorm1(self.wide_conv1(feature)))
        x2 = self.activation_fn(self.wbatchnorm2(self.wide_conv2(feature)))
        output_f = torch.cat([x1, x2], dim=1)
        # import pdb;pdb.set_trace()
        output = self.dropout_layer(output_f)
        return output

class Fire_residual_vertical(nn.Module):
    def __init__(self, inplane, firstplane, secondplane, dilation=1, k_size_fire=3, k_size_residual=0, keep_prob=1):
        super(Fire_residual_vertical, self).__init__()
        self.k_size_residual = k_size_residual
        self.f_vert = Fire_vertical(inplane, firstplane, secondplane, dilation, k_size_fire)
        if self.k_size_residual > 0:
            self.conv1 = nn.Conv2d(inplane, secondplane * 2, k_size_residual, padding=k_size_residual//2)
            self.batchnorm1 = nn.BatchNorm2d(secondplane * 2)
        self.dropout_layer = nn.Dropout2d(1-keep_prob)
        self.activation_fn = nn.ELU()
    def forward(self, x):
        if self.k_size_residual > 0:
            output = self.f_vert(x) + self.activation_fn(self.batchnorm1(self.conv1(x)))
        else:
            output = self.f_vert(x) + x
        output = self.dropout_layer(output)
        return output

class Logic(nn.Module):
    def __init__(self, inplane, num_classes):
        super(Logic, self).__init__()
        self.conv_to_class = nn.Conv2d(inplane, num_classes, 1)
        self.batchnorm = nn.BatchNorm2d(num_classes)
        self.activation_fn = nn.ELU()
    def forward(self, x):
        return self.activation_fn(self.batchnorm(self.conv_to_class(x)))

class SF_Conv2d(nn.Module):
    def __init__(self, inplane, outplane, k_size=3, dilation=1, keep_prob=1):
        super(SF_Conv2d, self).__init__()
        self.conv = nn.Conv2d(inplane, outplane, k_size, dilation=dilation, padding=k_size//2 * dilation)
        self.batchnorm = nn.BatchNorm2d(outplane)
        self.activation_fn = nn.ELU()
        self.dropout_layer = nn.Dropout2d(1-keep_prob)
    def forward(self, x):
        x = self.activation_fn(self.batchnorm(self.conv(x)))
        return self.dropout_layer(x)

class Skip_connect(nn.Module):
    def __init__(self, inplane, outplane, k_size=3):
        super(Skip_connect, self).__init__()
        self.conv = nn.Conv2d(inplane, outplane, k_size, padding= k_size//2)
        self.batchnorm = nn.BatchNorm2d(outplane)
        self.activation_fn = nn.ELU()
    def forward(self, x, merge_sample):
        x_conv = self.activation_fn(self.batchnorm(self.conv(merge_sample)))
        return x + x_conv

