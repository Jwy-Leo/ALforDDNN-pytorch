import torch
import torch.nn as nn
from modules.model.base_modules import *
# from base_modules import *
def main():
    model = FireResidualDetection(3)
    input_sample = torch.randn((3, 3, 640, 480))
    output = model(input_sample)

class FireResidualDetection(nn.Module):
    def __init__(self, inplane):
        super(FireResidualDetection, self).__init__()
        outplane_list = [32, 64, 128, 128]
        self.feature_extractor = Feature_extractor(inplane, outplane_list)
        self.deconv = Deconv(inplane_list=outplane_list)

    def forward(self, x):
        feature_out_list = self.feature_extractor(x)
        logic_prob_map_list = self.deconv(feature_out_list)
        return logic_prob_map_list

class Feature_extractor(nn.Module):
    def __init__(self, inplane, outplane_list):
        super(Feature_extractor, self).__init__()
        assert len(outplane_list) == 4
        # self.FE_modules1 = nn.Sequential( *self._make_modules(inplane, 61, 3, 2, 32, 32) )
        # self.FE_modules2 = nn.Sequential( *self._make_modules(32, 64, 3, 3, 64, 64) )
        # self.FE_modules3 = nn.Sequential( *self._make_modules(64, 128, 3, 3, 128, 128) )
        # self.FE_modules4 = nn.Sequential( *self._make_modules(128, 128, 3, 3, 128, 128) )
        # self.FE_modules5 = nn.Sequential( *self._make_modules(128, 128, 3, 2, 128, 128, dilation=2, use_downsample_layer=False) )
        self.FE_modules1 = nn.Sequential( *self._make_modules(inplane, outplane_list[0] * 2 - inplane, 3, 2, outplane_list[0], outplane_list[0]) )
        self.FE_modules2 = nn.Sequential( *self._make_modules(outplane_list[0] * 2, outplane_list[1], 3, 3, outplane_list[1], outplane_list[1]) )
        self.FE_modules3 = nn.Sequential( *self._make_modules(outplane_list[1] * 2, outplane_list[2], 3, 3, outplane_list[2], outplane_list[2]) )
        self.FE_modules4 = nn.Sequential( *self._make_modules(outplane_list[2] * 2, outplane_list[3], 3, 3, outplane_list[3], outplane_list[3], dilation=2, use_downsample_layer=False) )
        # self.FE_modules5 = nn.Sequential( *self._make_modules(outplane_list[3] * 2, outplane_list[4], 3, 2, outplane_list[4], outplane_list[4], dilation=2, use_downsample_layer=False) )
    def forward(self, x):
        x1 = self.FE_modules1(x)
        x2 = self.FE_modules2(x1)
        x3 = self.FE_modules3(x2)
        x4 = self.FE_modules4(x3)
        return x1, x2, x3, x4
        # return x4
    def _make_modules(self, inplane,
                            downsample_outplane, downsample_ksize,
                            residual_layers, r_outplane_1, r_outplane_2,
                            dilation=1, keep_prob=1, use_downsample_layer=True):
        layers_list = []
        if use_downsample_layer:
            layers_list.append(
                Downsample(inplane, downsample_outplane, k_size_conv=downsample_ksize)
            )
        get_inplane = downsample_outplane + inplane if use_downsample_layer else inplane
        for i in range(residual_layers):
            if i != 0:
                _layer = Fire_residual_vertical(r_outplane_2 * 2, r_outplane_1, r_outplane_2, dilation=dilation)
            else:
                _layer = Fire_residual_vertical(get_inplane, r_outplane_1, r_outplane_2, dilation=dilation)
            layers_list.append( _layer)
        return layers_list
 
class Deconv(nn.Module):
    def __init__(self, inplane_list, num_classes=1):
        super(Deconv, self).__init__()
        self.dconv1 = SF_Conv2d(256, 128, 1)
        self.dconv2 = SF_Conv2d(128, 128, 1)
        self.sc2 = Skip_connect(256, 128)
        self.dconv3 = SF_Conv2d(128, 128, 1)
        self.sc3 = Skip_connect(256, 128)
        self.dconv4 = SF_Conv2d(128, 128, 1)
        self.sc4 = Skip_connect(256, 128)
        self.dconv5 = SF_Conv2d(128, 128, 1)
        self.sc5 = Skip_connect(128, 128)
        self.Logic1 = Logic(128, num_classes)
        self.Logic2 = Logic(128, num_classes)
        self.Logic3 = Logic(128, num_classes)
        self.Logic4 = Logic(128, num_classes)
        self.Logic5 = Logic(128, num_classes)

    def forward(self, x_list):
        x1 = self.dconv1(x_list[3])
        x2 = self.sc2(self.dconv2(x1), x_list[3])
        x3 = self.sc3(self.dconv3(x2), x_list[2])
        x4 = self.sc4(self.dconv4(x3), x_list[2])
        up_feature = nn.functional.upsample(x4,(x_list[1].shape[2], x_list[1].shape[3]))
        x5 = self.dconv5(up_feature)
        x5 = self.sc5(x5, x_list[1])

        l1 = self.Logic1(x1)
        l2 = self.Logic2(x2)
        l3 = self.Logic3(x3)
        l4 = self.Logic4(x4)
        l5 = self.Logic5(x5)
        l_list = [l1, l2, l3, l4, l5]

        prob_map_output = torch.cat([ nn.functional.interpolate(_l, (x_list[-1].shape[2], x_list[-1].shape[3])) for _l in l_list ], dim=1)
        prob_map_output = torch.sigmoid(prob_map_output)
        
        return prob_map_output


if __name__ == "__main__":
    main()
