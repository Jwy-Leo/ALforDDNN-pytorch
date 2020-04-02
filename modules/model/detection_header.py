import torch
import torch.nn as nn

def uncertainty_spatial(model, _input, uncertainty_module):
    prob_map_output = model(_input)
    uncertainty_spatial = uncertainty_module(prob_map_output)
    return uncertainty_spatial

def MC_drop_Parser(model, _input, uncertainty_module):
    T = 15
    mc_logics_list = []
    for i in range(T):
        prob_map_output = model(_input)
        mc_prob_list.append(prob_map_output)
    uncertainty = uncertainty_module(mc_prob_list)
    return uncertainty

class Detection_header(nn.Module):
    def __init__(self):
        self.uncertainty_module = Uncertainty_header()
        pass
    def forward(self, logic_map_list):

        # Prediction generate
        prob_map_output = torch.cat([ torch.sigmoid(nn.functional.interpolate(_l,(x_list[1].shape[2], x_list[1].shape[3]))) for _l in logic_map_list], dim=1)

        # probs Smoothing
        # AvgPooling
        # prob_smooth = tf.nn.avg_pool(prob, (1, 3, 3, 1), [1] * 4, 'SAME')

        # spatial prob non-maximum supress
                    #         sz = 19
                    # ones = tf.ones_like(prob_smooth)
                    # zeros = tf.zeros_like(prob_smooth)

                    # prob_smooth_locally_max = tf.nn.max_pool(prob_smooth,
                    #                                          (1, sz, sz, 1),
                    #                                          (1, 1, 1, 1),
                    #                                          padding='SAME',
                    #                                          name='spatial_maximum')
                    # local_max = tf.where(tf.less_equal(prob_smooth_locally_max - prob_smooth, 1e-16), ones, zeros)


        # Probability threshold
        #  shp = tf.shape(pred_binary)
        #             sz = 19
        #             num_detections = tf.nn.depthwise_conv2d(pred_binary,
        #                                                     tf.ones((sz, sz, shp[-1], 1)),
        #                                                     (1, 1, 1, 1),
        #                                                     padding='SAME',
        #                                                     name='number_of_detection')
        #             min_detection = tf.fill(tf.shape(prob_smooth), self._tf_detection_min_detected_bbs)
        #             min_detection_supress = tf.where(tf.greater_equal(num_detections, min_detection), ones, zeros)
        # self.num_detections = 

        # detection map
        # detection_map = tf.multiply(tf.multiply(local_max, min_detection_supress), pred_binary)
        # return detection_map, prob_map, unc

class Uncertainty_header(nn.Module):
    def __init__(self, uncertainty_method='Mutual_information'):
        super(Uncertainty_header, self).__init__()
        self.uncertainty_method = uncertainty_method

        def Mutual_information(prob):
            ent_pixel = -1 * prob * torch.log(prob + 1e-8)
            ent_pixel = torch.nn.AvgPool2d(9, 1, padding = 9//2)(ent_pixel)
            mean_of_ent = torch.sum(ent_pixel, dim=1)
            
            prob_local = torch.nn.AvgPool2d(9, 1, padding=9//2)(prob)
            ent_of_mean = torch.sum(-1 * prob_local * torch.log(prob_local + 1e-8), dim=1)
            return ent_of_mean - mean_of_ent
        def Mutual_information_mc(probs):
            mean_of_ent_list = [-1 * prob * torch.log(prob + 1e-8) for prob in probs]
            mean_of_ent = torch.sum(torch.mean(torch.stack(mean_of_ent_list), dim=0),dim=1)
            
            mean_prob = torch.mean(torch.stack(probs), dim=0)
            ent_of_mean = torch.sum(-1 * mean_prob * torch.log(mean_prob + 1e-8), dim=1)
            return ent_of_mean - mean_of_ent

        self.default_uncertainty_method = {
            'entropy': lambda prob: torch.sum(-1 * prob * torch.log(prob + 1e-8), dim=1), #/ torch.log(prob.shape[1])
            'Mutual_information': Mutual_information,
            'MC_dropout': Mutual_information_mc
        }

    def forward(self, prob):
        # prob or probs(MC_drop)
        uncertainty = self.default_uncertainty_method[self.uncertainty_method](prob)
        return uncertainty
