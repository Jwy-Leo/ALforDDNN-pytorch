import torch
import torch.nn as nn

def optim_and_criteria(args, model):
    optim_type = args.config['training_parameters']['optim']
    default_optim_class = {
        "Adam": torch.optim.Adam
    }
    optim_parameters_setting = {
        "Adam": {
            'params': model.parameters(),
            'lr': args.config['training_parameters']['lr']
        }
    }
    #optim = default_optim_class[optim_type](model.parameters(), lr=args.config['training_parameters']['lr'])
    optim = default_optim_class[optim_type](**optim_parameters_setting[optim_type])
    criteria = criteria_seg #torch.nn.CrossEntropy()
    return optim, criteria

def criteria_seg(_input, _target):
    b, c, w, h = _input.shape
    _target_resize = torch.nn.functional.interpolate(_target.float(), size=(w, h)).long()
    _target_resize = _target_resize.repeat(1, c, 1, 1)
    loss = torch.nn.BCELoss()(_input, _target_resize.float())
    '''
    for index, _input_l in enumerate(_input):
        # if index >1 : break
        import pdb;pdb.set_trace()
        b, c, w, h = _input_l.shape

        _target_resize = torch.nn.functional.interpolate(_target.float(), size=(w, h)).long().squeeze()
        if index != 0:
            import pdb;pdb.set_trace()
            loss += nn.CrossEntropyLoss()(_input_l, _target_resize)
        else:
            loss = nn.CrossEntropyLoss()(_input_l, _target_resize)

        # _target_resize = torch.nn.functional.interpolate(_target.float(), size=(w, h)).float()
        # loss += nn.MSELoss()(_input_l, _target_resize)
    '''
    return loss
