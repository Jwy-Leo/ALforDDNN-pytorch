import numpy as np
import torch

def Train(model, dataloader, optim, criteria):
    running_loss = []
    for index, (img, gt) in enumerate(dataloader):

        img, gt = img.float().cuda(), gt.long().cuda()
        segment_list = model(img)
        loss = criteria(segment_list, gt)
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
            loss = criteria(segment_list, gt)
            running_loss.append(loss.item())
    return np.mean(running_loss) 
