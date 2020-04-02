def Vis_result(model, dataloader):
    with torch.no_grad():
        for img, gt in dataloader:
            b, c, w, h = gt.shape
            cv2.imshow('img', img.data.cpu().numpy()[0,...].transpose(1, 2, 0))
            cv2.imshow('gt', gt.data.cpu().numpy()[0,...].transpose(1, 2, 0))
            pred_list = model(img)
            pred = torch.mean(torch.cat([torch.nn.functional.interpolate(_pred, size=(w,h)) for _pred in pred_list], dim=1), dim=1)

            cv2.imshow('pred', pred.data.cpu().numpy()[0, ...])
            cv2.waitKey()
    return

