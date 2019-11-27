import numpy as np
import cv2
import torch
from utils.dataset_processing.grasp import GraspRectangles, detect_grasps, Grasp
import math
from matplotlib import pyplot as plt
import os
def gridshow(name, imgs, scales, cmaps, width, border=10):
    """
    Display images in a grid.
    :param name: cv2 Window Name to update
    :param imgs: List of Images (np.ndarrays)
    :param scales: The min/max scale of images to properly scale the colormaps
    :param cmaps: List of cv2 Colormaps to apply
    :param width: Number of images in a row
    :param border: Border (pixels) between images.
    """
    imgrows = []
    imgcols = []

    maxh = 0
    for i, (img, cmap, scale) in enumerate(zip(imgs, cmaps, scales)):

        # Scale images into range 0-1
        if scale is not None:
            img = (np.clip(img, scale[0], scale[1]) - scale[0])/(scale[1]-scale[0])
        elif img.dtype == np.float:
            img = (img - img.min())/(img.max() - img.min() + 1e-6)

        # Apply colormap (if applicable) and convert to uint8
        if cmap is not None:
            try:
                imgc = cv2.applyColorMap((img * 255).astype(np.uint8), cmap)
            except:
                imgc = (img*255.0).astype(np.uint8)
        else:
            imgc = img

        if imgc.shape[0] == 3:
            imgc = imgc.transpose((1, 2, 0))
        elif imgc.shape[0] == 4:
            imgc = imgc[1:, :, :].transpose((1, 2, 0))

        # Arrange row of images.
        maxh = max(maxh, imgc.shape[0])
        imgcols.append(imgc)
        if i > 0 and i % width == (width-1):
            imgrows.append(np.hstack([np.pad(c, ((0, maxh - c.shape[0]), (border//2, border//2), (0, 0)), mode='constant') for c in imgcols]))
            imgcols = []
            maxh = 0

    # Unfinished row
    if imgcols:
        imgrows.append(np.hstack([np.pad(c, ((0, maxh - c.shape[0]), (border//2, border//2), (0, 0)), mode='constant') for c in imgcols]))

    maxw = max([c.shape[1] for c in imgrows])

    cv2.imshow(name, np.vstack([np.pad(r, ((border//2, border//2), (0, maxw - r.shape[1]), (0, 0)), mode='constant') for r in imgrows]))

def show_image(x, y, pred, data, didx, rot, zoom_factor,epoch, batch_idx, time, session, save=False):
    fig = plt.figure(session + '-' + str(epoch) + '-' + str(batch_idx))
    fig.suptitle(session, fontsize=14)
    ax = fig.add_subplot(311)
    # gtbbs = data.dataset.get_gtbb(didx, rot, zoom_factor)
    # gtbbs.show(ax)

    ax.set_title('prediction')
    pred_numpy = torch.Tensor.cpu(pred).detach().numpy()[:, :]
    pos_pred = pred_numpy[0, 0:2].flatten() * 224
    cos_pred =(pred_numpy[0, 2] - 0.5 ) * 2
    sin_pred = (pred_numpy[0, 3] - 0.5) * 2
    height = pred_numpy[0, 4] * 224
    width = pred_numpy[0, 5] * 224
    # print(pos_pred.tolist(), math.atan2(sin_pred, cos_pred), height, width)
    gr_pred = Grasp(pos_pred, math.atan2(sin_pred, cos_pred), height, width)
    gr_pred.plot(ax=ax)


    x_numpy = torch.Tensor.cpu(x).detach().numpy()[0,:,:,:]
    x_numpy = x_numpy.transpose((1, 2, 0))
    ax.imshow(x_numpy)
    # img = data.dataset.get_image(didx, rot, zoom_factor, normalise=False)
    # img.show(ax=ax)
    # pred_rect.plot(ax=ax)

    ax = fig.add_subplot(312)
    ax.set_title('target')
    ax.text(1, 1, 'rot={0}\n, zoom_factor={1}\n '.format(rot, zoom_factor), style='italic'
            )
    # gtbbs = data.dataset.get_gtbb(didx, rot, zoom_factor)
    # gtbbs.show(ax)

    pred_numpy = torch.Tensor.cpu(y).detach().numpy()[:, :]
    pos_pred = pred_numpy[0, 0:2].flatten() * 224
    cos_pred =(pred_numpy[0, 2] - 0.5) * 2
    sin_pred =( pred_numpy[0, 3] - 0.5) * 2
    height = pred_numpy[0, 4] * 224
    width = pred_numpy[0, 5] * 224
    # print(pos_pred.tolist(), math.atan2(sin_pred, cos_pred), height, width)
    gr_label = Grasp(pos_pred, math.atan2(sin_pred, cos_pred), height, width)
    ground_truth_bbs = data.dataset.get_gtbb(didx, rot, zoom_factor, normalise=False)
    if not isinstance(ground_truth_bbs, GraspRectangles):
        gt_bbs = GraspRectangles.load_from_array(ground_truth_bbs)
    else:
        gt_bbs = ground_truth_bbs
    max_io_gr = gr_pred.get_max_iou_gr(gt_bbs)
    if max_io_gr != None:
        max_io_gr.plot(ax=ax)


    x_numpy = torch.Tensor.cpu(x).detach().numpy()[0, :, :, :]
    x_numpy = x_numpy.transpose((1, 2, 0))
    ax.imshow(x_numpy)

    # img = data.dataset.get_image(didx, rot, zoom_factor, normalise=False)
    # img.show(ax=ax)

    ax = fig.add_subplot(313)
    ax.set_title('all targets')
    # print(didx)
    # print('99999999999', torch.Tensor.cpu(didx).detach().numpy(), 'type', type(didx))
    # first_image = torch.Tensor.cpu(didx).detach().numpy()[0]
    # print(torch.IntTensor([first_image]))
    gtbbs = data.dataset.get_gtbb(didx, rot, zoom_factor, normalise=False)
    gtbbs.show(ax)

    x_numpy = torch.Tensor.cpu(x).detach().numpy()[0, :, :, :]
    x_numpy = x_numpy.transpose((1, 2, 0))
    ax.imshow(x_numpy)
    # img = data.dataset.get_image(didx, rot, zoom_factor, normalise=False)
    # img.show(ax=ax)

    if save == True:
        if os.path.isdir(os.path.join(time, session)):
            pass
        else:
            os.makedirs(os.path.join(time, session))
        file_name = '{0}_{1}'.format(epoch, batch_idx)
        plt.savefig(os.path.join(time, session, file_name))
        plt.close('all')
    else:
        plt.show()
        plt.close('all')