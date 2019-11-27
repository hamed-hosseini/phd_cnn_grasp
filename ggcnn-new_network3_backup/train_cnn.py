import numpy as np
import datetime
import os
import sys
import argparse
import logging

# import cv2
from matplotlib import pyplot as plt
import torch
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

import tensorboardX

from utils.visualisation.gridshow import gridshow

from utils.dataset_processing import evaluation
from utils.data import get_dataset
from models import get_network
from models.common import post_process_output

logging.basicConfig(level=logging.INFO)

from utils.visualisation.gridshow import show_image
from datetime import datetime as dtime
now = dtime.now()
time = 'alex_net_18_depth_test'
class Args():
  network= 'ggcnn'
  dataset = 'cornell'
  dataset_path = "/home/taarlab-ros/Desktop/cornell_dataset"
  use_rgb = True
  use_depth = True
  split = 0.9
  ds_rotate = 0.0
  num_workers = 8
  batch_size = 128
  epochs = 25
  batches_per_epoch = 100
  val_batches = 1
  description = ''
  # outdir ='output/models/'
  # logdir = 'tensorboard/'
  vis = 0

def parse_args():
    parser = argparse.ArgumentParser(description='Train GG-CNN')

    # Network
    parser.add_argument('--network', type=str, default='ggcnn', help='Network Name in .models')

    # Dataset & Data & Training
    parser.add_argument('--dataset', type=str, help='Dataset Name ("cornell" or "jaquard")')
    parser.add_argument('--dataset-path', type=str, help='Path to dataset')
    parser.add_argument('--use-depth', type=int, default=1, help='Use Depth image for training (1/0)')
    parser.add_argument('--use-rgb', type=int, default=0, help='Use RGB image for training (0/1)')
    parser.add_argument('--split', type=float, default=0.9, help='Fraction of data for training (remainder is validation)')
    parser.add_argument('--ds-rotate', type=float, default=0.0,
                        help='Shift the start point of the dataset to use a different test/train split for cross validation.')
    parser.add_argument('--num-workers', type=int, default=16, help='Dataset workers')

    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=40, help='Training epochs')
    parser.add_argument('--batches-per-epoch', type=int, default=100, help='Batches per Epoch')
    parser.add_argument('--val-batches', type=int, default=10, help='Validation Batches')

    # Logging etc.
    parser.add_argument('--description', type=str, default='', help='Training description')
    parser.add_argument('--outdir', type=str, default='output/models/', help='Training Output Directory')
    parser.add_argument('--logdir', type=str, default='tensorboard/', help='Log directory')
    parser.add_argument('--vis', action='store_true', help='Visualise the training process')

    args = parser.parse_args()
    return args


def validate(net, device, val_data, batches_per_epoch, epoch, val_losses):
    """
    Run validation.
    :param net: Network
    :param device: Torch device
    :param val_data: Validation Dataset
    :param batches_per_epoch: Number of batches to run
    :return: Successes, Failures and Losses
    """
    net.eval()

    results = {
        'correct': 0,
        'failed': 0,
        'loss': 0,
        'losses': {

        }
    }

    ld = len(val_data)
    with torch.no_grad():
        batch_idx = 0
        while batch_idx < batches_per_epoch:
            for x, y, didx, rot, zoom_factor in val_data:
                batch_idx += 1
                # print('y', torch.Tensor.cpu(y).detach().numpy())
                # print('val didx:', didx, 'val batch_idx', batch_idx)
                if batches_per_epoch is not None and batch_idx >= batches_per_epoch:
                    break

                xc = x.to(device)

                yc = y.to(device)
                pred = net(xc)
                loss = F.mse_loss(pred, yc)
                results['loss'] += loss.item()/ld

                s = evaluation.calculate_iou_match_hamed(pred, val_data.dataset.get_gtbb(didx, rot, zoom_factor, normalise=False))

                if s:
                    results['correct'] += 1
                else:
                    results['failed'] += 1

                if batch_idx % 10 == 0:
                    logging.info('Batch: {}, Loss: {:0.4f}'.format(batch_idx, F.mse_loss(pred, yc)))
                    val_losses.append(F.mse_loss(pred, yc).item())
                    show_image(xc, yc, pred, val_data, didx, rot, zoom_factor, epoch, batch_idx, time, 'validation',  True)
                # print('pred', pred)
                #printing Loss_validation
                # print('vallos:', F.mse_loss(pred, yc))
    return results, val_losses


def train(epoch, net, device, train_data, optimizer, batches_per_epoch, losses, time, vis=False):
    """
    Run one training epoch
    :param epoch: Current epoch
    :param net: Network
    :param device: Torch device
    :param train_data: Training Dataset
    :param optimizer: Optimizer
    :param batches_per_epoch:  Data batches to train on
    :param vis:  Visualise training progress
    :return:  Average Losses for Epoch
    """
    results = {
        'loss': 0,
        'losses': {
        }
    }
    net.train()

    batch_idx = 0
    # Use batches per epoch to make training on different sized datasets (cornell/jacquard) more equivalent.
    while batch_idx < batches_per_epoch:
        for x, y, idx, rot, zoom_factor in train_data:
            batch_idx += 1
            if batch_idx >= batches_per_epoch:
                break

            xc = x.to(device)

            pred = net(xc)
            yc = y.to(device)

            # criteration = nn.NLLLoss()
            loss = F.mse_loss(pred, yc)
            if batch_idx % 10 == 0:
                logging.info('Epoch: {}, Batch: {}, Loss: {:0.4f}'.format(epoch, batch_idx, loss.item()))
                # show_image(xc, y, pred, train_data,idx, rot, zoom_factor,epoch, batch_idx, time, 'train', save=True)
                losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    results['loss'] /= batch_idx
    for l in results['losses']:
        results['losses'][l] /= batch_idx

    return results, losses


def run():
    # args = parse_args()
    args = Args()
    # Set-up output directories
    dt = datetime.datetime.now().strftime('%y%m%d_%H%M')
    net_desc = '{}_{}'.format(dt, '_'.join(args.description.split()))

    # save_folder = os.path.join(args.outdir, net_desc)
    # if not os.path.exists(save_folder):
    #     os.makedirs(save_folder)
    # tb = tensorboardX.SummaryWriter(os.path.join(args.logdir, net_desc))

    # Load Dataset
    print('arg dataset:', args.dataset)
    logging.info('Loading {} Dataset...'.format(args.dataset.title()))
    Dataset = get_dataset(args.dataset)

    train_dataset = Dataset(args.dataset_path, start=0.0, end=args.split, ds_rotate=args.ds_rotate,
                            random_rotate=True, random_zoom=True,
                            include_depth=args.use_depth, include_rgb=args.use_rgb)


    train_data = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    val_dataset = Dataset(args.dataset_path, start=args.split, end=1.0, ds_rotate=args.ds_rotate,
                          random_rotate=True, random_zoom=True,
                          include_depth=args.use_depth, include_rgb=args.use_rgb)
    val_data = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.val_batches,
        shuffle=True,
        num_workers=args.num_workers
    )

    logging.info('Done')

    # Load the network
    logging.info('Loading Network...')
    input_channels = 1 * args.use_depth + 3 * args.use_rgb


    # ggcnn = get_network('alexnet')
    # net = ggcnn(input_channels = input_channels)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    # n_inputs = 1000
    from torchvision import models
    net = models.alexnet(pretrained=True)

    # Freeze model weights
    for param in net.parameters():
        param.requires_grad = False
    # Add on classifier
    # print('++++++++++\n',net, '++++++++++++++++\n')
    net.classifier = nn.Sequential(
        nn.Linear(9216, 512),
        nn.Tanh(),
        nn.Dropout(0.5),
        nn.Linear(512, 512),
        nn.Tanh(),
        nn.Dropout(0.5),
        nn.Linear(512, 6),
        nn.Tanh()
        )
    # print(net)
    # net.classifier[6].requires_grad = True
    net = net.to(device)
    # optimizer = optim.Adam(net.parameters(), lr=0.05)
    optimizer = optim.Adam(net.parameters(), lr=0.0005)
    # optimizer = optim.Adam()
    logging.info('Done')

    # Print model architecture.
    # summary(net, (input_channels, 640, 480))
    # f = open(os.path.join(save_folder, 'arch.txt'), 'w')
    # sys.stdout = f
    # summary(net, (input_channels, 640, 480))
    # sys.stdout = sys.__stdout__
    # f.close()

    best_iou = 0.0
    train_losses = []
    val_losses = []
    for epoch in range(args.epochs):
        logging.info('Beginning Epoch {:02d}'.format(epoch))
        train_results, train_losses = train(epoch, net, device, train_data, optimizer, args.batches_per_epoch, train_losses, time,  vis=args.vis)

        # Log training losses to tensorboard
        # tb.add_scalar('loss/train_loss', train_results['loss'], epoch)
        # for n, l in train_results['losses'].items():
        #     tb.add_scalar('train_loss/' + n, l, epoch)

        # Run Validation
        logging.info('Validating...')
        test_results, val_losses = validate(net, device, val_data, args.batches_per_epoch, epoch, val_losses)
        logging.info('%d/%d = %f' % (test_results['correct'], test_results['correct'] + test_results['failed'],
                                     test_results['correct']/(test_results['correct'] + test_results['failed'])))

        # Log validation results to tensorbaord
        # tb.add_scalar('loss/IOU', test_results['correct'] / (test_results['correct'] + test_results['failed']), epoch)
        # tb.add_scalar('loss/val_loss', test_results['loss'], epoch)
        # for n, l in test_results['losses'].items():
        #     tb.add_scalar('val_loss/' + n, l, epoch)

        # Save best performing network
        iou = test_results['correct'] / (test_results['correct'] + test_results['failed'])
        # if iou > best_iou or epoch == 0 or (epoch % 10) == 0:
        #     torch.save(net, os.path.join(save_folder, 'epoch_%02d_iou_%0.2f' % (epoch, iou)))
        #     best_iou = iou
    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax.title.set_text('train_loss')
    ax.plot(train_losses)
    ax = fig.add_subplot(212)
    ax.plot(val_losses)
    ax.title.set_text('val_loss')
    plt.savefig('total_Losses')
if __name__ == '__main__':
    run()