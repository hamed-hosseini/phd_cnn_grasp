import numpy as np

import torch
import torch.utils.data

import random


class GraspDatasetBase(torch.utils.data.Dataset):
    """
    An abstract dataset for training GG-CNNs in a common format.
    """
    def __init__(self, output_size=224, include_depth=False, include_rgb=True, random_rotate=False,
                 random_zoom=False, input_only=False):
        """
        :param output_size: Image output size in pixels (square)
        :param include_depth: Whether depth image is included
        :param include_rgb: Whether RGB image is included
        :param random_rotate: Whether random rotations are applied
        :param random_zoom: Whether random zooms are applied
        :param input_only: Whether to return only the network input (no labels)
        """
        self.output_size = output_size
        self.random_rotate = random_rotate
        self.random_zoom = random_zoom
        self.input_only = input_only
        self.include_depth = include_depth
        self.include_rgb = include_rgb

        self.grasp_files = []

        if include_depth is False and include_rgb is False:
            raise ValueError('At least one of Depth or RGB must be specified.')

    @staticmethod
    def numpy_to_torch(s):
        if len(s.shape) == 2:
            return torch.from_numpy(np.expand_dims(s, 0).astype(np.float32))
        else:
            return torch.from_numpy(s.astype(np.float32))

    def get_gtbb(self, idx, rot=0, zoom=1.0):
        raise NotImplementedError()

    def get_depth(self, idx, rot=0, zoom=1.0):
        raise NotImplementedError()

    def get_rgb(self, idx, rot=0, zoom=1.0):
        raise NotImplementedError()

    def __getitem__(self, idx):
        if self.random_rotate:
            rotations = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi,  5*np.pi/4, 3*np.pi/2, 7*np.pi/4]
            rot = random.choice(rotations)
            # rot = np.random.uniform(0, 2 * np.pi)
        else:
            rot = 0.0

        if self.random_zoom:
            zooms = [0.5, 0.6, 0.7, 0.8, 0.9, 1]
            zoom_factor = random.choice(zooms)
        else:
            zoom_factor = 1.0

        # Load the depth image
        if self.include_depth:
            depth_img = self.get_depth(idx, rot, zoom_factor)

        # Load the RGB image
        if self.include_rgb:
            rgb_img = self.get_rgb(idx, rot, zoom_factor)

        # Load the grasps
        bbs = self.get_gtbb(idx, rot, zoom_factor)

        # pos_img, ang_img, width_img = bbs.draw((self.output_size, self.output_size))
        # width_img = np.clip(width_img, 0.0, 150.0)/150.0



        if self.include_depth and self.include_rgb:
            x = self.numpy_to_torch(
                np.concatenate(
                    (rgb_img[0:2, :, :],
                     np.expand_dims(depth_img, 0)
                     ),
                    0
                )
            )
        elif self.include_depth:
            x = self.numpy_to_torch(depth_img)
        elif self.include_rgb:
            x = self.numpy_to_torch(rgb_img)
        my_gr = random.choice(bbs.grs)
        center_pos = my_gr.center
        cos_angle = (np.cos(my_gr.angle) + 1) / 2
        sin_angle = (np.sin(my_gr.angle) + 1) / 2
        length = my_gr.length
        width = my_gr.width

        ground_truth = np.append(center_pos, cos_angle)
        ground_truth = np.append(ground_truth, sin_angle)
        ground_truth = np.append(ground_truth, length)
        ground_truth = np.append(ground_truth, width)


        # pos = self.numpy_to_torch(pos_img)
        # cos = self.numpy_to_torch(np.cos(2*ang_img))
        # sin = self.numpy_to_torch(np.sin(2*ang_img))
        # width = self.numpy_to_torch(width_img)
        ground_truth_torch = self.numpy_to_torch(ground_truth)
        return x, ground_truth_torch, idx, rot, zoom_factor

    def __len__(self):
        return len(self.grasp_files)
