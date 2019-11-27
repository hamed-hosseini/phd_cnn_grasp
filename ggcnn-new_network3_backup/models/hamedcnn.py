import torch.nn as nn
import torch.nn.functional as F

# filter_sizes = [32, 16, 8, 8, 16, 32]
# kernel_sizes = [9, 5, 3, 3, 5, 9]
# strides = [3, 2, 2, 2, 2, 3]
#

filter_sizes = [64, 128, 128, 128, 256]
kernel_sizes = [3, 3, 3, 3, 1]
strides = [2, 2, 2, 2, 2]


class HamedCNN(nn.Module):
    """
    GG-CNN
    Equivalient to the Keras Model used in the RSS Paper (https://arxiv.org/abs/1804.05172)
    """
    def __init__(self, input_channels=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, filter_sizes[0], kernel_sizes[0], stride=strides[0], padding=3)
        self.conv2 = nn.Conv2d(filter_sizes[0], filter_sizes[1], kernel_sizes[1], stride=strides[1], padding=2)
        self.pool1 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(filter_sizes[1], filter_sizes[2], kernel_sizes[2], stride=strides[2], padding=1)
        self.conv4 = nn.Conv2d(filter_sizes[2], filter_sizes[3], kernel_sizes[3], stride=strides[3], padding=1)
        self.conv5 = nn.Conv2d(filter_sizes[3], filter_sizes[4], kernel_sizes[4], stride=strides[4], padding=1)
        self.pool2 = nn.MaxPool2d(2)
        self.fc = nn.Sequential(
            nn.Linear(4096, 100),
            nn.ReLU(),
            nn.Linear(100, 6),
            # nn.Softmax()
        )
        #
        # self.pos_output = nn.Linear(100, 2)
        # self.cos_output = nn.Linear(100, 1)
        # self.sin_output = nn.Linear(100, 1)
        # self.width_output = nn.Linear(100, 1)

        # self.pos_output = nn.Conv2d(filter_sizes[4], 1, kernel_size=2)
        # self.cos_output = nn.Conv2d(filter_sizes[4], 1, kernel_size=2)
        # self.sin_output = nn.Conv2d(filter_sizes[4], 1, kernel_size=2)
        # self.width_output = nn.Conv2d(filter_sizes[4], 1, kernel_size=2)

        # for m in self.modules():
        #     if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        #         nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.pool1(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.pool2(x))
        # x = self.fc(x)
        # x = F.relu(self.conv6(x))

        # pos_output = self.pos_output(x)
        # cos_output = self.cos_output(x)
        # sin_output = self.sin_output(x)
        # width_output = self.width_output(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def compute_loss(self, xc, yc):
        # y_pos, y_cos, y_sin, y_width = yc
        # pos_pred, cos_pred, sin_pred, width_pred = self(xc)

        p_loss = F.mse_loss(xc, yc)
        # cos_loss = F.mse_loss(cos_pred, y_cos)
        # sin_loss = F.mse_loss(sin_pred, y_sin)
        # width_loss = F.mse_loss(width_pred, y_width)

        # return {
        #     'loss': p_loss + cos_loss + sin_loss + width_loss,
        #     'losses': {
        #         'p_loss': p_loss,
        #         'cos_loss': cos_loss,
        #         'sin_loss': sin_loss,
        #         'width_loss': width_loss
        #     },
        #     'pred': {
        #         'pos': pos_pred,
        #         'cos': cos_pred,
        #         'sin': sin_pred,
        #         'width': width_pred
        #     }
        # }
        #

        return {
            'loss': p_loss ,
            'losses': {
                'p_loss': p_loss
            }
        }
