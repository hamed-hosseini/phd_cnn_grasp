import torch
import torch.nn as nn
import torch.nn.functional as F

class AlexNet(nn.Module):
    def __init__(self, input_channels=1):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=11, stride=4, padding=2),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            # nn.ReLU(inplace=True),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True),
            nn.Tanh(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True),
            nn.Tanh(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.Dropout(),
            # nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.Dropout(),
            # nn.ReLU(inplace=True),
            nn.Linear(4096, 6),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def compute_loss(self, pred, yc):
        p_loss = F.mse_loss(pred, yc)
        # pred_numpy = torch.Tensor.cpu(pred).detach().numpy()[:, :]
        # pos_pred = pred_numpy[:, 0:2]
        # cos_pred = pred_numpy[:,2]
        # sin_pred = pred_numpy[:, 3]
        # height = pred_numpy[:, 4]
        # width = pred_numpy[:, 5]

        pred_numpy = torch.Tensor.cpu(pred).detach().numpy()[:, :]
        pos_pred = pred_numpy[0, 0:2].flatten().tolist()
        cos_pred = pred_numpy[0, 2]
        sin_pred = pred_numpy[0, 3]
        height = pred_numpy[0, 4]
        width = pred_numpy[0, 5]

        # return {
        #     'loss': p_loss ,
        #     'losses': {
        #         'p_loss': p_loss
        #     }
        # }
        return {
            'loss': p_loss,
            'losses': {
                'p_loss': p_loss,
            }
            ,
            'pred': {
                'pos': pos_pred,
                'cos': cos_pred,
                'sin': sin_pred,
                'height': height,
                'width': width
            }
        }

