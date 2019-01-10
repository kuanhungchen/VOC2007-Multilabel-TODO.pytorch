import os
import time
import numpy as np

import torch
import torch.nn.functional
from torch import nn, Tensor
from torchvision import models


class Model(nn.Module):

    def __init__(self, num_classes=20):
        super(Model, self).__init__()
        # TODO: CODE BEGIN
        # raise NotImplementedError
        vgg16 = models.vgg16(pretrained=True)
        classifier = list(vgg16.classifier.children())
        classifier.pop()
        classifier.append(nn.Linear(4096, num_classes))
        vgg16.classifier = nn.Sequential(*classifier)
        self._network = vgg16
        # TODO: CODE END

    def forward(self, images: Tensor) -> Tensor:
        # TODO: CODE BEGIN
        # logits = xxx
        # raise NotImplementedError
        logits = self._network.forward(images)
        # TODO: CODE END

        return logits

    def loss(self, logits: Tensor, multilabels: Tensor) -> Tensor:
        # TODO: CODE BEGIN
        # raise NotImplementedError
        loss = torch.nn.functional.binary_cross_entropy(torch.sigmoid(logits), multilabels)

        return loss
        # TODO: CODE END

    def save(self, path_to_checkpoints_dir: str, step: int) -> str:
        path_to_checkpoint = os.path.join(path_to_checkpoints_dir,
                                          'model-{:s}-{:d}.pth'.format(time.strftime('%Y%m%d%H%M'), step))
        torch.save(self.state_dict(), path_to_checkpoint)
        return path_to_checkpoint

    def load(self, path_to_checkpoint: str) -> 'Model':
        self.load_state_dict(torch.load(path_to_checkpoint))
        return self
