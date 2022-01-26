import torch.nn as nn
from base_model import Basenet_volleyball
import torch

class VolleyballNN(nn.Module):
    def __init__(self, cfg, model=None):
        super(VolleyballNN, self).__init__()

        if cfg.only_last:
            # no training for pretrained model
            path = '/model_best.pth'
            self.pretrained_model = Basenet_volleyball(cfg)
            self.pretrained_model.loadmodel(path)
            for param in self.pretrained_model.parameters():
                param.requires_grad = False
        else:
            self.pretrained_model = model

        self.classifier_layer = nn.Sequential(
            nn.Dropout(p=0.05),
            nn.Linear(10,8)
        )

    def forward(self, inputs, inputs_ball, is_test=False):
        x = self.pretrained_model(inputs)
        action_scores = x[0]

        if is_test:
            inputs_ball = inputs_ball[:,5,:]
        else:
            inputs_ball = inputs_ball.squeeze(1)
            inputs_ball[:,0] = inputs_ball[:,0]/1920
            inputs_ball[:,1] = inputs_ball[:,1]/1080
            x = self.classifier_layer(torch.cat((x[1], inputs_ball), 1))
            return action_scores, x
