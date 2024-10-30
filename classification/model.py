import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.criterions.label_smoothed_cross_entropy import label_smoothed_nll_loss

from factory.calculator_factory import create_calculator
from factory.queue_factory import create_queue
from resnet import *

class MyModel(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        dropout = 0.
        self.hidden = nn.Sequential(
            nn.Linear(in_features=self.args.embedding_dim, out_features=self.args.class_num),
        )
       
        self.ce = torch.nn.CrossEntropyLoss().cuda()
        
    def forward(self, x1, x2, label):
        out_x1 = self.hidden(x1)
        # out_x2 = self.key_hidden(x2)

        class_loss = self.ce(out_x1, label)
        loss = class_loss

        return out_x1, loss, class_loss
    
    def encode(self, x):
        output = self.hidden(x)
        return output
        
    def fc_encode(self, x, labels):
        output = self.fc2(x)
        ce_loss = self.ce(output, labels)
        return output, ce_loss 

    @torch.no_grad()
    def _momentum_update_key_hidden(self):
        state_dict_q = self.hidden.state_dict()
        state_dict_k = self.key_hidden.state_dict()

        for name_k, name_q in self.k2q_mapping.items():
            param_q = state_dict_q[name_q]
            param_k = state_dict_k[name_k]
            param_k.data.copy_(param_q.data * (1. - self.args.m) + param_k.data * self.args.m)
        
