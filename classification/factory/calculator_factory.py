import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Sequential

def create_calculator(T=1, num_classes=1, dim=64, args=None):
    """
    T: temperature
    num_classes: voc size
    dim: embedding size hidden
    """
    
    return ContrastByClassCalculator(T, num_classes, dim, args)

class LogitsLabelsCalculator(object):
    def calculate(self, q_outs):
        raise NotImplementedError

class ContrastByClassCalculator():
    def __init__(self, T, num_classes, dim, args) -> None:
        super().__init__()
        self.T = T
        self.num_classes = num_classes
        self.dim = dim
        self.args = args

    def calculate(self, q_outs, k, queue, cls_labels, **kwargs):
        q = q_outs
        queue = queue[:self.num_classes, :, :]

        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)

        labels_onehot = torch.zeros((cls_labels.shape[0], self.num_classes)).cuda().scatter(
            1, cls_labels.unsqueeze(1), 1
        )

        q_onehot = labels_onehot.unsqueeze(-1) * q.unsqueeze(1)
        l_neg = torch.einsum('ncd, cdk->nk', q_onehot, queue)
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.T
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        return logits, labels
