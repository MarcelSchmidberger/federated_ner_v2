import syft as sy

import torch as th
from torch import nn
import torch.nn.functional as F 

def set_model_params(module, params_list, start_param_idx=0):
    """ Set params list into model recursively
    """
    param_idx = start_param_idx

    for name, param in module._parameters.items():
        module._parameters[name] = params_list[param_idx]
        param_idx += 1

    for name, child in module._modules.items():
        if child is not None:
            param_idx = set_model_params(child, params_list, param_idx)

    return param_idx


class Net(nn.Module):
    def __init__(self, word_embeddings=None):
        super(Net, self).__init__()
        self.embedding_dim = 300
        self.embedding = nn.Embedding.from_pretrained(word_embeddings)
        self.fc1 = nn.Linear(300, 3)
        self.fc2 = nn.Linear(3, 2)

    def forward(self, x):
        x = self.embedding(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def softmax_cross_entropy_with_logits(logits, targets, batch_size):
    """ Calculates softmax entropy
        Args:
            * logits: (NxC) outputs of dense layer
            * targets: (NxC) one-hot encoded labels
            * batch_size: value of N, temporarily required because Plan cannot trace .shape
    """
    # numstable logsoftmax
    norm_logits = logits - logits.max()
    log_probs = norm_logits - norm_logits.exp().sum(dim=1, keepdim=True).log()
    # NLL, reduction = mean
    return -(targets * log_probs).sum() / batch_size


def naive_sgd(param, **kwargs):
    return param - kwargs['lr'] * param.grad