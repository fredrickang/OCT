import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

__all__ = ['np', 'torch', 'nn','F', 'Variable', 'labels']

labels = np.array(['Normal', 'observation-Dry', 'observation-Wet', 'anti-VEGF'])

yGrad = {}