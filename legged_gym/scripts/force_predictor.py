import unittest
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import random

if torch.cuda.is_available():
  device = torch.device('cuda')
else: 
  device = torch.device('cpu')


class MLP(nn.Module):
  '''
    Multilayer Perceptron for regression.
  '''
  def __init__(self):
    super().__init__()
    # self.soft = nn.Softmax(dim=1)
    self.layers = nn.Sequential(
      nn.Linear(30, 64),
      nn.ReLU(),
      nn.Linear(64, 128),
      nn.ReLU(),
      nn.Linear(128, 64),
      nn.ReLU(),
      nn.Linear(64, 32),
      nn.ReLU(),
      nn.Linear(32, 1)
    )


  def forward(self, x):
    '''
      Forward pass
    '''
    return self.layers(x)

def test():
  predictor = MLP()

  predictor = torch.load('classifier_20.pth')

  x = torch.tensor([ 3.7442e-02, -1.4912e-03,  3.0854e-02,  9.9882e-01, -2.1341e-01,
          4.3084e-01, -6.8637e-05,  2.1363e+01, -1.1008e+00, -2.0857e+00]).to(device)

  print(predictor(x))
