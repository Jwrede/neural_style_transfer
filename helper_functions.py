import torch
import torch.nn as nn

def std_and_mean(features, eps = 1e-5):
  size = features.size()
  N, C = size[:2]
  features_var = features.view(N, C, -1).var(dim=2) + eps
  features_std = features_var.sqrt().view(N, C, 1, 1)
  features_mean = features.view(N, C, -1).mean(dim = 2).view(N, C, 1, 1)
  return features_std, features_mean

def adaIN(x, y):
  size = x.size()
  std_y, mean_y = std_and_mean(y)
  std_x, mean_x = std_and_mean(x)
  normalized_feat = (x - mean_x.expand(size)) / std_x.expand(size)
  return normalized_feat * std_y.expand(size) + mean_y.expand(size)