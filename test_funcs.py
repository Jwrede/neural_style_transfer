import torch
import torch.nn as nn

from helper_functions import adaIN
from network import Net
from model import vgg, decoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as image
import cv2
from skimage import transform
from torch.utils.data import Dataset, DataLoader
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = Net(vgg, decoder).to(device)

net.load_state_dict(torch.load("model_checkpoint.pt", map_location=device)['state_dict'])

class ToTensor(object):
  def __call__(self, image):
    image = image.transpose((2,0,1))
    return image

def plot_data(data):
  if type(data) == list:
    fig, ax = plt.subplots(1,len(data))
    fig.set_size_inches(30,10)
    for i, im in enumerate(data):
      ax[i].imshow(im)
  else:
    fig, ax = plt.subplots(1,1)
    fig.set_size_inches(30,10)
    ax.imshow(data)

def test(content, style, alpha = 1.0, plot = True, encode = True):
  with torch.no_grad():
    if encode:
      toTensor = ToTensor()
      x = toTensor(content)
      x = np.expand_dims(x, 0)
      y = toTensor(style)
      y = np.expand_dims(y, 0)
      x = net.encode(torch.tensor(x).float().to(device))
      y = net.encode(torch.tensor(y).float().to(device))
      result = adaIN(x, y)
      result = (1-alpha) * x + alpha * result

    else:
      result = style

    result = net.decoder(result)
    result = result[0].to("cpu").numpy().transpose((1, 2, 0)).reshape(content.shape)
    if plot:
      plot_data([content, style, result])
    return result

def rgb_to_yiq(image):
  return image @ np.array([[0.299,0.587,0.114],[0.59590059,-0.27455667,-0.32134392],[0.21153661, -0.52273617, 0.31119955]])

def yiq_to_rgb(image):
  return image @ np.linalg.inv(np.array([[0.299,0.587,0.114],[0.59590059,-0.27455667,-0.32134392],[0.21153661, -0.52273617, 0.31119955]]))

def preserve_color_stable(content, result):
  L_c = rgb_to_yiq(content)

  L_result = rgb_to_yiq(result)
  L_result[..., 1] = L_c[..., 1]
  L_result[..., 2] = L_c[..., 2]
  result = yiq_to_rgb(L_result)

  return result
