import torch
import torch.nn as nn
from helper_functions import adaIN

class Net(nn.Module):
  def __init__(self, encoder, decoder):
    super(Net, self).__init__()
    encoder_layers = list(encoder.children())

    self.encoder1 = nn.Sequential(*encoder_layers[:4])
    self.encoder2 = nn.Sequential(*encoder_layers[4:11])
    self.encoder3 = nn.Sequential(*encoder_layers[11:18])
    self.encoder4 = nn.Sequential(*encoder_layers[18:31])
    for nnet in [self.encoder1, self.encoder2, self.encoder3, self.encoder4]:
      for layer in nnet.modules():
        if type(layer) != nn.Sequential:
          layer.requires_grad = False
    
    self.decoder = decoder
    self.loss = nn.MSELoss()

  def encode_with_layers(self, input):
    enc_list = [input]
    for i,nnet in enumerate([self.encoder1, self.encoder2,
                             self.encoder3, self.encoder4]):
      enc_list.append(nnet(enc_list[i]))
    return enc_list

  def encode(self, input):
    enc_input = input
    for nnet in [self.encoder1, self.encoder2, self.encoder3, self.encoder4]:
      enc_input = nnet(enc_input)
    return enc_input

  def content_loss(self, input, target):
    return self.loss(input, target)

  def style_loss(self, input, style_target):
    input_std, input_mean = std_and_mean(input)
    style_std, style_mean = std_and_mean(style_target)
    return self.loss(input_mean, style_mean) + self.loss(input_std, style_std)

  def forward(self, content, style, alpha = 1.0, full = True):
    enc_x = self.encode(content)
    enc_y = self.encode_with_layers(style)

    target = adaIN(enc_x, enc_y[-1])
    target = (1-alpha) * enc_x + alpha * target
    dec = self.decoder(target)
    dec_layers = self.encode_with_layers(dec)

    content_loss = self.content_loss(dec_layers[-1], target)
    style_loss = self.style_loss(dec_layers[1], enc_y[1])
    for i in range(2,5):
      style_loss += self.style_loss(dec_layers[i], enc_y[i])
    return dec, content_loss, style_loss