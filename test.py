import torch
from test_funcs import *
import cv2

def style_image(content, style, alpha = 1.0, plot = True,  preserve_color = False):
  if preserve_color:
    result = test(content, style, alpha, plot = False)
    result = preserve_color_stable(content, result)
    if plot:
      plot_data([content, style, result])
    return result
  else:
    result = test(content, style, alpha, plot)
    return result

def style_interpolation(content, styles, weights, alpha = 1.0, plot = True, preserve_color = False):
  assert len(styles) == len(weights)
  assert sum(weights) > 0.99 and sum(weights) < 1.01
  assert [True for i in styles if i.shape == content.shape]

  toTensor = ToTensor()

  with torch.no_grad():
    x = torch.tensor(np.expand_dims(toTensor(content),0)).float().to(device)
    x = net.encode(x)

    for i,style in enumerate(styles):
      styles[i] = torch.tensor(np.expand_dims(toTensor(style),0)).float().to(device)

    for i,y in enumerate(styles):
      with torch.no_grad():
        y = net.encode(y)
        styles[i] = weights[i] * adaIN(x, y)
  
  result = torch.cat(styles)
  result = torch.sum(result, dim = 0, keepdims=True)

  if preserve_color:
    if plot:
      result = test(content, style, alpha, plot = False)
      result = preserve_color_stable(content, result)
    else:
      result = test(content, style, alpha, plot = False)
      return preserve_color_stable(content, result)
  else:
    if plot:
      result = test(content, result, alpha, plot = False, encode = False)
    else:
      return test(content, result, alpha, plot = False, encode = False)
  
  plot_data([content, result])
  return result


'''K = 7
styles = [transform.resize(plt.imread(f"style{i}.jpg"), content.shape[:-1]) for i in range(1,K+1)]
weights = [1/K for i in range(K)]
style_interpolation(content, styles, weights, plot = False, preserve_color = True)'''

import imageio
import imageio_ffmpeg
from tqdm import tqdm

def style_video(video_path, style, output_name = None, alpha = 1.0, preserve_color = False, custom_resolution = None):
  toTensor = ToTensor()

  reader = imageio.get_reader(video_path)
  resolution = None

  if custom_resolution is None:
    resolution = reader.get_meta_data()["source_size"]
  else:
    resolution = custom_resolution
  fps = reader.get_meta_data()["fps"]

  style = transform.resize(style,resolution)
  
  if not preserve_color:
    style_tensor = torch.tensor(np.expand_dims(toTensor(style), 0)).float().to(device)
    style_tensor = net.encode(style_tensor)
  
  frame_number = 0
  driving_video = []
  for frame_number in tqdm(range(reader.get_meta_data()['nframes'])):
    try:
      content = reader.get_next_data()
    except imageio.core.CannotReadFrameError:
      break
    except IndexError:
      break
    else:
      if custom_resolution is not None:
        content = transform.resize(content, list(resolution)[::-1])
      content = content.astype(float)/255
      if preserve_color:
        style_tensor = preserve_color_stable(content, style)
        style_tensor = torch.tensor(np.expand_dims(toTensor(style_tensor), 0)).float().to(device)
        style_tensor = net.encode(style_tensor)

      x = torch.tensor(toTensor(content)).unsqueeze(0).float().to(device)
      x = net.encode(x)
      result = adaIN(x, style_tensor)
      result = (1-alpha) * x + alpha * result
      result = test(content, result, alpha, False, False)
      result = cv2.convertScaleAbs(result*255)
      driving_video.append(result)
  reader.close()
  
  if output_name == None:
    return driving_video

  writer = imageio_ffmpeg.write_frames(f'{output_name}.mp4',
                                       (resolution), fps = fps,
                                       macro_block_size=1)
  writer.send(None)  # seed the generator
  for frame in driving_video:
    writer.send(frame)
  writer.close()
