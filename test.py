import torch
from test_funcs import *
import cv2

def style_image(content, style, alpha = 1.0, plot = True,  color = None):
  if color == "preserve color 2":
    result = test(content, style, alpha, plot = False)
    result = preserve_color_stable(content, result)
    if plot:
      plot_data([content, style, result])
    else:
      return result

  elif color == "preserve color 1":
    result = preserve_color(content, style)
    result = test(content, result, alpha)
    if not plot:
      return result
  else:
    result = test(content, style, alpha, plot)
    if not plot:
      return result

def style_interpolation(content, styles, weights, alpha = 1.0, plot = True, color = None):
  assert len(styles) == len(weights)
  assert sum(weights) > 0.99 and sum(weights) < 1.01
  assert [True for i in styles if i.shape == content.shape]

  toTensor = ToTensor()

  with torch.no_grad():
    x = torch.tensor(np.expand_dims(toTensor(content),0)).float().to(device)
    x = net.encode(x)

    for i,style in enumerate(styles):
      if color == "preserve color 1":
        style = preserve_color(content, style)
      styles[i] = torch.tensor(np.expand_dims(toTensor(style),0)).float().to(device)

    for i,y in enumerate(styles):
      with torch.no_grad():
        y = net.encode(y)
        styles[i] = weights[i] * adaIN(x, y)
  
  result = torch.cat(styles)
  result = torch.sum(result, dim = 0, keepdims=True)

  if color == "preserve color 2":
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


'''K = 7
styles = [transform.resize(plt.imread(f"style{i}.jpg"), content.shape[:-1]) for i in range(1,K+1)]
weights = [1/K for i in range(K)]
style_interpolation(content, styles, weights, color = "preserve color 2")'''

def style_video(video_path, output_name, style, alpha = 1.0, color = None):
  cap = cv2.VideoCapture(video_path)
  frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  fps = int(cap.get(cv2.CAP_PROP_FPS))
  frame_number = 0
  frame_width = int( cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  frame_height =int( cap.get( cv2.CAP_PROP_FRAME_HEIGHT))
  resolution = (frame_height, frame_width)

  toTensor = ToTensor()

  net.to(device)

  style = transform.resize(style,resolution)
  
  if color not in ["preserve color 1", "preserve color 2"]:
    style_tensor = torch.tensor(np.expand_dims(toTensor(style), 0)).float().to(device)
    style_tensor = net.encode(style_tensor)

  fourcc = cv2.VideoWriter_fourcc(*'XVID')
  
  out = cv2.VideoWriter(f"{output_name}.mp4", fourcc, fps, (frame_width, frame_height), isColor = True)

  while frame_number < frame_count:
    cap.set(fps, frame_number)
    success, content = cap.read()
    content = content.astype(float)/255
    if success:
      if color in ["preserve color 1", "preserve color 2"]:
        if color == "preserve color 1":
          style_tensor = preserve_color(content, style)
        elif color == "preserve color 2":
          style_tensor = preserve_color_stable(content, style)
        style_tensor = torch.tensor(np.expand_dims(toTensor(style_tensor), 0)).float().to(device)
        style_tensor = net.encode(style_tensor)

      x = torch.tensor(toTensor(content)).unsqueeze(0).float().to(device)
      x = net.encode(x)
      result = adaIN(x, style_tensor)
      result = (1-alpha) * x + alpha * result
      result = test(content, result, alpha, False, False)
      result = cv2.convertScaleAbs(result*255)
      out.write(result)
      print(f"{frame_number+1} of {frame_count}")
      frame_number += 1
    else:
      print("conversion failed!")
  
  cap.release()
  out.release()
  cv2.destroyAllWindows()