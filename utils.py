import os
import numpy as np
import cPickle as pickle
import cv2
import json
from skimage.io import imsave

def load_np_array(file_name):
  try:
    with open(file_name, 'rb') as infile:
      return pickle.load(infile)
  except:
    loaded_np = np.load(file_name)
    if type(loaded_np) == np.lib.npyio.NpzFile:
      return loaded_np['arr_0']
    return loaded_np

def save_np_array(array, filename):
  with open(filename, 'wb') as outfile:
    pickle.dump(array, outfile, pickle.HIGHEST_PROTOCOL)

def mkdir(dir):
  if not os.path.exists(dir):
    os.mkdir(dir)
  return

def myresize(im, target_imshape, channel_in_first_dimension=False, mode='INTER_LINEAR'):
  '''
  :param im:
  :param target_imshape:
  :param channel_in_first_dimension:
  :param mode: Same as cv2.resize, default is INTER_LINEAR
      INTER_NEAREST - a nearest-neighbor interpolation
      INTER_LINEAR - a bilinear interpolation (used by default)
      INTER_AREA - resampling using pixel area relation.
      INTER_CUBIC - a bicubic interpolation over 4x4 pixel neighborhood
      INTER_LANCZOS4 - a Lanczos interpolation over 8x8 pixel neighborhood
  :return:
  '''
  modes = {'INTER_NEAREST': cv2.INTER_NEAREST,
           'INTER_LINEAR': cv2.INTER_LINEAR,
           'INTER_AREA': cv2.INTER_AREA,
           'INTER_CUBIC': cv2.INTER_CUBIC,
           'INTER_LANCZOS4': cv2.INTER_LANCZOS4
           }
  #assumes that the channel is the LAST dimension
  if channel_in_first_dimension:
    im = np.transpose(im, [1,2,0])
  nchannels = None
  if len(target_imshape) == 3:
    if channel_in_first_dimension:
      target_imshape = target_imshape[1:3]
      nchannels = im.shape[0]
    else:
      target_imshape = target_imshape[0:2]
      nchannels = im.shape[2]
  elif len(im.shape) == 3:
    nchannels = im.shape[2]
  im = cv2.resize(im, (target_imshape[1],target_imshape[0]), interpolation=modes[mode])
  #cv2.resize removes the the channel dimmension if ther is only 1 channel, so we need to restore it.
  if not nchannels is None and nchannels == 1:
    im = np.expand_dims(im, 3)
  if channel_in_first_dimension:
    im = np.transpose(im, [2,0,1])
  return im

class Homography:
  def __init__(self, dst_shape, clicks):
    #cv2 works opposite as matplotlib, as coordinates are specified (x,y) instead of y,x
    #dst_shape must be reverted to match cv2, and we pass
    self._dst_shape = (dst_shape[1], dst_shape[0])
    if type(clicks) is np.ndarray:
      self.clicks = clicks
    else:
      try:
        self.clicks = load_np_array(clicks)
      except:
        self.clicks = np.genfromtxt(clicks, delimiter=',')

    # Four corners in source image
    # x,y
    self.pts_src = np.array(self.clicks)

    # Four corners in destination image.
    self.pts_dst = np.array([[0, 0], [self._dst_shape[0], 0], [self._dst_shape[0], self._dst_shape[1]], [0, self._dst_shape[1]]])

    # Calculate Homography
    self.h, status = cv2.findHomography(self.pts_src, self.pts_dst)

    return

  def adapt_clicks(self, clicks):
    for i in range(len(clicks)):
      clicks[i] = (clicks[i][1], clicks[i][0])
    return clicks

  def save_clicks(self, fname):
    np.savetxt(fname, self.clicks, delimiter=',', header='x,y')

  def __call__(self, im):
    #im = np.transpose(im, [1, 0, 2])
    #flags = cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS
    if im.shape[0] == 3:
      im_warped = np.transpose(cv2.warpPerspective(np.transpose(im, [1,2,0]), self.h, self._dst_shape), [2,0,1])
      return im_warped
    else:
      im = im[0]
      im = cv2.warpPerspective(im, self.h, self._dst_shape)
      return np.expand_dims(im,0)

  def inverse_homography(self, im):
    non_inverse_h = self.h
    self.h = np.linalg.inv(self.h)
    im_h = self.__call__(im)
    self.h = non_inverse_h
    return im_h

  def get_homography_matrix(self):
    return self.h

def get_json_params(json_file):
  try:
    with open(json_file) as json_data:
      params = json.loads(json_data)
  except:
    with open(json_file, 'r') as json_data:
      params = json.loads(json_data.read())
  return params

def tile_LF(LF):
  #tiles the LF at full resolution
  tiled_LF = np.zeros((LF.shape[0], LF.shape[1]*LF.shape[3], LF.shape[2]*LF.shape[4]))
  for i in range(LF.shape[3]):
    for j in range(LF.shape[4]):
        tiled_LF[:,LF.shape[1]*i:LF.shape[1]*(i+1),LF.shape[2]*j:LF.shape[2]*(j+1)] = np.rot90(LF[:,: ,:,i, j], k=2, axes=(1,2))
  return tiled_LF

def plot_LF(LF, save_path, clamp_negatives=True):
  if clamp_negatives:
    LF = LF * (LF > 0)
  LF = (LF - LF.min())/(LF.max() - LF.min())
  tiled_LF = tile_LF(LF)
  imsave(save_path, np.transpose(tiled_LF,[1,2,0]))

def get_obs_points_and_scene_points(params, ny, nx, obs_samples_y, obs_samples_x):

  scene_x1 = params['x1']
  scene_x2 = params['x2']
  scene_y1 = params['y1']
  scene_y2 = params['y2']

  obs_x1 = params['obs_x1']
  obs_x2 = params['obs_x2']
  obs_y1 = params['obs_y1']
  obs_y2 = params['obs_y2']

  obs_points_y, obs_points_x = np.meshgrid(
    np.arange(obs_y2, obs_y1 - 0.0001, -(obs_y2 - obs_y1 + .0) / (obs_samples_y - 1)), \
    np.arange(obs_x1, obs_x2 + 0.001, (obs_x2 - obs_x1 + .0) / (obs_samples_x - 1)))
  # we need to transpose them, so when rasterized they are sorted by the outter dimmension
  obs_points_x = np.transpose(obs_points_x)
  obs_points_y = np.transpose(obs_points_y)
  obs_points = np.array([zip(x, y) for x, y in zip(obs_points_y, obs_points_x)])

  scene_points_y, scene_points_x = np.meshgrid(
    np.arange(scene_y1, scene_y2 + 0.001, (scene_y2 - scene_y1 + .0) / (ny - 1)), \
    np.arange(scene_x1, scene_x2 + 0.001, (scene_x2 - scene_x1 + .0) / (nx - 1)))
  # we need to transpose them, so when rasterized they are sorted by the outter dimmension
  scene_points_x = np.transpose(scene_points_x)
  scene_points_y = np.transpose(scene_points_y)
  scene_points = np.array([zip(x, y) for x, y in zip(scene_points_y, scene_points_x)])

  return obs_points, scene_points
