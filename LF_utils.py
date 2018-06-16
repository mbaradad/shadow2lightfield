from utils import *

def get_LF_filter(LF_spectrum_shape, LF_shape, filter_args, params, return_fs_by_a=False):
  LF_filter = np.ones(LF_spectrum_shape, dtype=bool)
  fs_by_a = None
  if 'lowpass' in filter_args['filter']:
    LF_filter = LF_filter * get_lowpass_filter_LF_spectrum(LF_spectrum_shape, LF_shape, filter_args['nfreq'])
  if 'slope' in filter_args['filter']:
    slope_filter = None
    if return_fs_by_a:
      fs_by_a = dict()
    for i in range(len(filter_args['a0'])):
      actual_slope_filter, actual_fs_by_a = get_filter_LF_slope(LF_spectrum_shape, params, a0=filter_args['a0'][i], a1=filter_args['a1'][i])
      if return_fs_by_a:
        for a in actual_fs_by_a.keys():
          if not a in fs_by_a.keys():
            fs_by_a[a] = actual_fs_by_a[a]
          else:
            fs_by_a[a].extend(actual_fs_by_a[a])
      if slope_filter is None:
        slope_filter = actual_slope_filter
      else:
        slope_filter = np.logical_or(slope_filter, actual_slope_filter)
    LF_filter = LF_filter * slope_filter
  if return_fs_by_a:
    return LF_filter, fs_by_a
  else:
    return LF_filter

def get_filter_LF_slope(LF_shape, params, a0=-1, a1=1, slope_discretization=1e-2, dilate=False):
  mask = np.zeros(LF_shape[1:])
  fy, fx, fv, fu = get_frequencies(LF_shape[1:], params)
  fs_by_a = dict()
  for i in range(LF_shape[1]):
    for j in range(LF_shape[2]):
      actual_fy = fy[i]
      actual_fx = fx[j]
      for a in np.arange(a0, a1, slope_discretization):
        actual_fv = -actual_fy * a
        #the minus was found experimentally. without the minus, the results are incorrect (different slope behabior for x and y).
        #probably, compensates an error somewhere else
        actual_fu = -actual_fx * a
        if actual_fu > fu.max() or actual_fu < fu.min() or actual_fv > fv.max() or actual_fv < fv.min():
          continue
        k = np.argmin(abs(actual_fv - fv))
        l = np.argmin(abs(actual_fu - fu))
        mask[i, j, k, l] = 1
        if a in fs_by_a.keys():
          fs_by_a[a].append((i,j,k,l))
        else:
          fs_by_a[a] = [(i,j,k,l)]
  if dilate:
    for i in range(LF_shape[3]):
      for j in range(LF_shape[4]):
        from scipy import ndimage
        mask[:, :, i, j] = ndimage.binary_dilation(mask[:, :, i, j], iterations=10)
  return np.expand_dims(mask, 0), fs_by_a


def get_frequencies(LF_shape, params):
  fy = np.mgrid[0:1:complex(LF_shape[0] / 2 + 1)].tolist()
  if LF_shape[0] % 2 == 0:
    fy.extend([-1 * i for i in fy[-2:0:-1]])
  else:
    fy.extend([-1 * i for i in fy[:0:-1]])
  fx = np.mgrid[0:1:complex(LF_shape[1] / 2 + 1)].tolist()
  if LF_shape[1] % 2 == 0:
    fx.extend([-1 * i for i in fx[-2:0:-1]])
  else:
    fx.extend([-1 * i for i in fx[:0:-1]])
  fv = np.mgrid[0:1:complex(LF_shape[2] / 2 + 1)].tolist()
  if LF_shape[2] % 2 == 0:
    fv.extend([-1 * i for i in fv[-2:0:-1]])
  else:
    fv.extend([-1 * i for i in fv[:0:-1]])
  fu = np.mgrid[0:1:complex(LF_shape[3] / 2 + 1)].tolist()
  if LF_shape[3] % 2 == 0:
    fu.extend([-1 * i for i in fu[-2:0:-1]])
  else:
    fu.extend([-1 * i for i in fu[:0:-1]])
  fx = np.asarray(fx, dtype='float32')
  fy = np.asarray(fy, dtype='float32')
  fu = np.asarray(fu, dtype='float32')
  fv = np.asarray(fv, dtype='float32')
  x_spacing = (params['x2'] - params['x1'] + .0) / len(fx)
  y_spacing = (params['y2'] - params['y1'] + .0) / len(fy)
  u_spacing = (params['obs_x2'] - params['obs_x1'] + .0) / len(fu)
  v_spacing = (params['obs_y2'] - params['obs_y1'] + .0) / len(fv)
  fx = fx * 1 / x_spacing
  fy = fy * 1 / y_spacing
  fu = fu * 1 / u_spacing
  fv = fv * 1 / v_spacing
  return fy, fx, fv, fu

def get_lowpass_filter_LF_spectrum(LF_spectrum_shape, LF_shape, nfreq, separable=False):
  '''
  The lowpass returns the equivalent lowpass zeropaded spatial (in the primal domain) and the zero paded frequencies
  in the fourier domain. This is to take into acount that LF are zero outside the scene space, but are similar to
  mirrored in the angular space.
  '''

  # remove channel if present
  if len(LF_spectrum_shape) == 5:
    LF_spectrum_shape = LF_spectrum_shape[-4:]
  if len(LF_shape) == 5:
    LF_shape = LF_shape[-4:]
  a = np.array(range(LF_spectrum_shape[0]))
  b = np.array(range(LF_spectrum_shape[1]))
  c = np.array(range(LF_spectrum_shape[2]))
  d = np.array(range(LF_spectrum_shape[3]))
  # Take into account the zero pad of the LF, by doubling the fs in space, as the spectrum is subsampled i.e. zeropad in the
  # primal domain
  nfreq = [i for i in nfreq]
  nfreq[0] = nfreq[0] * LF_spectrum_shape[0] / LF_shape[0]
  nfreq[1] = nfreq[1] * LF_spectrum_shape[1] / LF_shape[1]

  as_list = list()
  bs_list = list()
  cs_list = list()
  ds_list = list()

  as_list.extend(a[:nfreq[0]].tolist())
  if nfreq[0] > 1:
    as_list.extend((-a[1:nfreq[0]]).tolist())
  bs_list.extend(b[:nfreq[1]].tolist())
  if nfreq[1] > 1:
    bs_list.extend((-b[1:nfreq[1]]).tolist())
  cs_list.extend(c[:nfreq[2]].tolist())
  if nfreq[2] > 1:
    cs_list.extend((-c[1:nfreq[2]]).tolist())
  ds_list.extend(d[:nfreq[3]].tolist())
  if nfreq[3] > 1:
    ds_list.extend((-d[1:nfreq[3]]).tolist())
  a = np.zeros([LF_spectrum_shape[0]], dtype=bool)
  b = np.zeros([LF_spectrum_shape[1]], dtype=bool)
  c = np.zeros([LF_spectrum_shape[2]], dtype=bool)
  d = np.zeros([LF_spectrum_shape[3]], dtype=bool)
  # if padding is not none, only keep 1 of each padding samples
  a[np.asarray(as_list, dtype='int32')] = 1
  b[np.asarray(bs_list, dtype='int32')] = 1
  c[np.asarray(cs_list, dtype='int32')] = 1
  d[np.asarray(ds_list, dtype='int32')] = 1

  if separable:
    return a, b, c, d
  LF_spectrum_filter = np.expand_dims(np.expand_dims(np.expand_dims(a, 1), 2), 3) * \
                       np.expand_dims(np.expand_dims(np.expand_dims(b, 0), 3), 4) * \
                       np.expand_dims(np.expand_dims(np.expand_dims(c, 0), 1), 4) * \
                       np.expand_dims(np.expand_dims(np.expand_dims(d, 0), 1), 2)
  return LF_spectrum_filter


# returns deltas grouped to avoid weird artifacts (symmetric around the constant) if zero_pad_in_space
def get_LF_spectrum_delta_space_frequencies(LF_spectrum_shape, zero_pad_in_space):
  if len(LF_spectrum_shape) == 5:
    LF_spectrum_shape = LF_spectrum_shape[1:]
  fy, fx, fu, fv = (range(LF_spectrum_shape[0]), range(LF_spectrum_shape[1]),
                    range(LF_spectrum_shape[2]), range(LF_spectrum_shape[3]))

  spectrum_indices = list()
  if zero_pad_in_space:
    # add pairs symetrically in space:
    for i in range(0, max(LF_spectrum_shape[0] / 2, 2), 2):
      for j in range(0, max(LF_spectrum_shape[1] / 2, 2), 2):
        # add the 4 frequencies of the zeropaded delta, for the 4 quadrants
        # 1st
        actual_index = list()
        actual_index.append((fy[i], fx[j]))
        if LF_spectrum_shape[0] > 1:
          actual_index.append((fy[i + 1], fx[j]))
        if LF_spectrum_shape[1] > 1:
          actual_index.append((fy[i], fx[j + 1]))
        if LF_spectrum_shape[0] > 1 and LF_spectrum_shape[1] > 1:
          actual_index.append((fy[i + 1], fx[j + 1]))
        spectrum_indices.append(actual_index)

        actual_index = list()
        actual_index.append((fy[-i], fx[j]))
        if LF_spectrum_shape[0] > 1:
          actual_index.append((fy[-(i + 1)], fx[j]))
        if LF_spectrum_shape[1] > 1:
          actual_index.append((fy[-i], fx[(j + 1)]))
        if LF_spectrum_shape[0] > 1 and LF_spectrum_shape[1] > 1:
          actual_index.append((fy[-(i + 1)], fx[(j + 1)]))
        spectrum_indices.append(actual_index)

        actual_index = list()
        actual_index.append((fy[i], fx[-(j)]))
        if LF_spectrum_shape[0] > 1:
          actual_index.append((fy[i + 1], fx[-(j)]))
        if LF_spectrum_shape[1] > 1:
          actual_index.append((fy[i], fx[-(j + 1)]))
        if LF_spectrum_shape[0] > 1 and LF_spectrum_shape[1] > 1:
          actual_index.append((fy[i + 1], fx[-(j + 1)]))
        spectrum_indices.append(actual_index)

        actual_index = list()
        actual_index.append((fy[-(i)], fx[-(j)]))
        if LF_spectrum_shape[0] > 1:
          actual_index.append((fy[-(i + 1)], fx[-(j)]))
        if LF_spectrum_shape[1] > 1:
          actual_index.append((fy[-(i)], fx[-(j + 1)]))
        if LF_spectrum_shape[0] > 1 and LF_spectrum_shape[1] > 1:
          actual_index.append((fy[-(i + 1)], fx[-(j + 1)]))
        spectrum_indices.append(actual_index)
    return spectrum_indices
  else:
    return [[(i, j)] for i in range(LF_spectrum_shape[0]) for j in range(LF_spectrum_shape[1])]

def pad_LF(LF, args, return_shape_only=False):
  LF_shape = LF.shape[1:]
  if args['zero_pad_space']:
    padding_1 = int(LF_shape[0] * 0.5)
    padding_2 = int(LF_shape[1] * 0.5)
  else:
    padding_1 = padding_2 = 0
  if args['angular_mirror_padding']:
    padding_3 = int(LF_shape[2] * args['angular_mirror_padding'][0])
    padding_4 = int(LF_shape[3] * args['angular_mirror_padding'][1])
  else:
    padding_3 = padding_4 = 0
  if return_shape_only:
    return (1, LF_shape[0] + padding_1 * 2, LF_shape[1] + padding_2 * 2, LF_shape[2] + padding_3 * 2,
            LF_shape[3] + padding_4 * 2)
  LF = np.pad(LF, [[0, 0], [padding_1, padding_1], [padding_2, padding_2], [0, 0], [0, 0]], mode='constant',
              constant_values=0)
  LF = np.pad(LF, [[0, 0], [0, 0], [0, 0], [padding_3, padding_3], [padding_4, padding_4]], mode='reflect')
  return LF, (padding_1, padding_2, padding_3, padding_4)

def unpad_LF(LF, paddings):
  if paddings[0] > 0:
    LF = LF[:, paddings[0]:-paddings[0], :, :, :]
  if paddings[1] > 0:
    LF = LF[:, :, paddings[1]:-paddings[1], :, :]
  if paddings[2] > 0:
    LF = LF[:, :, :, paddings[2]:-paddings[2], :]
  if paddings[3] > 0:
    LF = LF[:, :, :, :, paddings[3]:-paddings[3]]
  return LF

def reshape_LF(LF, target_LF_shape, mode='linear'):
  ny, nx, obs_ny, obs_nx = target_LF_shape
  if len(LF.shape) == 4:
    LF = np.expand_dims(LF, 0)
  if LF.shape[1] != ny or LF.shape[2] != nx:
    LF_reshaped = np.zeros([LF.shape[0], ny, nx, LF.shape[3], LF.shape[4]])
    for i in range(LF.shape[3]):
      print 'Reshaping ' + str(i) + ' of ' + str(LF.shape[3])
      for j in range(LF.shape[4]):
        LF_reshaped[:, :, :, i, j] = myresize(LF[:, :, :, i, j], (ny, nx), channel_in_first_dimension=True)
    LF = LF_reshaped
  if LF.shape[3] != obs_ny or LF.shape[4] != obs_nx:
    LF_reshaped = np.zeros([LF.shape[0], ny, nx, obs_ny, obs_nx])
    for i in range(LF.shape[1]):
      print 'Reshaping ' + str(i) + ' of ' + str(LF.shape[1])
      for j in range(LF.shape[2]):
        if mode == 'linear':
          LF_reshaped[:, i, j, :, :] = myresize(LF[:, i, j, :, :], (obs_ny, obs_nx), channel_in_first_dimension=True,
                                              mode='INTER_LINEAR')
        if mode == 'nearest':
          LF_reshaped[:, i, j, :, :] = myresize(LF[:, i, j, :, :], (obs_ny, obs_nx), channel_in_first_dimension=True,
                                              mode='INTER_NEAREST')
    LF = LF_reshaped
  return LF