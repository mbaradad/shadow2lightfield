from args_processor import get_parser
import sys
from solver import Solver
from utils import *
import scipy.sparse as sparse
from tqdm import tqdm
import warnings

#encodes bad regions on misc, and transform from coordinates into masks
class IgnoreRegion():
  def __init__(self, ignore_region_file):
    self.ignore_region_file = ignore_region_file
    self.clicks = np.genfromtxt(ignore_region_file, delimiter=',').reshape([-1,4,2])

  def get_ignore_region_mask(self, obs_shape):
    mask = np.ones(obs_shape)
    obs_ny = obs_shape[0]
    obs_nx = obs_shape[1]
    for group in self.clicks:
      #get the rectangle that fits the points
      y1 = int(np.floor(min(group[0][0], group[3][0])*obs_ny))
      y2 = int(np.ceil(max(group[1][0], group[2][0])*obs_ny))
      x1 = int(np.floor(min(group[2][1], group[3][1])*obs_nx))
      x2 = int(np.ceil(max(group[0][1], group[1][1])*obs_nx))
      mask[y1:y2,x1:x2] = 0
    return mask

def get_processed_obs(args, obs_shape):
  original_obs = np.load(args['observations'])

  background = np.load(args['background']) if not args['background'] is None else None

  obs = original_obs - background

  h = Homography(obs.shape[1:], args['homography'])
  obs = h(obs)
  obs = myresize(obs, obs_shape, channel_in_first_dimension=True)

  # normalize
  obs = (obs - obs.min())/(obs.max() - obs.min())

  ignore_region = IgnoreRegion(args['ignore_region'])
  ignore_region_mask = ignore_region.get_ignore_region_mask(obs_shape)

  return obs, ignore_region_mask, original_obs

def get_near_field_projection(obs_point, scene_points, params):
  zdist = params['z']
  # dist min is zdist, the distance between both planes
  dirs = np.array(obs_point) - scene_points
  # the z component is equal to -zdist, as both planes are parallel
  dirs = np.hstack([dirs, -zdist * np.ones((scene_points.shape[0], 1))])
  # normalized but the minimum distance
  dist = np.sqrt(np.sum(dirs * dirs, axis=1))
  normalized_dirs = np.transpose(np.transpose(dirs) / dist)

  normalized_dist = dist / zdist
  projection = 1 / normalized_dist ** 2

  projection = projection * abs(normalized_dirs[:, 2])**2
  return projection

def load_a_mat(args, params):
  original_A = [load_np_array(args['amat'])]
  if not args['compute_propagation']:
    return original_A
  # We binarize A_mat to ignore the precomputed nearfield effects and recompute them again using the scene geometrical
  # params

  params['nx'] = 25
  params['ny'] = 35
  params['obs_nx'] = 150
  params['obs_ny'] = 225

  obs_points, scene_points = get_obs_points_and_scene_points(params)

  rasterized_scene_points = scene_points.reshape([-1, 2])
  rasterized_obs_points = obs_points.reshape([-1, 2])
  A = [sparse.lil_matrix(original_A[0].shape, dtype='float32')]
  print 'Adding nearfield effects to A matrix'
  for i in tqdm(range(len(rasterized_obs_points))):
    index = range(i, original_A[0].shape[1], len(rasterized_obs_points))
    nf_projection = get_near_field_projection(rasterized_obs_points[i], rasterized_scene_points, params)
    #only set the ones that are nonzero in the loaded mat
    nonzero = original_A[0][i, index].nonzero()[1]
    mask = np.zeros(nf_projection.shape)
    mask[nonzero] = 1
    nf_projection_masked = nf_projection*mask
    A[0][i, index] = nf_projection_masked
    #it should be the same as the precomputed amat
  return A

def main(args):
  print 'Results will be saved in: ' + args['out_dir']
  mkdir(args['out_dir'])

  params = get_json_params(args['params_file'])
  A = load_a_mat(args, params)

  ny = 25
  nx = 35
  obs_ny = 150
  obs_nx = 225
  assert A[0].shape[0] == obs_ny*obs_nx
  assert A[0].shape[1] == ny*nx*obs_ny*obs_nx

  obs, ignore_region_mask, original_obs = get_processed_obs(args, (obs_ny, obs_nx))
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    imsave(args['out_dir'] + '/full_obs.png', np.transpose(obs,(1,2,0)))
    # known bad observation points are masked
    imsave(args['out_dir'] + '/masked_obs.png', np.transpose(obs*ignore_region_mask, (1, 2, 0)))
    imsave(args['out_dir'] + '/original_obs.png', np.transpose(original_obs, (1, 2, 0)))

  solver = Solver(A, args, [ny,nx,obs_ny,obs_nx], ignore_region_mask, params, ifft_inverse=args['ifft_inverse'])

  LF = []
  spectrum = []
  # solve for each color channel independently
  for k in range(3):
    LF_ch, spectrum_ch = solver.solve_inverse(obs[k], args['beta'])
    LF.append(LF_ch)
    spectrum.append(spectrum_ch)
  LF = np.array(LF)[:,0]
  spectrum = np.array(spectrum)[:,:,0]
  print "Results computed. Saving at: " + args['out_dir']
  if not os.path.exists(args['out_dir']):
    os.makedirs(args['out_dir'])

  plot_LF(LF, args['out_dir'] + '/LF' + ('_ifft' if args['ifft_inverse'] else '') + '.png')
  print "Results saved!"

if __name__ == "__main__":
  parser = get_parser()
  command_args = []

  '''
  Testing params
  scenario = 'head'
  command_args.extend(['--amat','data/plants_amat.npz'])
  command_args.extend(['--params_file', 'data/{}/params.json'.format(scenario)])
  command_args.extend(['--observations', 'data/{}/obs.npz'.format(scenario)])
  command_args.extend(['--homography', 'data/{}/homography.npz'.format(scenario)])
  command_args.extend(['--background','data/{}/background.npz'.format(scenario)])
  command_args.extend(['--ignore_region','data/{}/ignore_region'.format(scenario)])
  command_args.extend(['--out_dir','output/{}'.format(scenario)])
  '''

  command_args.extend(sys.argv[1:])
  args = vars(parser.parse_args(command_args))
  main(args)