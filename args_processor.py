import argparse

def get_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument("--amat", type=str, required=True, help="Transfer matrix in the primal domain. Assumed to contain nearfield effects if not --compute_propagation set")
  parser.add_argument("--compute_propagation", action="store_true", help="Whether to compute the propagation effects or using amat as it is. If set, amat is binarized into the visibility mat.")
  parser.add_argument("--params_file", type=str, help="Geometrical parameters of the scenario under test")
  parser.add_argument("--observations", type=str, required=True, help="Observation (with background if background file provided)")

  parser.add_argument("--ifft_inverse", action="store_true", help="If set, compute full ifft.")

  parser.add_argument("--background", type=str, required=True, help="Background")
  parser.add_argument("--homography", type=str, required=True, help="Points of the rectangular observation plane")
  parser.add_argument("--ignore_region", type=str, required=False, help="Region to ignore")

  parser.add_argument("--result_subsample", type=int, default=30, help="The result lightfield will be nx*ny*obs_samples_x/result_subsample*obs_samples_y/result_subsample")

  parser.add_argument("--K_components", type=int, default=10000)
  parser.add_argument("--beta", type=float, default='1e-6')

  #relative depth parameters
  parser.add_argument("--a0", type=str, default="0")
  parser.add_argument("--a1", type=str, default="0.5")

  parser.add_argument("--out_dir", type=str, default="output")
  parser.add_argument("--cache_dir", type=str, default="cache", help="To store computations.")


  return parser
