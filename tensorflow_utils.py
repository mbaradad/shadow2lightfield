import warnings
with warnings.catch_warnings():
  warnings.simplefilter("ignore")
  import tensorflow as tf
import numpy as np

#To accelerate fft/ifft computations and allow easy interoperability between CPU and GPU, we use tensorflow.
class Tensorflow_Utils():
  def __init__(self, A_mat, LF_shape, obs_shape=None):
    self.A_mat = [A_mat]
    self.A_sparse_tensors = list()
    #arbitrary number, to overcome the limitation of constants bigger than 2GB. Could be better adjusted.
    self.n_sparse_tensors = 13
    self.sparse_tensors_length = self.A_mat[0].shape[1] / (self.n_sparse_tensors)
    Amat_type = 'complex64'
    self.sparse_placeholder_dict = dict()
    for a in range(self.n_sparse_tensors - 1):
        self.A_sparse_tensors.append(self.convert_sparse_matrix_to_sparse_tensor(self.A_mat[0][:, a * self.sparse_tensors_length:(a + 1) * self.sparse_tensors_length], Amat_type))
    self.A_sparse_tensors.append(self.convert_sparse_matrix_to_sparse_tensor(
      self.A_mat[0][:, (self.n_sparse_tensors-1) * self.sparse_tensors_length:], Amat_type))
    self.obs_shape = obs_shape
    self.LF_shape = LF_shape
    #point that minimizes the distance scene->obs_plane->camera
    self.filter = None
    self.uniform_subsampling = 10
    self.tf_config = tf.ConfigProto(allow_soft_placement=True)
    self.tf_config.gpu_options.allow_growth = True

    self.get_obs_from_LF_op = None
    self.ifft_op = None
    self.get_obs_from_LF_delta_op = None
    sess = tf.Session(config=self.tf_config)
    self.session = sess

  def get_ifft_op(self, spectrum_tensor, axis):
    if axis == 'all':
      ifft_angles = tf.spectral.ifft2d(spectrum_tensor)
      ifft_all = tf.transpose(tf.spectral.ifft2d(tf.transpose(ifft_angles, [0, 3, 4, 1, 2])), [0, 3, 4, 1, 2])
    else:
      transposed_axis = range(len(spectrum_tensor.shape))
      transposed_axis.pop(axis)
      transposed_axis.insert(0, axis)
      ifft_all = tf.spectral.ifft(tf.transpose(spectrum_tensor, transposed_axis))
      transposed_axis = range(len(spectrum_tensor.shape))
      transposed_axis.insert(axis, 0)
      transposed_axis.pop(0)
      ifft_all = (tf.transpose(ifft_all, transposed_axis))
    return ifft_all

  def get_ifft(self, LF_spectrum, axis='all'):
    if self.ifft_op is None or self.ifft_op_axis != axis:
      self.ifft_op_axis = axis
      self.ifft_spectrum_ph = tf.placeholder(dtype=tf.complex64, shape=LF_spectrum.shape)
      self.ifft_op = self.get_ifft_op(self.ifft_spectrum_ph, axis)
    with self.session.as_default():
      LF = self.session.run(self.ifft_op, feed_dict={self.ifft_spectrum_ph: LF_spectrum})
      return LF

  def get_obs_from_LF_spectrum(self, LF_spectrum):
    if self.get_obs_from_LF_op is None:
      with tf.device('/gpu:0'):
        self.LF_spectrum_ph = tf.placeholder(dtype=tf.complex64, shape=np.expand_dims(LF_spectrum, 0).shape, name='Spectrum')
        ifft_angles = tf.spectral.ifft2d(self.LF_spectrum_ph)
      with tf.device('/gpu:1'):
        ifft_all = tf.transpose(tf.spectral.ifft2d(tf.transpose(ifft_angles, [0, 3, 4, 1, 2])), [0, 3, 4, 1, 2])
        self.get_obs_from_LF_op = self.get_eval_op(tf.reshape(ifft_all, [-1,1]))
        self.get_obs_from_LF_feed_dict = self.sparse_placeholder_dict.copy()
    with self.session.as_default():
      LF_spectrum = np.expand_dims(LF_spectrum, 0)
      self.get_obs_from_LF_feed_dict[self.LF_spectrum_ph] = LF_spectrum
      obs = self.session.run(self.get_obs_from_LF_op, feed_dict=self.get_obs_from_LF_feed_dict)
      return obs

  def get_obs_from_LF_delta(self, LF_shape, LF_pos, LF_spectrum_shape=None, return_base=False, return_border=True):
    if LF_spectrum_shape is None:
      LF_spectrum_shape = LF_shape
    if len(LF_spectrum_shape) == 5:
      LF_spectrum_shape = LF_spectrum_shape[1:]
    if len(LF_shape) == 5:
      LF_shape = LF_shape[1:]
    #if anything doen't match previous execution, rebuild the op
    if self.get_obs_from_LF_delta_op is None \
        or self.return_base != return_base  \
        or np.array([self.get_obs_from_LF_delta_op_LF_spectrum_shape[i] != LF_spectrum_shape[i] for i in range(len(LF_spectrum_shape))]).any() \
        or np.array([self.get_obs_from_LF_delta_op_LF_shape[i] != LF_shape[i] for i in range(len(LF_shape))]).any():
      self.get_obs_from_LF_delta_op_LF_spectrum_shape = LF_spectrum_shape
      self.get_obs_from_LF_delta_op_LF_shape = LF_shape
      self.return_base = return_base
      self.delta_1_ph = tf.placeholder(shape=(None,LF_spectrum_shape[0]), dtype=tf.complex64)
      self.delta_2_ph = tf.placeholder(shape=(None,LF_spectrum_shape[1]), dtype=tf.complex64)
      self.delta_3_ph = tf.placeholder(shape=(None,LF_spectrum_shape[2]), dtype=tf.complex64)
      self.delta_4_ph = tf.placeholder(shape=(None,LF_spectrum_shape[3]), dtype=tf.complex64)

      #undo padding after separable ifft, grabbing the central portions for having mirror padding in the LF space

      offsets = [LF_spectrum_shape[i]/2 - LF_shape[i]/2 for i in range(4)]

      ifft_1 = tf.ifft(self.delta_1_ph)#[:,offsets[0]:offsets[0] + LF_shape[0]]
      ifft_2 = tf.ifft(self.delta_2_ph)#[:,offsets[1]:offsets[1] + LF_shape[1]]

      ifft_3 = tf.ifft(self.delta_3_ph)[:,offsets[2]:offsets[2] + LF_shape[2]]
      ifft_4 = tf.ifft(self.delta_4_ph)[:,offsets[3]:offsets[3] + LF_shape[3]]

      #maybe the product can be made more efficient, taking into account the crops that are made after, but take care!
      #its not as separable as the angular bases
      ifft_1_per_ifft_2 = tf.expand_dims(ifft_1,2)*tf.expand_dims(ifft_2,1)
      #per quadrants, check notebook FFT padding to understand how shifts work in when zeropadding bases

      odd_0 = (LF_shape[0] % 2) * (offsets[0] != 0)
      odd_1 = (LF_shape[1] % 2) * (offsets[1] != 0)
      ifft_1_per_ifft_2_0_0 = ifft_1_per_ifft_2[:,-offsets[0]:, -offsets[1]:]
      ifft_1_per_ifft_2_0_1 = ifft_1_per_ifft_2[:,-offsets[0]:, :(offsets[1] + odd_1):]
      ifft_1_per_ifft_2_1_0 = ifft_1_per_ifft_2[:,:(offsets[0] + odd_0), -offsets[1]:]
      ifft_1_per_ifft_2_1_1 = ifft_1_per_ifft_2[:,:(offsets[0] + odd_0):, :(offsets[1] + odd_1):]

      ifft_1_per_ifft_2_cropped_0 = tf.concat((ifft_1_per_ifft_2_0_0, ifft_1_per_ifft_2_0_1), axis=2)
      ifft_1_per_ifft_2_cropped_1 = tf.concat((ifft_1_per_ifft_2_1_0, ifft_1_per_ifft_2_1_1), axis=2)

      ifft_1_per_ifft_2_cropped = tf.concat((ifft_1_per_ifft_2_cropped_0, ifft_1_per_ifft_2_cropped_1), axis=1)

      ifft_all = tf.expand_dims(tf.expand_dims(ifft_1_per_ifft_2_cropped, 3), 4) * \
                 tf.expand_dims(tf.expand_dims(tf.expand_dims(ifft_3, 1), 2), 4) * \
                 tf.expand_dims(tf.expand_dims(tf.expand_dims(ifft_4, 1), 2), 3)
      ifft_all = tf.reduce_sum(ifft_all, axis=0)
      if return_base:
        self.get_obs_from_LF_delta_op = tf.reshape(ifft_all, [-1, 1])
      else:
        self.get_obs_from_LF_delta_op = self.get_eval_op(tf.reshape(ifft_all, [-1, 1]))
        if return_border:
          self.get_obs_from_LF_delta_op = [self.get_obs_from_LF_delta_op, ifft_all]

      self.get_obs_from_LF_delta_feed_dict = self.sparse_placeholder_dict.copy()
    if not type(LF_pos) == list:
      LF_pos = [LF_pos]
    deltas_1 = list()
    deltas_2 = list()
    deltas_3 = list()
    deltas_4 = list()
    for pos in LF_pos:
      delta_1 = np.zeros(LF_spectrum_shape[0]);
      delta_1[pos[0]] = 1
      delta_2 = np.zeros(LF_spectrum_shape[1]);
      delta_2[pos[1]] = 1
      delta_3 = np.zeros(LF_spectrum_shape[2]);
      delta_3[pos[2]] = 1
      delta_4 = np.zeros(LF_spectrum_shape[3]);
      delta_4[pos[3]] = 1
      deltas_1.append(delta_1);deltas_2.append(delta_2);deltas_3.append(delta_3);deltas_4.append(delta_4);
    with self.session.as_default():
      self.get_obs_from_LF_delta_feed_dict[self.delta_1_ph] = np.asarray(deltas_1)
      self.get_obs_from_LF_delta_feed_dict[self.delta_2_ph] = np.asarray(deltas_2)
      self.get_obs_from_LF_delta_feed_dict[self.delta_3_ph] = np.asarray(deltas_3)
      self.get_obs_from_LF_delta_feed_dict[self.delta_4_ph] = np.asarray(deltas_4)
    return self.session.run(self.get_obs_from_LF_delta_op, feed_dict=self.get_obs_from_LF_delta_feed_dict)

  def convert_sparse_matrix_to_sparse_tensor(self, X, Amat_type):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    coo_placeholder = tf.placeholder(shape=coo.data.shape, dtype=Amat_type)
    indices_placeholder = tf.placeholder(shape=indices.shape, dtype=tf.int64)

    self.sparse_placeholder_dict[coo_placeholder] = np.asarray(coo.data,dtype=Amat_type)
    self.sparse_placeholder_dict[indices_placeholder] = indices
    return tf.SparseTensor(indices_placeholder, coo_placeholder, coo.shape)

  def get_predicted_obs_from_LF(self, scene):
    flattened_shape = (reduce(lambda x,y: x*y, self.LF_shape), 1)
    real_LF_input = tf.placeholder(shape=flattened_shape, dtype='float32')
    predicted_obs = self.get_eval_op(real_LF_input)
    feed_dict = self.sparse_placeholder_dict.copy()
    feed_dict[real_LF_input] = scene.reshape(flattened_shape)

    with self.session.as_default():
      predicted_obs_tf = self.session.run(predicted_obs, feed_dict=feed_dict)
    return predicted_obs_tf.squeeze()

  def get_eval_op(self, input):
    A_times_x = None
    i = 0
    for mat in self.A_sparse_tensors[:-1]:
      actual_A_times_x = tf.sparse_tensor_dense_matmul(tf.cast(mat,dtype=input.dtype), input[i * self.sparse_tensors_length:(i + 1) * self.sparse_tensors_length])
      #increase precision to match np dense multiplication
      if A_times_x is None:
        A_times_x = actual_A_times_x
      else:
        A_times_x = A_times_x + actual_A_times_x
      i = i + 1
    actual_A_times_x = tf.sparse_tensor_dense_matmul(tf.cast(self.A_sparse_tensors[-1], dtype=input.dtype),
                                                     input[i * self.sparse_tensors_length:])
    A_times_x = A_times_x + actual_A_times_x
    return A_times_x

