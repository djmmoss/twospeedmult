import numpy as np
import tensorflow as tf

def to_fixed_np(x, fl):
	res = np.array(np.round(x * int(2 ** fl)), dtype=np.int)
	return res

def to_float_np(x, fl):
	res = x / np.power(2, fl)
	return res

def calculate_fl_np(x, bit_width):
	x_max = np.ceil(np.max(np.abs(x)))
	# Always need a bit for the sign
	s = 1
	# Now we need log2 bits for the integer pars
	if (x_max == 0): il = 0
	else: il = np.log2(x_max)
	return int(bit_width - (s + il))

def to_fixed(x, fl):
	res = tf.round(tf.multiply(x,tf.cast(tf.pow(2, fl), tf.float32)))
	return res

def to_float(x, fl):
	res = tf.div(x, tf.cast(tf.pow(2, fl), tf.float32))
	return res

def tf_log2(x):
	return tf.div(tf.log(x), tf.log(2.0))

def calculate_fl(x, bit_width):
	x_max = tf.ceil(tf.reduce_max(tf.abs(x)))
	# Always need a bit for the sign
	s = 1
	# Now we need log2 bits for the integer pars
	if (x_max == 0): il = 0
	else: il = tf.cast(tf.ceil(tf_log2(tf.cast(x_max, tf.float32))), tf.int32)
	return bit_width - (s + il)

def apply_fxd(x, W, strides, bit_width=16):
	# For the 16 bits calcaulte the max fractional length
	W_fl = calculate_fl(W, bit_width)
	x_fl = calculate_fl(x, bit_width)
	fl = tf.minimum(W_fl, x_fl)

	W = tf.cast(to_float(to_fixed(W, fl), fl), tf.float32)
	x = tf.cast(to_float(to_fixed(x, fl), fl), tf.float32)

	return x, W 

def fxd_conv2d(x, W, strides, padding, do_fxd=False, bit_width=16):
	n_x, n_W = apply_fxd(x, W, bit_width)

	x = tf.cond(do_fxd, lambda: n_x, lambda: x)
	W = tf.cond(do_fxd, lambda: n_W, lambda: W)

	return tf.nn.conv2d(x, W, strides=strides, padding=padding)

def fxd_matmul(x, W, do_fxd=False, bit_width=16):
	n_x, n_W = apply_fxd(x, W, bit_width)

	x = tf.cond(do_fxd, lambda: n_x, lambda: x)
	W = tf.cond(do_fxd, lambda: n_W, lambda: W)

	return tf.matmul(x, W)
