import os.path
import numpy as np
from tqdm import *
import pickle as cPickle

def get_score_old(x, bit_width):
	br_x = get_bit_rep(int(x), bit_width)
	N = int(len(br_x) / 2);
	score = 0
	for i in range(0, N):
		l = 2*i-1
		h = 2*i+1
		if (l < 0): l = 0
		x_part = br_x[l:h+1]
		if (len(x_part) < 3): x_part = np.insert(x_part,0,0)
		if ((x_part.sum() != 0) and (x_part.sum() != 3)) : score += 1
		#print h, l, x_part, x_part.sum(), score
	return score


def get_new_state(err, x, x_max, x_min):
	val = int(np.random.randint(-int(err), int(err))) + x
	if val > x_max: return x_max
	elif val < x_min: return x_min
	else: return val

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def get_energy(x, s, err, bit_width):
	skip_val = get_skip_val(s, bit_width)
	# dist = gaussian(np.abs(x - s), 0, err/2)
	# return skip_val*dist 
	return skip_val 

def get_sa_best_val(x, bit_width, err):
	if x == 0:
		return x

	# Simulate Annealing Approach
	Tmax = 2500
	Tmin = 2.5
	steps = int(np.ceil(0.1*err))
	x_max = int(np.math.pow(2, bit_width-1)-1) 
	x_min = int(-np.math.pow(2, bit_width-1))

	# Choose a Starting Locations
	state = get_new_state(err, x, x_max, x_min)
	energy = get_energy(x, state, err, bit_width)
	b_state = state
	b_energy = energy
	for i in range(steps):
		temp = Tmax * np.math.exp(-np.math.log(Tmax / Tmin) * i / steps)
		n_state = get_new_state(err, x, x_max, x_min)
		n_energy = get_energy(x, n_state, err, bit_width)
		delta_energy = energy - n_energy
		reject = delta_energy > 0.0 and np.math.exp(-delta_energy / temp) < np.random.random()
		if not reject:
			state = n_state
			energy = n_energy
			if energy > b_energy:
				b_state = state
				b_energy = energy
	return b_state


def get_best_val(x, bit_width, err):
	if x == 0:
		return x
	search_range = range(1,int(err))
	bi = x # Best Index
	bi_c = get_skip_val(bi, bit_width)
	x_max = int(np.math.pow(2, bit_width-1)-1) 
	x_min = int(-np.math.pow(2, bit_width-1))

	for i in search_range:
		l = x-i
		h = x+i
		if (l <= x_min): l = x_min
		if (h >= x_max): h = x_max
		l_c = get_skip_val(l, bit_width)
		h_c = get_skip_val(h, bit_width)
		if (l_c > bi_c):
			bi = l
			bi_c = l_c
		if (h_c > bi_c):
			bi = h
			bi_c = h_c
	return bi


vals8 = [2,1,1,1,0,0,0,1,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,1,0,0,0,1,1,1,2]
def perf_tbl8(x, b):
	return vals8[((x << 1) & (0x1F << 4*b)) >> 4*b]

def get_skip_val(x, bit_width):
	if x == 0:
		return int(bit_width/2)

	# inputs = range(int(bit_width/4))
	# num_cores = multiprocessing.cpu_count()
	# res = Parallel(n_jobs=num_cores)(delayed(perf_encode)(x, i) for i in inputs)
	# return np.array(res).sum()

	val = 0
	for i in range(int(bit_width/4)):
		val += perf_tbl8(x, i)
	return val


	# vals = np.array([2,1,1,1,0,0,0,1,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,1,0,0,0,1,1,1,2])
	# b_r = np.arange(int(32/4))
	# v_r = (x & (0x1F << 4*b_r)) >> 4*b_r	
	# return vals[v_r].sum()

# def get_skip_val(x, bit_width):
# 	if x == 0:
# 		return int(bit_width/2)
# 	br_x = get_bit_rep(int(x), bit_width)
# 	br_x = np.insert(br_x, 0, 0)
# 	N = int(len(br_x)/2)
# 	shape = (N, 3)
# 	b = np.lib.stride_tricks.as_strided(br_x, shape=shape, strides=(2,1)).sum(axis=1)
# 	zeros = len(b[b == 0]) + len(b[b == 3])
# 	return zeros

def create_skip_val(bit_width):
	x_max = int(np.math.pow(2, bit_width-1)-1) 
	x_min = int(-np.math.pow(2, bit_width-1))
	x_dict = np.zeros(int(np.math.pow(2, bit_width)), dtype=np.uint8)
	pickle_filename = "lut/" + str(bit_width) + "_skip_val_lut.p"

	if (os.path.isfile(pickle_filename)):
		x_dict = cPickle.load(open(pickle_filename, 'rb'))
	else:
		for i in tqdm(range(x_min, x_max+1)):
			set_dict(x_dict, i, get_skip_val(i, bit_width))
		cPickle.dump(x_dict, open(pickle_filename, 'wb'), protocol=4)
	return x_dict

def get_score(x, bit_width, pp, la):
	br_x = get_bit_rep(int(x), bit_width)
	br_x = np.insert(br_x, 0, 0)
	N = int(len(br_x)/2)
	shape = (N, 3)
	b = np.lib.stride_tricks.as_strided(br_x, shape=shape, strides=(2,1)).sum(axis=1)

	b[b == 3] = 0

	sco = 0
	i = 0
	# Handle the Score Calculation for Different Partial Products and LookAheads
	while i < len(b):
		# Where ever we are up to we need to do these partial products
		sco += 1

		# Move Past the PP
		i = i + pp
		if i >= len(b):
			break;

		skip_ready = True
		sk = 0
		# Next we handle the lookahead
		for j in range(la):
			if i+j >= len(b):
				break
			if skip_ready and (b[i+j] == 0):
				sk = sk + 1
			else:
				skip_ready = False
		i = i + sk

	return sco 

def get_bit_rep(val, bit_width):
	val_bin = np.binary_repr(val, width=bit_width)
	val_arr = np.array(list(val_bin), dtype=np.uint8)
	return val_arr[::-1]

def set_dict(x_dict, index, score):
	index = int(index)
	# Figure out the BitWidth
	bit_width = int(np.log2(len(x_dict)))
	x_max = int(np.math.pow(2, bit_width))
	if index < 0: index += x_max
	x_dict[index] = int(score)

def get_dict(x_dict, index):
	index = int(index)
	# Figure out the BitWidth
	bit_width = int(np.log2(len(x_dict)))
	x_max = int(np.math.pow(2, bit_width))
	if index < 0: index += x_max
	#print(bit_width, index, x_dict[index], get_bit_rep(index, bit_width))
	return x_dict[index]

def get_best_approx(x, x_dict, err, abs_err):
	if x == 0:
		return x
	search_range = int(err)
	if not abs_err: search_range = int(np.abs(x)*err)
	bit_width = int(np.log2(len(x_dict)))
	bi = x # Best Index
	x_max = int(np.math.pow(2, bit_width-1)-1) 
	x_min = int(-np.math.pow(2, bit_width-1))
	for i in range(search_range):
		l = x-i
		h = x+i
		if (l <= x_min): l = x_min
		if (h >= x_max): h = x_max
		if (get_dict(x_dict,l) < get_dict(x_dict, bi)):
			bi = l
		if (get_dict(x_dict, h) < get_dict(x_dict, bi)):
			bi = h
	return bi

def create_dict(bit_width, pp, la):
	x_max = int(np.math.pow(2, bit_width-1)-1) 
	x_min = int(-np.math.pow(2, bit_width-1))
	x_dict = np.zeros(int(np.math.pow(2, bit_width)), dtype=np.uint8)
	pickle_filename = "lut/" + str(bit_width) + "_" + str(pp) + "_" + str(la) + "_lut.p"

	if (os.path.isfile(pickle_filename)):
		x_dict = cPickle.load(open(pickle_filename, 'rb'))
	else:
		for i in tqdm(range(x_min, x_max+1)):
			set_dict(x_dict, i, get_score(i, bit_width, pp, la))
		cPickle.dump(x_dict, open(pickle_filename, 'wb'), protocol=4)
	return x_dict

def perform_bit_sweep(x_dict, bit_width, pp, la, err, abs_err):
	x_max = int(np.math.pow(2, bit_width-1)-1) 
	x_min = int(-np.math.pow(2, bit_width-1))

	if abs_err: err = int(np.math.pow(2, err))
	
	error = []
	a_score = []
	p_score = []

	error_filename = "res/" + str(bit_width) + "_" + str(err) + "_" + str(abs_err) + "_err.p"
	a_score_filename = "res/" + str(bit_width) + "_" + str(err) + "_" + str(abs_err) + "_a_sco.p"
	p_score_filename = "res/" + str(bit_width) + "_" + str(err) + "_" + str(abs_err) + "_p_sco.p"

	if(os.path.isfile(error_filename) and os.path.isfile(p_score_filename) and os.path.isfile(a_score_filename)):
		error = cPickle.load(open(error_filename, 'rb'))
		a_score = cPickle.load(open(a_score_filename, 'rb'))
		p_score = cPickle.load(open(p_score_filename, 'rb'))
	else:
		for i in tqdm(range(x_min, x_max+1)):
			if i != 0:
				i_approx = get_best_approx(i, x_dict, err, abs_err)
				p_score.append(get_score(i, bit_width, pp, la))
				a_score.append(get_score(i_approx, bit_width, pp, la))
				error.append(np.abs(np.abs(i) - np.abs(i_approx))/np.abs(i))
			else:
				p_score.append(0)
				a_score.append(0)
				error.append(0.0)

		error = np.array(error)
		p_score = np.array(p_score)
		a_score = np.array(a_score)
		cPickle.dump(error, open(error_filename, 'wb'), protocol=4)
		cPickle.dump(p_score, open(p_score_filename, 'wb'), protocol=4)
		cPickle.dump(a_score, open(a_score_filename, 'wb'), protocol=4)

	p_err = error
	m_sco = a_score.mean()
	p_sco = p_score.mean()
	print('Sweep Results:')
	print('  Mean Error: %.4f'% (p_err.mean())) 
	print('  Max Error:%.4f' % (np.max(p_err)))
	print('  Min Error: %.4f' % (np.min(p_err)))
	print('  Average Cycle Time(P): %.2f' % p_sco)
	print('  Average Cycle Time(A): %.2f' % m_sco)

	return error, p_score, a_score

def get_approx_v(x, x_dict, err, abs_err):
	x_max = int(np.math.pow(2, err)) 
	if abs_err: err = int(x_max)
	f_get_best_approx = np.vectorize(get_best_approx, excluded=['x_dict', 'err', 'abs_err'])
	return f_get_best_approx(x=x, x_dict=x_dict, err=err, abs_err=abs_err)

def get_score_v(x, bit_width, pp, la):
	f_get_score = np.vectorize(get_score, excluded=['bit_width', 'pp', 'la'])
	return f_get_score(x=x, bit_width=bit_width, pp=pp, la=la)

def get_skip_val_v(x, bit_width):
	f_get_skip_val = np.vectorize(get_skip_val, excluded=['bit_width'])
	return f_get_skip_val(x=x, bit_width=bit_width)

def get_best_val_v(x, bit_width, err):
	x_max = int(np.math.pow(2, err)) 
	f_get_best_val_v = np.vectorize(get_sa_best_val, excluded=['bit_width', 'err'])
	return f_get_best_val_v(x=x, bit_width=bit_width, err=x_max)
