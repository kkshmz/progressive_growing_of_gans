import os
import pickle
import numpy as np
import tensorflow as tf
import PIL.Image
import random

models = ["karras2018iclr-lsun-airplane-256x256.pkl","karras2018iclr-lsun-bedroom-256x256.pkl","karras2018iclr-lsun-bicycle-256x256.pkl",
	"karras2018iclr-lsun-bird-256x256.pkl","karras2018iclr-lsun-boat-256x256.pkl","karras2018iclr-lsun-bottle-256x256.pkl",
	"karras2018iclr-lsun-bridge-256x256.pkl","karras2018iclr-lsun-bus-256x256.pkl","karras2018iclr-lsun-car-256x256.pkl",
	"karras2018iclr-lsun-cat-256x256.pkl","karras2018iclr-lsun-chair-256x256.pkl","karras2018iclr-lsun-churchoutdoor-256x256.pkl",
	"karras2018iclr-lsun-classroom-256x256.pkl","karras2018iclr-lsun-conferenceroom-256x256.pkl","karras2018iclr-lsun-cow-256x256.pkl",
	"karras2018iclr-lsun-dining_room-256x256.pkl","karras2018iclr-lsun-diningtable-256x256.pkl","karras2018iclr-lsun-dog-256x256.pkl",
	"karras2018iclr-lsun-horse-256x256.pkl","karras2018iclr-lsun-kitchen-256x256.pkl","karras2018iclr-lsun-livingroom-256x256.pkl",
	"karras2018iclr-lsun-motorbike-256x256.pkl","karras2018iclr-lsun-person-256x256.pkl","karras2018iclr-lsun-pottedplant-256x256.pkl",
	"karras2018iclr-lsun-restaurant-256x256.pkl","karras2018iclr-lsun-sheep-256x256.pkl","karras2018iclr-lsun-sofa-256x256.pkl",
	"karras2018iclr-lsun-tower-256x256.pkl","karras2018iclr-lsun-train-256x256.pkl","karras2018iclr-lsun-tvmonitor-256x256.pkl"]


models = ["network-snapshot-012033.pkl", "network-snapshot-013044.pkl", 
		"network-snapshot-008604.pkl", "network-snapshot-011022.pkl", 
		"network-snapshot-006012.pkl", "network-snapshot-001467.pkl",
		"network-snapshot-010011.pkl"]

styles = ["abstract-art", "abstract-expressionism", "academicism", 
		"art-deco", "art-informel", "art-nouveau", 
		"baroque", "conceptual-art", "contemporary-realism", 
		"cubism", "early-renaissance", "expressionism", 
		"fauvism", "high-renaissance", "impressionism", 
		"magic-realism", "minimalism", "naive-art-primitivism", 
		"neo-expressionism", "neoclassicism", "northern-renaissance", 
		"op-art", "pointillism", "pop-art", 
		"post-impressionism", "realism", "rococo", 
		"romanticism", "socialist-realism", "surrealism", 
		"symbolism", "ukiyo-e"]


def get_latents(num_frames, num_endpoints, dim):
	r_seed = int(1000 * random.random())
	endpoints = np.random.RandomState(r_seed).randn(num_endpoints, dim)
	L = np.zeros((num_frames, dim))
	n = int(num_frames / num_endpoints)
	for e in range(num_endpoints):
		e1, e2 = e, (e+1)%num_endpoints
		for t in range(n):
			frame = e * n + t
			#r = float(t) / n  # linear
			r = 0.5 - 0.5 * np.cos(np.pi*t/(n-1)) 	# easing
			L[frame, :] = (1.0-r) * endpoints[e1,:] + r * endpoints[e2,:]
	return L


def run_model(idx_model, style, num_frames, num_endpoints):
	batch_size = 8
	model = 'models/%s' % models[idx_model]
	class_name = (models[idx_model]).split('-')[2]
	out_dir, out_movie_dir = 'frames', 'out'
	idx_style = styles.index(style)

	print("open model %s" % model)
	with open(model, 'rb') as file:
	    G, D, Gs = pickle.load(file)
	
	latents = get_latents(num_frames, num_endpoints, *Gs.input_shapes[0][1:])
	labels = np.zeros([latents.shape[0]] + Gs.input_shapes[1][1:])
	labels[:,idx_style] = 1.0

	num_batches = int(np.ceil(num_frames/batch_size))
	images = []
	for b in range(num_batches):
		print("batch %d / %d"%(b+1, num_batches))
		new_images = Gs.run(latents[b*batch_size:min((b+1)*batch_size, num_frames-1), :], labels[b*batch_size:min((b+1)*batch_size, num_frames-1),:])
		new_images = np.clip(np.rint((new_images + 1.0) / 2.0 * 255.0), 0.0, 255.0).astype(np.uint8) # [-1,1] => [0,255]
		new_images = new_images.transpose(0, 2, 3, 1) # NCHW => NHWC
		for img in new_images:
			images.append(img)

	for idx in range(len(images)):
		print(' - save frame %d'%idx)
		PIL.Image.fromarray(images[idx], 'RGB').save('%s/img%05d.png' % (out_dir, idx))

	filename = 'm%03d_%s' % (idx_model, style)
	cmd = 'ffmpeg -i %s/img%%05d.png -c:v libx264 -pix_fmt yuv420p %s/%s.mp4' % (out_dir, out_movie_dir, filename)
	os.system(cmd)


tf.InteractiveSession()

num_endpoints = 5
num_frames = 5*40

#for idx_model in range(len(models)):
for s in styles:
	run_model(1, s, num_frames, num_endpoints)
