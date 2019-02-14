import pickle
import numpy as np
import tensorflow as tf
import PIL.Image
import random
from runway import RunwayModel


pgan = RunwayModel()


@pgan.setup
def setup(alpha=0.5):
    global Gs
    tf.InteractiveSession()
    batch_size = 8
    model = 'network-final.pkl'
    print("open model %s" % model)
    with open(model, 'rb') as file:
        G, D, Gs = pickle.load(file)
    return Gs


@pgan.command('convert', inputs={'z': 'vector'}, outputs={'output': 'image'})
def convert(Gs, inp):
    latents = np.array(inp['z']).reshape((1, 512))  # np.random.RandomState(1000).randn(1, *Gs.input_shapes[0][1:])
    labels = np.zeros([latents.shape[0]] + Gs.input_shapes[1][1:])
    images = Gs.run(latents, labels)
    images = np.clip(np.rint((images + 1.0) / 2.0 * 255.0), 0.0, 255.0).astype(np.uint8) # [-1,1] => [0,255]
    images = images.transpose(0, 2, 3, 1) # NCHW => NHWC
    output = np.clip(images[0], 0, 255).astype(np.uint8)
    return dict(output=output)


if __name__ == '__main__':
    pgan.run()
