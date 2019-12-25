import numpy as np
from models import Net
import skimage.io as skio

net = Net()
net.build((None, 784))
net.summary()

net.load_weights('weights/fcmodel.9.h5')

img = np.array(skio.imread('test.png')) / 255
img = img.reshape((1, -1))

print(np.argsort(net(img)))
