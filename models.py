import tensorflow as tf
from tensorflow.keras.models import Model

class Net(Model):
	def __init__(self):
		super(Net, self).__init__()
		self.fc_input = tf.keras.layers.Dense(256)
		self.fc_1 = tf.keras.layers.Dense(256)
		self.fc_2 = tf.keras.layers.Dense(256)
		self.fc_output = tf.keras.layers.Dense(10)

	def call(self, x):
		x = self.fc_input(x)
		x = tf.nn.relu(x)
		x = self.fc_1(x)
		x = tf.nn.relu(x)
		x = self.fc_2(x)
		x = tf.nn.relu(x)
		x = self.fc_output(x)
		x = tf.nn.softmax(x)
		return x
