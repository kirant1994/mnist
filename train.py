import numpy as np
from dataloader import get_data
from models import Net
import tensorflow as tf
from tensorflow.keras.utils import Progbar
import pickle
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

BATCH_SIZE = 16
EPOCHS = 10
MODEL_NAME = 'fcmodel'

X_train, X_val, y_train, y_val = get_data()
print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)

train_set = tf.data.Dataset.from_tensor_slices((X_train, y_train))
val_set = tf.data.Dataset.from_tensor_slices((X_val, y_val))
train_set_batch = train_set.batch(BATCH_SIZE)
val_set_batch  = val_set.batch(BATCH_SIZE)

num_batches = len(X_train) // BATCH_SIZE

net = Net()
optim = tf.keras.optimizers.Adam(lr=1e-5)
loss = tf.keras.losses.sparse_categorical_crossentropy

@tf.function
def train_step(batch):
	batch_x, batch_y = batch
	with tf.GradientTape() as tape:
		batch_pred = net(batch_x)
		loss_output = loss(batch_y, batch_pred)
		loss_output = tf.reduce_mean(loss_output)
	grad = tape.gradient(loss_output, net.trainable_variables)
	optim.apply_gradients(zip(grad, net.trainable_variables))
	return loss_output

@tf.function
def val_step(batch):
	batch_x, batch_y = batch
	batch_pred = net(batch_x)
	loss_output = loss(batch_y, batch_pred)
	loss_output = tf.reduce_mean(loss_output)
	return loss_output

train_loss_list = []
val_loss_list = []

# Initialising the loss file
with open('weights/losses.txt', 'w') as file:
	pass

# Training
for epoch in range(EPOCHS):
	idx = 0
	prog = Progbar(num_batches)
	train_loss =[] 
	for batch in train_set_batch:
		loss_output = train_step(batch)
		prog.update(idx, [('loss', loss_output.numpy()), ('epoch', epoch+1)])
		train_loss.append(loss_output.numpy())
		idx += 1

	val_loss = []
	for batch in val_set_batch:
		 val_loss.append(val_step(batch).numpy())
	prog.update(idx, [('val_loss', np.mean(val_loss))])

	train_loss_list.append(np.mean(train_loss))
	val_loss_list.append(np.mean(val_loss))

	with open('weights/losses.txt', 'a') as file:
		file.write('{0:.6f}\t{1:.6f}\n'.format(np.mean(train_loss), np.mean(val_loss)))
	net.save_weights('weights/{0:s}.{1:d}.h5'.format(MODEL_NAME, epoch))

