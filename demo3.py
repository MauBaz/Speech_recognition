from __future__ import division, print_function, absolute_import
import sys
import speech_data2
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM,Dropout
import numpy as np
from keras import optimizers

# from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
# from sklearn.metrics import roc_auc_score




data_size = 2402
iters=0

width = 20  # mfcc features
height = 80  # (max) length of utterance
classes = 10  # digits
Valpercent = 0.1
SPS_per_epoch = int(2402 - 2402*Valpercent)
VALSPS = 2402 - SPS_per_epoch
NUMEPOCHS = 10

Xs,ys = speech_data2.mfcc_batch_generator(data_size)
X_train,y_train = Xs[:SPS_per_epoch,:,:],ys[:SPS_per_epoch,:]
X_val, y_val = Xs[SPS_per_epoch:,:,:],ys[SPS_per_epoch:,:]

# X_test, y_test = np.asarray(Xv), np.asarray(Yv) #overfit for now
# 
# space = {
# 
#             'units1': hp.choice('layers', np.arange(100, 256, dtype=int)),
# 
# 
#             'dropout1': hp.uniform('dropout1', .25,.75),
#             'dropout2': hp.uniform('dropout2',  .25,.75),
# 
# 
# 
#             'nb_epochs' :  NUMEPOCHS,
#             'optimizer': hp.choice('optimizer',['adadelta','adam','rmsprop'])
#         }


print('Build model...')
model = Sequential()
## Keras' convention is that the batch dimension (number of examples (not the same as timesteps)) is typically omitted in the input_shape arguments. The batching (number of examples per batch) is handled in the fit call. I hope that helps.
model.add(LSTM(16, input_shape=[width,height],recurrent_dropout=0.7))
model.add(Dropout(0.4))
model.add(Dense(classes,activation='softmax'))

model.compile(loss='categorical_crossentropy',
			  optimizer='adam',
			  metrics=['accuracy'])

## Recall can otherwise do steps_per_epoch = SPS_per_epoch and set batch_size to None
## AND REMOVE THE ITERATING!!!

dotheshuffle = np.random.permutation(data_size)
for iters in range(NUMEPOCHS):
	print('Epoch #: ',iters+1)
	dotheshuffle = np.random.permutation(SPS_per_epoch)
	keepdoingtheshuffle = np.random.permutation(VALSPS)
	X_train,y_train = X_train[dotheshuffle,:,:],y_train[dotheshuffle,:]
	X_val,y_val = X_val[keepdoingtheshuffle,:,:],y_val[keepdoingtheshuffle,:]
	model.fit(X_train[dotheshuffle,:,:], y_train[dotheshuffle,:], batch_size=None,epochs=1,
		  verbose=2,steps_per_epoch=SPS_per_epoch)
	# score, acc = model.evaluate( was x_test,y_test here
	#                             batch_size=batch_size)
	_,acc = model.evaluate(X_val,y_val)
	# print('Test score:', score)
	print('Test accuracy:', acc)

	iters+=1

