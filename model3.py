import numpy as np
from os import listdir
from os.path import isfile, join
import pickle
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
import tensorflow as tf
from random import shuffle
import time
import gzip
import matplotlib.pyplot as plt

start_time = time.time()

print 'Hi there!'

training_amout = 0.8
number_of_epochs = 10

# big_data_pickle = '/Volumes/HD_M/big_data_pickle/'
# mypath = 'pickled4'
# images_path = '/Volumes/HD_M/images_4'

big_data_pickle = '/Volumes/HD_M/v2.5/big_data/' #folder onde eubi vou salvar os pickles mistos
mypath = 'v2.5/results_pickle' #folder com os xml picklados de results
images_path = 'v2.5/images' #folder com os pickles de imagem

NAME_total = []
X_total = []
Y_total = []
files_list = [f for f in listdir(mypath) if isfile(join(mypath, f)) and '.pickle' in f and 'res' in f]

images_list = [f for f in listdir(images_path) if '._out' not in f and '_speed' in f]




# index = 0
# one_hot_array = [0 for x in range(0, 100)]
# for image in images_list:
# 	current_code = image.split('step_')[1].split('.xml')[0]
# 	for file in files_list:
# 		if current_code in file:
# 			print file
# 			print image
# 			print '-----'

# 			with open(mypath + '/' + file, 'rb') as current_pickle:
# 				data_object = pickle.load(current_pickle)

# 			# with open(images_path + '/' + image, 'rb') as current_pickle:
# 			# 	image_object = pickle.load(current_pickle)

# 			current_pickle = gzip.open(images_path + '/' + image, 'rb')
# 			image_object = pickle.load(current_pickle)
# 			current_pickle.close()

# 			current_score = data_object['score']
# 			current_score = current_score * 100
# 			print current_score

# 			current_data_list = image_object

# 			# traffic_list = [0 for x in range(0,280)]
# 			# i = 0
# 			# for state in data_object['input_data']:
# 			# 	for x in xrange(0,3):
# 			# 		traffic_list[i+x] = state
# 			# 	i = i+3
# 			# traffic_list = np.asarray(traffic_list)
# 			# traffic_list = np.repeat([traffic_list], 50, 0)
# 			# traffic_list = np.transpose(traffic_list)
# 			# traffic_list = traffic_list.astype(int)

# 			# current_X = np.concatenate((current_data_list, traffic_list), axis = 1)
# 			current_X = np.asarray([current_data_list])

# 			# plt.figure(figsize=(10,10), dpi=80)
# 			# cax = plt.imshow(current_X, interpolation='nearest')
# 			# plt.colorbar(cax, orientation='horizontal')
# 			# plt.show()

# 			y_array = list(one_hot_array)
# 			y_array[int(current_score//1)] = 1
# 			current_Y = np.asarray(y_array)

# 			print int(current_score//1)

# 			index = index + 1
# 			print 'Current processing - ' + str(index) 

# 			dict_to_pickle = {'x': current_X, 'y': current_Y}
# 			with open(big_data_pickle + current_code + '.pickle', 'wb') as f:
# 				pickle.dump(dict_to_pickle, f)





big_data_pickle_list = [f for f in listdir(big_data_pickle) if isfile(join(big_data_pickle, f)) and '.pickle' in f]
index = 0
for file in big_data_pickle_list:
	index = index + 1
	print '-----'
	print 'Now processing - ' + str(index)
	with open(big_data_pickle + '/' + file, 'rb') as current_pickle:
		data_object = pickle.load(current_pickle)

	X_total.append(data_object['x'])
	Y_total.append(data_object['y'][:50])



X = X_total[:int(index*training_amout)]
Y = Y_total[:int(index*training_amout)]
test_x = X_total[int(index*training_amout):]
test_y = Y_total[int(index*training_amout):]

print X[0].shape

my_net = input_data(shape=X[0].shape, name='input')

my_net = conv_2d(my_net, 32, 2, activation='relu')
my_net = max_pool_2d(my_net, 2)

my_net = fully_connected(my_net, 100, activation='relu')

my_net = fully_connected(my_net, 100, activation='relu')

my_net = dropout(my_net, 0.8)

my_net = fully_connected(my_net, 50, activation='softmax')
my_net = regression(my_net, optimizer='adam', learning_rate=0.1, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(my_net)

model.fit(X, Y, validation_set=({'input': test_x}, {'targets': test_y}), n_epoch=number_of_epochs, show_metric=True, snapshot_step=100)
model.save('models/model_1_0_1_discrete.model')

for index in xrange(0,10):
	print '----------'
	argmax_test = np.argmax(test_y[index])
	print'expected'
	print argmax_test

	pred = model.predict([test_x[index]])[0]
	argmax_pred = np.argmax(pred)
	print 'predicted'
	print argmax_pred

print("--- %s seconds ---" % (time.time() - start_time))





