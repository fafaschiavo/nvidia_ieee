import numpy as np
from os import listdir
from os.path import isfile, join
import pickle
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf
from random import shuffle

print 'Hi there!'

training_amout = 0.8
number_of_epochs = 10

mypath = 'pickled4'
NAME_total = []
X_total = []
Y_total = []
files_list = [f for f in listdir(mypath) if isfile(join(mypath, f)) and '.pickle' in f]

index = 0
for file in files_list:
	with open(mypath + '/' + file, 'rb') as current_pickle:
		data_object = pickle.load(current_pickle)

	current_data_list = data_object['input_data']
	X_total.append(np.asarray([current_data_list]))
	current_score = data_object['score']
	Y_total.append(np.asarray([current_score]))
	NAME_total.append(file)
	print current_score

	index = index + 1
	print 'Current processing - ' + str(index) 

X = X_total[:int(index*training_amout)]
Y = Y_total[:int(index*training_amout)]
NAME = NAME_total[:int(index*training_amout)]
test_x = X_total[int(index*training_amout):]
test_y = Y_total[int(index*training_amout):]
NAME_test = NAME_total[int(index*training_amout):]

my_net = input_data(shape=X[0].shape, name='input')

my_net = fully_connected(my_net, 32, activation='relu')

my_net = fully_connected(my_net, 32, activation='relu')

# my_net = fully_connected(my_net, 32, activation='relu')

# my_net = fully_connected(my_net, 32, activation='relu')

# my_net = fully_connected(my_net, 1024, activation='relu')

# my_net = fully_connected(my_net, 1024, activation='relu')

# my_net = fully_connected(my_net, 1024, activation='relu')

# my_net = fully_connected(my_net, 1024, activation='relu')

# my_net = fully_connected(my_net, 1024, activation='tanh')

my_net = dropout(my_net, 0.9)

my_net = fully_connected(my_net, 1, activation='linear')
my_net = regression(my_net, optimizer='sgd', learning_rate=0.01, loss='mean_square', name='targets')

model = tflearn.DNN(my_net)

model.fit(X, Y, validation_set=({'input': test_x}, {'targets': test_y}), n_epoch=number_of_epochs, show_metric=True, snapshot_step=100)
model.save('models/model_1_0_3.model')

# model.load('models/model_1_0_2.model')

total_norm = 0
total_distance = 0
expected_ranking = []
found_ranking = []
global_average = 0
global_variance = 0
for index in xrange(0,len(test_x)):
	possible_result = model.predict([test_x[index]])[0]
	expected = test_y[index]
	
	diff = possible_result - expected
	diff = diff[0]
	total_distance = total_distance + abs(diff)
	
	expected = expected[0]
	total_norm = total_norm + abs(expected)

	global_average = total_norm/len(test_x)

	error_percentage = total_distance/total_norm
	error_percentage = error_percentage*100

	possible_result = possible_result[0]
	expected_ranking.append(expected)
	found_ranking.append(possible_result)

expected_ranking = [x for (y,x) in sorted(zip(expected_ranking,NAME_test))]
found_ranking = [x for (y,x) in sorted(zip(found_ranking,NAME_test))]
random_ranking = NAME_test
shuffle(random_ranking)

total_ranking_error = 0
random_ranking_error = 0
for x in xrange(0,len(test_x)):
	current_name = NAME_test[x]
	expected_index = expected_ranking.index(current_name)
	found_index = found_ranking.index(current_name)
	random_index = random_ranking.index(current_name)
	total_ranking_error = total_ranking_error + abs(expected_index-found_index)
	random_ranking_error = random_ranking_error + abs(expected_index-random_index)

diff_for_variance = 0
for value in test_y:
	diff_for_variance = diff_for_variance + abs(value-global_average)
global_variance = diff_for_variance/len(test_y)

print '*************************************************'
print 'training size - ' + str(len(X))
print 'validation size - ' + str(len(test_x))
print 'Number of EPOCHS - ' + str(number_of_epochs)
print 'Average distance - ' + str(total_distance/len(test_x))
print 'error_percentage - ' + str(error_percentage) + '%'
print 'Accuracy - ' + str(100 - error_percentage) + '%'
print 'Total ranking error - ' + str(total_ranking_error)
print 'Totally random error - ' + str(random_ranking_error)
print 'Global average - ' + str(global_average)
print 'Global variance - ' + str(global_variance)
print '*************************************************'

print 'Lets play ----------------------'
print 'text sample ----'

for x in xrange(1,30):
	print 'expected - ' + str(test_y[-x])
	possible_result = model.predict([test_x[-x]])[0]
	print 'found - ' + str(possible_result)
	print '------------'

# print 'expected - ' + str(test_y[-1])
# possible_result = model.predict([test_x[-1]])[0]
# print 'found - ' + str(possible_result)

# print 'text sample ----'
# print 'expected - ' + str(test_y[-50])
# possible_result = model.predict([test_x[-50]])[0]
# print 'found - ' + str(possible_result)

# print 'text sample ----'
# print 'expected - ' + str(test_y[-60])
# possible_result = model.predict([test_x[-60]])[0]
# print 'found - ' + str(possible_result)

# print 'text sample ----'
# print 'expected - ' + str(test_y[-100])
# possible_result = model.predict([test_x[-100]])[0]
# print 'found - ' + str(possible_result)







