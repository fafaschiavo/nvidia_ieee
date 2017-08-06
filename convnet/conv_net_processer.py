import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
from os import listdir
from os.path import isfile, join

start_time = time.time()

mypath = 'demo/out_1.4cps_1step_pack1'
output_path = 'image_pickles'
files_list = [f for f in listdir(mypath) if isfile(join(mypath, f)) and '.xml' in f]

XINIT = 0
XEND = 2800
YINIT = 0
YEND = 2900
AREASTEP = 10

for file in files_list:
	x_list = []
	y_list = []

	current_x = XINIT
	current_y = YINIT
	while current_x < XEND:
		while current_y < YEND:
			y_list.append(0)
			current_y = current_y + AREASTEP

		current_y = YINIT
		x_list.append(y_list)
		y_list = []
		current_x = current_x + AREASTEP

	picture_array = np.asarray(x_list)
	pins_array = np.asarray(x_list)
	print 'Mesh prepared!'

	print 'Reading RAW file...'
	# outputTree = ET.parse("output_test.xml")
	outputTree = ET.parse(mypath + '/' + file)
	outputRoot = outputTree.getroot()

	ambulance_speed = []
	index = 0
	last_time_step = 0
	print 'Started processing...'
	for timestep in outputRoot.iter('timestep'):
		last_time_step = timestep.get('time')
		for vehicle in timestep:
			current_x = float(vehicle.get('x'))
			mesh_x = current_x//AREASTEP
			mesh_x = int(mesh_x)

			current_y = float(vehicle.get('y'))
			mesh_y = current_y//AREASTEP
			mesh_y = int(mesh_y)

			current_speed = float(vehicle.get('speed'))

			index = index + 1
			# print index
			# print '-------'

			picture_array[mesh_x][mesh_y] = picture_array[mesh_x][mesh_y] + current_speed
			pins_array[mesh_x][mesh_y] = pins_array[mesh_x][mesh_y] + 1

	last_time_step = float(last_time_step)
	average_speed_array = picture_array / last_time_step
	average_pins_array = pins_array / last_time_step

	with open(output_path + '/' + file + '_pins.pickle', 'wb') as f:
		pickle.dump(average_pins_array, f)

	with open(output_path + '/' + file + '_speed.pickle', 'wb') as f:
		pickle.dump(average_speed_array, f)

	print("--- %s seconds ---" % (time.time() - start_time))

	# plt.figure(figsize=(10,10), dpi=80)
	# cax = plt.imshow(average_pins_array, interpolation='nearest')
	# plt.colorbar(cax, orientation='horizontal')
	# plt.savefig(output_path + '/' + file + '_pins.jpeg')

		