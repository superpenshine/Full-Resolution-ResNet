# convert raw img(s) to h5py batchs

import numpy as np
import cv2
import h5py
import os.path
from tqdm import trange


def load_data(data_dir, num_datasets):

	if os.path.isfile(data_dir):
		hdf5_dir = data_dir
	else:
		raise ValueError("Wrong data type {}".format(data_type))

	data = []
	labels = []

	for idx in range(num_datasets):
		f = h5py.File(hdf_path, "r")
		data += [np.array(f['inputs']["dataset{}".format(idx)])]
		labels += [np.array(f['labels']["dataset{}".format(idx)])]

	data = np.concatenate(data)
	labels = np.concatenate(labels)

	# cv2.imshow('origin_image', labels[1])
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	return data, labels

if __name__ == '__main__':

	data = []
	labels = []
	# keep track of total dataset numbers
	cur_dataset_idx = 0
	# each dataset contains # of imgs
	dataset_size = 5
	# shape of hdf5 dataset N*H*W*C
	dataset_shape = (dataset_size, 720, 960, 3)
	# total number of images using
	num_img_using = 13
	# output hdf_path
	hdf_path = "E:\\data\\SYNTHIA\\data.h5"

	name_file = open("E:\\data\\SYNTHIA\\ALL.txt")
	img_names = name_file.readlines()
	name_file.close()

	f = h5py.File(hdf_path, "w")
	input_group = f.create_group('inputs')
	label_group = f.create_group('labels')

	for idx in trange(num_img_using):
		rgb_img_path = "E:\\data\\SYNTHIA\\RGB\\{}".format(img_names[idx].rstrip('\n'))
		gt_img_path = "E:\\data\\SYNTHIA\\GT\\{}".format(img_names[idx].rstrip('\n'))

		input_img = cv2.imread(rgb_img_path,1)
		label_img = cv2.imread(gt_img_path,1)

		# testing if image loaded correctly
		# cv2.imshow('origin_image', img)
		# cv2.waitKey(0)
		# cv2.destroyAllWindows()
		data += [input_img]
		labels += [label_img]

		# every time dataset_size number of image read, save it to hdf5
		if (idx + 1) % dataset_size == 0 and idx != 0:
			data = np.array(data)
			labels = np.array(labels)
			# row, column
			link = input_group.create_dataset("dataset{}".format(cur_dataset_idx), shape=(dataset_size, 720, 960, 3), data=data)
			label_group.create_dataset("dataset{}".format(cur_dataset_idx), shape=(dataset_size, 720, 960, 3), data=labels)
			print(link.chunks)
			cur_dataset_idx += 1
			data = []
			labels = []

	# if there's some image unsaved with the 'if' statement in previous for statement, save it.
	if num_img_using % dataset_size != 0:
		input_group.create_dataset("dataset{}".format(cur_dataset_idx), shape=(num_img_using % dataset_size, 720, 960, 3), data=data)
		label_group.create_dataset("dataset{}".format(cur_dataset_idx), shape=(num_img_using % dataset_size, 720, 960, 3), data=labels)

	f.close()
	print("{} pair of dataset (inputs + labels) have been created.".format(cur_dataset_idx + 1))

	# with h5py.File(hdf_path, "r") as f:
	# 	cv2.imshow('origin_image', f['labels']["dataset2"][0])
	# 	cv2.waitKey(0)
	# 	cv2.destroyAllWindows()

	load_data(hdf_path, 3)
	f.close()

