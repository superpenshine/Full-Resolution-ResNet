# convert raw img(s) to h5py batchs

import numpy as np
import cv2
import h5py
from tqdm import trange


if __name__ == '__main__':

	data = []
	# keep track of total dataset numbers
	cur_dataset_num = 0
	# each dataset contains # of imgs
	dataset_size = 5
	# total number of images using
	num_img_using = 13
	# output hdf_path
	hdf_path = "E:\\data\\SYNTHIA\\RGB.h5"

	name_file = open("E:\\data\\SYNTHIA\\ALL.txt")
	img_names = name_file.readlines()
	name_file.close()

	f = h5py.File(hdf_path, "w")
	g = f.create_group('images')

	for idx in trange(num_img_using):

		img_path = "E:\\data\\SYNTHIA\\RGB\\{}".format(img_names[idx].rstrip('\n'))
		img = cv2.imread(img_path,1)

		# testing if image loaded correctly
		# cv2.imshow('origin_image', img)
		# cv2.waitKey(0)
		# cv2.destroyAllWindows()
		data += [img]

		if (idx + 1) % dataset_size == 0 and idx != 0:
			data = np.array(data)
			# row, column
			g.create_dataset("dataset{}".format(cur_dataset_num), shape=(dataset_size, 720, 960, 3), data=data)
			cur_dataset_num += 1
			data = []

	if num_img_using % dataset_size != 0:
		g.create_dataset("dataset{}".format(cur_dataset_num), shape=(num_img_using % dataset_size, 720, 960, 3), data=data)

	f.close()
	print("{} number of dataset have been created.".format(cur_dataset_num))

	# with h5py.File(hdf_path, "r") as f:
	# 	cv2.imshow('origin_image', f['images']["dataset2"][0])
	# 	cv2.waitKey(0)
	# 	cv2.destroyAllWindows()
	# f.close()

