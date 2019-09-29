 #  This python file takes samples per second from a video 
 #  And determines the amount of information that it has using summing all the pixel intensities in 
 #  its grayscale version 
import cv2
import numpy as np
import os
import time
import math

# Global variables 
number_samples = 100
length_of_file = 60

#  Take samples every one second
def extract_image_one_fps(video_source_path):
	# number_samples = 100
	length_of_file = 60	
	vidcap = cv2.VideoCapture(video_source_path)
	count = 0
	success = True
	while success:
		vidcap.set(cv2.CAP_PROP_POS_MSEC,((length_of_file/number_samples)*count*1000))      
		success,image = vidcap.read()

		## Stop when last frame is identified
		# image_last = cv2.imread("./photos/frame{}.png".format(count-1))
		# if np.array_equal(image,image_last):
		#     break

		cv2.imwrite("./photos/%d.jpg" % count, image)     # save frame as PNG file
		print ('{}reading a new frame: {} '.format(count,success))
		count += 1


def take_l2_norm():
	threshold = 10
	for i in range(number_samples):
		norm = 0
		# Convert to grayscale
		image = cv2.imread('./photos/{}.jpg'.format(i), 0) 
		rows, cols = image.shape
		for row in range (rows ):
			for col in range (cols ):
				norm += image[row][col]
		norm = pow(norm, 1/6)	
		print(norm)	
		if norm >= threshold:	
			print("This frame ./photos/{}.jpg contains noticable features".format(i))
		else:
			print()
			print("This frame ./photos/{}.jpg doesnot contains noticable features".format(i))




if __name__ == "__main__":
	try:
		os.mkdir('photos')
	except OSError:
		pass
	# extract_image_one_fps('anne-marie-live.mp4')
	take_l2_norm()
