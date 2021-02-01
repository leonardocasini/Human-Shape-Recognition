import cv2
import os
import numpy as np
from scipy.io import loadmat

import time


def CreateDatasetRun(run, directoryTrainingSet,directoryDataset):
	
	directoryRun = os.path.join(directoryTrainingSet, run)
	directoryDatasetRun = os.path.join(directoryDataset, run)
	for subfolder in os.listdir(directoryRun):
		directorySubFolderRun = os.path.join(directoryRun, subfolder)
		print(directorySubFolderRun)
		directorySubFolderDataset = os.path.join(directoryDatasetRun, subfolder)
		if(os.path.exists(directorySubFolderDataset) == False):
			os.mkdir(directorySubFolderDataset) 
			#non ancora caricato
			#Array of frames
			x = []
			#Array of shape parameters
			y = []
			for filename in os.listdir(directorySubFolderRun):
				if filename.endswith("info.mat"):
					pathToInfoMat = os.path.join(directorySubFolderRun, filename)
					mat = loadmat(pathToInfoMat)
					seq = mat['sequence'][0]
					mp4filename = seq+".mp4"
					pathToVideo = os.path.join(directorySubFolderRun, mp4filename)
					theta = []
					for i in range(len(mat['shape'])):
						theta.append(mat['shape'][i][0])
					#open video
					cap= cv2.VideoCapture(pathToVideo)
					while(cap.isOpened()):
						ret, frame = cap.read()
						if ret == False:
							break
						x.append(frame)
						y.append(theta) 
					continue
				else:
					continue
			x = np.array(x)
			y = np.array(y)
			#Creating path
			np.save(directorySubFolderDataset+"/frames_"+subfolder+".npy",x)
			np.save(directorySubFolderDataset+"/thetas_"+subfolder+".npy",y)
		else:
			print("already stored")


# # Opens the Video file
start_time = time.time()
directoryTrainingSet = "/media/leonardo/Elements/cmu/train/"
directoryDataset = "Dataset/"
run = "run2"
CreateDatasetRun(run,directoryTrainingSet,directoryDataset)
print("--- %s seconds ---" % (time.time() - start_time))