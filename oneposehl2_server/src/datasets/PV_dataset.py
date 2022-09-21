import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse
import os
import time

class PVDataset:
	def __init__(self, data_dir, gray = True):
		self.data_dir = data_dir
		self.img_dir = os.path.join(self.data_dir, 'PV')
		self.img_lst = os.listdir(self.img_dir)
		self.img_lst = list(filter(lambda x:x[-3:]=='png',self.img_lst))
		self.img_lst.sort()
		self.img_lst_pth = list(map(lambda x:os.path.join(self.img_dir, x), self.img_lst))
		self.gray = gray
		
	def __getitem__(self, idx):
		img_pth = self.img_lst_pth[idx]
		if self.gray == True:
			return cv2.imread(img_pth, 0)
		else:
			return cv2.imread(img_pth)
		
	def __len__(self):
		return len(self.img_lst)
		
	def plot(self, idx):
		img = self[idx]
		plt.imshow(img)
		plt.show()

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--img_dir', help = 'path to directory of images', type=str)
	args = parser.parse_args()
	
	dataset = PVDataset(args.img_dir)
	
	i = 0
	while True:
		print(i, dataset.img_lst[i])
		out = dataset[i]
		cv2.imshow('frame', out)
		time.sleep(0.01)
		i += 1
		if i == len(dataset):
			i = 0
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

if __name__ == "__main__":
	main()
