import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse
import os
import re
import time

class StereoDataset:
	def __init__(self, img_dir):
		self.img_dir = img_dir
		self.img_lst = os.listdir(self.img_dir)
		self.img_pairs = dict()
		self.img_lst_L = list(filter(lambda x: 'Left' in x, self.img_lst))
		self.img_lst_Lt = list(map(lambda x: int(x[:18]), self.img_lst_L))
		self.img_lst_R = list(filter(lambda x: 'Right' in x, self.img_lst))
		self.img_lst_Rt = list(map(lambda x: int(x[:18]), self.img_lst_R))
		
		for i in range(len(self.img_lst_Lt)):
			img_L = self.img_lst_L[i]
			argmin = 0
			minval = np.Infinity
			for j in range(len(self.img_lst_Rt)):
				score = abs(self.img_lst_Lt[i] - self.img_lst_Rt[j])
				if score < minval:
					minval = score
					argmin = j
			img_R = self.img_lst_R[argmin]
			time = self.img_lst_Lt[i]
			self.img_pairs[time] = {'L': img_L, 'R': img_R}
		
		self.time_lst = list(self.img_pairs.keys())
		self.time_lst.sort()
		
	def __getitem__(self, idx):
		time = self.time_lst[idx]
		img_pair = self.img_pairs[time]
		L_pth, R_pth = img_pair['L'], img_pair['R']
		L_pth, R_pth = os.path.join(self.img_dir, L_pth), os.path.join(self.img_dir, R_pth)
		fL, fR = open(L_pth,'rb'), open(R_pth,'rb')
		L, R = np.frombuffer(fL.read(), dtype = np.uint8), np.frombuffer(fR.read(), dtype = np.uint8)
		fL.close(); fR.close();
		L, R = L.reshape(480,640), R.reshape(480,640)
		L, R = cv2.rotate(L, cv2.ROTATE_90_CLOCKWISE), cv2.rotate(R, cv2.ROTATE_90_COUNTERCLOCKWISE)
		return L, R
		
	def __len__(self):
		return len(self.time_lst)
		
	def plot(self, idx):
		L, R = self[idx]
		result = np.concatenate((L,R),axis=1)
		plt.imshow(result)
		plt.show()
		
	def cv2_out(self, idx):
		L, R = self[idx]
		result = np.concatenate((L,R),axis=1)
		return result

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--img_dir', help = 'path to directory of images', type=str)
	args = parser.parse_args()
	
	dataset = StereoDataset(args.img_dir)
	
	i = 0
	while True:
		print(i, dataset.time_lst[i])
		out = dataset.cv2_out(i)
		cv2.imshow('frame', out)
		time.sleep(0.01)
		i += 1
		if i == len(dataset):
			i = 0
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

if __name__ == "__main__":
	main()
	
