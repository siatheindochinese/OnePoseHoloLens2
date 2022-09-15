#!/usr/bin/env python
# coding: utf-8

# server imports
import ultra_pb2_grpc
import ultra_pb2
import grpc
import time
import datetime
from scipy.spatial.transform import Rotation as R
import threading
import queue
import os
import argparse
import io

# onepose imports
import cv2
import glob
import torch
import hydra
from tqdm import tqdm
import os.path as osp
import numpy as np
from PIL import Image
from loguru import logger
from torch.utils.data import DataLoader
from src.utils import data_utils, path_utils, eval_utils, vis_utils
from src.yolov5_detector import YoloV5Detector
from pytorch_lightning import seed_everything
from inference import load_model, pack_data

class Streamer(ultra_pb2_grpc.VTServiceServicer):
	@torch.no_grad()
	@hydra.main(config_path='configs/', config_name='config.yaml')
	def __init__(self, args, qm, stop_event, cfg = cfg):
		'''
		args has probe, anchor, calib, device, markerdict
		'''
		self.mapping_probe = np.loadtxt(os.path.join('markers', args.probe + '.txt'), dtype=str)
		self.mapping_anchor = np.loadtxt(os.path.join('markers', args.anchor + '.txt'), dtype=str)
		self.calib = np.loadtxt(os.path.join('markers', args.calib + '.txt'), dtype=str)
		self.left_lut = np.flip(load_lut('device_params/' + args.device + '/VLC LF_lut.bin').reshape((480,640,-1)), 0)
		self.right_lut = np.flip(load_lut('device_params/' + args.device + '/VLC RF_lut.bin').reshape((480,640,-1)), 1)
		
		# queue manager
		self.managerq = qm
		self.poseq = qm.get_pose_queue() 
		
		# some grpc attribute
		self._stop_event = stop_event
		
		# misc, might be useful for network traffic later?
		self.queue = queue.Queue()
		
		# import onepose configs, utils and files here
		self.matching_model, self.extractor_model = load_model(cfg)
		self.yolov5_detector = YoloV5Detector('/home/intern1/yolov5/','data/yolov5/ufcoco.pt')
		self.anno_dir = osp.join(cfg.sfm_model_dir, f'outputs_{cfg.network.detection}_{cfg.network.matching}', 'anno')
		self.avg_anno_3d_path = osp.join(self.anno_dir, 'anno_3d_average.npz')
		self.clt_anno_3d_path = osp.join(self.anno_dir, 'anno_3d_collect.npz')
		self.idxs_path = osp.join(self.anno_dir, 'idxs.npy')
		self.num_leaf = cfg.num_leaf
		self.avg_data = np.load(self.avg_anno_3d_path)
		self.clt_data = np.load(self.clt_anno_3d_path)
		self.idxs = np.load(self.idxs_path)
		self.bbox3d = np.loadtxt(cfg.box3d_path)
		self.keypoints3d = torch.Tensor(self.clt_data['keypoints3d']).cuda()
		self.num_3d = self.keypoints3d.shape[0]
		self.avg_descriptors3d, _ = data_utils.pad_features3d_random(self.avg_data['descriptors3d'], self.avg_data['scores3d'], self.num_3d)
		self.clt_descriptors, _ = data_utils.build_features3d_leaves(self.clt_data['descriptors3d'], self.clt_data['scores3d'], self.idxs, self.num_3d, self.num_leaf)
		self.K_full = np.loadtxt('intrin_LF.txt') # might wanna put this in hydra config instead
		
		# some OneWay flags
		self.init = False
		self.previous_frame_pose = np.eye(4)
		self.previous_inliers = []

	@torch.no_grad()
	@hydra.main(config_path='configs/', config_name='config.yaml')
	def OneWay(self, request, context):
		'''
		Process info from request and package them into ultrapb2.PoseData format
		context is f-ing useless lmao
		'''
		# drop packets from RF camera
		if request.camera_ID == "right":
			return ultra_pb2.PoseData(USData=b'\x01',
									  VolData = b'\x01',
									  PoseMat = "null_pose",
									  CalibMat = np.zeros(7),
									  AnchorMat = "null_pose",
									  RenderFrame = "False",
									  RenderVol = "False",
									  MiscMsg = "Misc")
		
		# parse image
		inp = np.frombuffer(request.ImageData, np.uint8).reshape((480, 640))
		inp = cv2.flip(cv2.transpose(inp), int(cameraID == "Left")) # GRAY2BGR? yolo takes RGB
		# is this even upright? 640x480 instead of 480x640, need to check
		
		# parse cam2world matrix
		LF2World = request.Matrix
		LF2World = np.array(LF2World.split(",")[:-1]).astype(float).reshape(4,4).transpose()
		
		# onepose processing
		#	object detection
		if self.init == False:
			bbox, inp_crop, K_crop = self.yolov5_detector.detect(inp, self.K_full)
			self.init == True
		else:
			if len(self.previous_inliers) < 8:
				bbox, inp_crop, K_crop = self.yolov5_detector.detect(inp, self.K_full)
			else:
				bbox, inp_crop, K_crop = self.yolov5_detector.previous_pose_detect(inp, self.K_full, self.previous_frame_pose, self.bbox3d)
		
		#	determine if object is detected
		object_detected = not (bbox == np.array([0, 0, 640, 480])).all()
		
		#	pass through onepose pipeline if object detected and return pose
		#	else, just reset the previous pose and inliers and return null_pose
		if object_detected:
			inp_crop_cuda = (torch.from_numpy(inp_crop).astype(np.float32))[None][None]/ 255.).cuda()
			pred_detection = self.extractor_model(inp_crop_cuda)
			pred_detection = {k: v[0].cpu().numpy() for k, v in pred_detection.items()}
			inp_data = pack_data(avg_descriptors3d, clt_descriptors, keypoints3d, pred_detection, np.array([640,480]))
			pred, _ = matching_model(inp_data)
			matches = pred['matches0'].detach().cpu().numpy()
			valid = matches > -1
			kpts2d = pred_detection['keypoints']
			kpts3d = inp_data['keypoints3d'][0].detach().cpu().numpy()
			confidence = pred['matching_scores0'].detach().cpu().numpy()
			mkpts2d, mkpts3d, mconf = kpts2d[valid], kpts3d[matches[valid]], confidence[valid]
			pose_pred, pose_pred_homo, inliers = eval_utils.ransac_PnP(K_crop, mkpts2d, mkpts3d, scale=1000)
			self.previous_frame_pose = pose_pred_homo
			self.previous_inliers = inliers
			result = pose_pred_homo
			result = result @ LF2World
		else:
			self.previous_frame_pose = np.eye(4)
			self.previous_inliers = []
			result = "null_pose"
		
		USData = b'\x01'
		VolData = b'\x01'
		PoseMatData = result
		CalibMatData = np.zeros(7)
		AnchorMatData = "null_pose"
		RenderFrame = "False"
		RenderVol = "False"
		MiscMsgData = "Misc"
		return ultra_pb2.PoseData(USData=USData, 
								  VolData=VolData, 
								  PoseMat=PoseMatData, 
								  CalibMat=CalibMatData, 
								  AnchorMat=AnchorMatData, 
								  RenderFrame=RenderFrame,
								  RenderVol=RenderVol,
								  MiscMsg=MiscMsgData)            

def serve(qm):
	server = grpc.server(futures.ThreadPoolExecutor(max_workers=20))
	
	parser = argparse.ArgumentParser()
	parser.add_argument('-p', '--probe', default='m0', help="filename for marker layout")
	parser.add_argument('-a', '--anchor', default='m1', help="filename for marker layout")
	parser.add_argument('-c', '--calib', default='calib1', help="filename for marker-tip calibration")
	parser.add_argument('-d', '--device', default='NUHS-HOLO1', help="specify device ID")
	parser.add_argument('-m', '--markerdict', default='DICT_4X4_50', help="aruco dictionary")
	args = parser.parse_args()
	
	stop_event = threading.Event()
	ultra_pb2_grpc.add_VTServiceServicer_to_server(Streamer(args, qm, stop_event), server)
	server.add_insecure_port('[::]:50061')
	server.start()
	print('server start')
	#server.wait_for_termination()
	stop_event.wait()
	server.stop(5)

if __name__ == '__main__':
	os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
	QueueManager.register('get_pose_queue', callable=return_pose_queue)
	qm = QueueManager(address=('127.0.0.1', 5005), authkey=b'abc')
	qm.start()
	serve(qm)
