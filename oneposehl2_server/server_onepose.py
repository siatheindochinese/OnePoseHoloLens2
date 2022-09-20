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
import io
from concurrent import futures
from multiprocessing.managers import BaseManager

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

class QueueManager(BaseManager):
	pass

pose_queue = queue.Queue()
def return_pose_queue():
	global pose_queue
	return pose_queue
	
def load_lut(lut_filename):
    with open(lut_filename, mode='rb') as depth_file:
        lut = np.frombuffer(depth_file.read(), dtype="f")
        lut = np.reshape(lut, (-1, 3))
    return lut

class Streamer(ultra_pb2_grpc.VTServiceServicer):
	@torch.no_grad()
	def __init__(self, qm, stop_event, cfg):
		self.mapping_probe = np.loadtxt(os.path.join('markers', cfg.probe + '.txt'), dtype=str)
		self.mapping_anchor = np.loadtxt(os.path.join('markers', cfg.anchor + '.txt'), dtype=str)
		self.calib = np.loadtxt(os.path.join('markers', cfg.calib + '.txt'), dtype=str)
		self.left_lut = np.flip(load_lut('device_params/' + cfg.device + '/VLC LF_lut.bin').reshape((480,640,-1)), 0)
		self.right_lut = np.flip(load_lut('device_params/' + cfg.device + '/VLC RF_lut.bin').reshape((480,640,-1)), 1)
		trans = self.calib[0].astype(float)
		rot = R.from_euler('xyz', self.calib[1].astype(float), degrees=True)
		probe_calib7 = np.zeros(7)
		probe_calib7[:3] = trans.squeeze()
		probe_calib7[3:] = rot.as_quat().squeeze()
		self.probe_calib7 = ','.join(np.round(probe_calib7, 10).astype(str))
		
		# queue manager
		self.managerq = qm
		self.poseq = qm.get_pose_queue() 
		
		# some grpc attribute
		self._stop_event = stop_event
		
		# misc, might be useful for network traffic later?
		self.queue = queue.Queue()
		
		# import onepose configs, utils and files here
		self.matching_model, self.extractor_model = load_model(cfg)
		self.yolov5_detector = YoloV5Detector(cfg.yolov5_dir, cfg.yolov5_weights_dir)
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
		self.K_full = np.loadtxt(cfg.intrin)
		
		# some OneWay flags
		self.init = False
		self.previous_frame_pose = np.eye(4)
		self.previous_inliers = []
	
	##############################################
	# Utility function to be used in OneWay(...) #
	# Detects desired object                     #
	##############################################
	@torch.no_grad()
	def detect_object(self, inp):
		if self.init == False:
			bbox, inp_crop, K_crop = self.yolov5_detector.detect(inp, self.K_full)
			self.init = True
		else:
			if len(self.previous_inliers) < 8:
				bbox, inp_crop, K_crop = self.yolov5_detector.detect(inp, self.K_full)
			else:
				bbox, inp_crop, K_crop = self.yolov5_detector.previous_pose_detect(inp, self.K_full, self.previous_frame_pose, self.bbox3d)
		
		object_detected = not (bbox == np.array([0, 0, 640, 480])).all()
		return object_detected, bbox, inp_crop, K_crop
		
	##############################################
	# Utility function to be used in OneWay(...) #
	# Passes input through OnePose pipeline      #
	##############################################
	@torch.no_grad()
	def oneposeFwdPass(self, inp_crop, K_crop):
		inp_crop_cuda = torch.from_numpy(inp_crop.astype(np.float32)[None][None]/255.).cuda()
		pred_detection = self.extractor_model(inp_crop_cuda)
		pred_detection = {k: v[0].cpu().numpy() for k, v in pred_detection.items()}
		inp_data = pack_data(self.avg_descriptors3d, self.clt_descriptors, self.keypoints3d, pred_detection, np.array([640,480]))
		pred, _ = self.matching_model(inp_data)
		matches = pred['matches0'].detach().cpu().numpy()
		valid = matches > -1
		notvalid = matches <= -1
		kpts2d = pred_detection['keypoints']
		kpts3d = inp_data['keypoints3d'][0].detach().cpu().numpy()
		confidence = pred['matching_scores0'].detach().cpu().numpy()
		mkpts2d, mkpts3d, mconf = kpts2d[valid], kpts3d[matches[valid]], confidence[valid]
		validcorners = mkpts2d
		notvalidcorners = kpts2d[notvalid]
		_, pose_pred_homo, inliers = eval_utils.ransac_PnP(K_crop, mkpts2d, mkpts3d, scale=1000)
		return pose_pred_homo, inliers, validcorners, notvalidcorners

	@torch.no_grad()
	def OneWay(self, request, context):
		'''
		Process info from request and package them into ultrapb2.PoseData format
		context is f-ing useless lmao
		'''
		# drop packets from RF camera
		if request.cameraID == "Right":
			return ultra_pb2.PoseData(USData=b'\x01',
									  VolData = b'\x01',
									  PoseMat = "null_pose",
									  CalibMat = self.probe_calib7,
									  AnchorMat = "null_pose",
									  RenderFrame = "False",
									  RenderVol = "False",
									  MiscMsg = "Misc")
		
		print('LF packet received')
		# parse image
		inp = np.frombuffer(request.ImageData, np.uint8).reshape((480, 640))
		inp = cv2.flip(cv2.transpose(inp), int(request.cameraID == "Left")) # GRAY2BGR? yolo takes RGB
		# is this even upright? 640x480 instead of 480x640, need to check
		
		# parse cam2world matrix
		LF2World = request.Matrix
		LF2World = np.array(LF2World.split(",")[:-1]).astype(float).reshape(4,4).transpose()
		
		# onepose processing
		##### object detection
		object_detected, bbox, inp_crop, K_crop = self.detect_object(inp)
		if object_detected:
			print('object detected')
		else:
			print('object not detected')
		
		##### onepose pipeline
		if object_detected:
			pose_pred_homo, inliers, validcorners, notvalidcorners = self.oneposeFwdPass(inp_crop, K_crop)
			self.previous_frame_pose = pose_pred_homo
			self.previous_inliers = inliers
			print('OnePosed')
		else:
			self.previous_frame_pose = np.eye(4)
			self.previous_inliers = []
		
		##### export results
		if object_detected and not np.array_equal(pose_pred_homo, np.eye(4)):
			pose_pred_homo[1, 3] *= -1
			pose_pred_homo = pose_pred_homo @ LF2World
			R3 = R.from_matrix(pose_pred_homo[:-1,:-1])
			T3 = pose_pred_homo[:-1,-1]
			result = np.zeros(7)
			result[:3] = T3.squeeze()                   
			result[3:] = R3.as_quat().squeeze()
			result = ','.join(np.round(result, 10).astype(str)) 
		else:
			result = "null_pose"
		
		USData = b'\x01'
		VolData = b'\x01'
		PoseMatData = result
		CalibMatData = self.probe_calib7
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

@torch.no_grad()
@hydra.main(config_path='configs/', config_name='config.yaml')
def main(cfg):
	os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
	QueueManager.register('get_pose_queue', callable=return_pose_queue)
	qm = QueueManager(address=('127.0.0.1', 5005), authkey=b'abc')
	qm.start()
	server = grpc.server(futures.ThreadPoolExecutor(max_workers=20))
	
	stop_event = threading.Event()
	ultra_pb2_grpc.add_VTServiceServicer_to_server(Streamer(qm, stop_event, cfg), server)
	server.add_insecure_port('[::]:50052')
	server.start()
	print('server start')
	#server.wait_for_termination()
	stop_event.wait()
	server.stop(5)

if __name__ == '__main__':
	main()
