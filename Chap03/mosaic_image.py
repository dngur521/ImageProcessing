# 필요한 패키지를 import함
from __future__ import print_function
import argparse
import cv2
import numpy as np
import random

def mosaic_image(img, rect, size=(5,5), mtype=1):
	dst = img.copy() # 원본 영상 복사해서 사용

	# 원하는 값들 뽑아내기
	start_x, start_y, end_x, end_y = rect
	block_w, block_h = size

	roi     = dst[start_y:end_y, start_x:end_x]
	h, w    = roi.shape[:2]
	img_len = len(img.shape)  # shape 속성이 길이

	# 영역 사이즈를 블록 단위로 자를 수 있도록 크기 맞추기
	h_trim = h - (h % block_h)
	w_trim = w - (w % block_w)
	roi    = roi[:h_trim, :w_trim]

	# 블록 개수
	n_blocks_y = h_trim // block_h
	n_blocks_x = w_trim // block_w

	# 컬러 영상의 경우
	if img_len == 3:
		roi_blocks = roi.reshape(n_blocks_y, block_h, n_blocks_x, block_w, 3) # 블록화
		if mtype == 1: # 1번: 평균 사용
			# 각 블록 단위로 평균 구하기
			block_means = roi_blocks.mean(axis=(1, 3), keepdims=True)

			# 구한 평균값을 채우고 원래 차원으로 reshape
			roi_mosaic  = np.broadcast_to(block_means, roi_blocks.shape).reshape(h_trim, w_trim, 3)

		elif mtype == 2: # 2번: 최대값 사용
			# 각 블록 단위로 최대값 구하기
			block_maxs = roi_blocks.max(axis=(1, 3), keepdims=True)

			# 구한 최대값을 채우고 원래 차원으로 reshape
			roi_mosaic = np.broadcast_to(block_maxs, roi_blocks.shape).reshape(h_trim, w_trim, 3)

		elif mtype == 3: # 3번: 최소값 사용
			# 각 블록 단위로 최소값 구하기
			block_mins = roi_blocks.min(axis=(1, 3), keepdims=True)

			# 구한 최소값을 채우고 원래 차원으로 reshape
			roi_mosaic = np.broadcast_to(block_mins, roi_blocks.shape).reshape(h_trim, w_trim, 3)

		elif mtype == 4: # 4번: 임의 위치 사용
			# 블록 개수
			nby, nbx = h_trim // block_h, w_trim // block_w

			# 블록 단위로 reshape → shape: (nby, block_h, nbx, block_w, 3)
			roi_blocks = roi.reshape(nby, block_h, nbx, block_w, 3)

			# (nby, nbx, block_h, block_w, 3) 로 transpose → 블록 단위 쉽게 다루기
			blocks = roi_blocks.transpose(0, 2, 1, 3, 4)

			# 각 블록마다 랜덤한 y, x 좌표 생성
			rand_y = np.random.randint(0, block_h, size=(nby, nbx))
			rand_x = np.random.randint(0, block_w, size=(nby, nbx))

			# 인덱스 메쉬 생성
			y_idx, x_idx = np.meshgrid(np.arange(nby), np.arange(nbx), indexing='ij')

			# 각 블록에서 무작위로 뽑은 픽셀 → shape: (nby, nbx, 3)
			random_pixels = blocks[y_idx, x_idx, rand_y, rand_x]

			# 이 픽셀들을 각 블록에 반복해서 채움
			filled_blocks = np.broadcast_to(random_pixels[:, :, None, None, :], blocks.shape)

			# 다시 원래 차원으로 reshape
			roi_mosaic = filled_blocks.transpose(0, 2, 1, 3, 4).reshape(h_trim, w_trim, 3)

	# 그레이스케일 영상의 경우
	elif img_len == 2:
		roi_blocks = roi.reshape(n_blocks_y, block_h, n_blocks_x, block_w)
		if mtype == 1:
			block_means = roi_blocks.mean(axis=(1, 3), keepdims=True)
			roi_mosaic  = np.broadcast_to(block_means, roi_blocks.shape).reshape(h_trim, w_trim)
		elif mtype == 2:  # 2번: 최대값 사용
			block_maxs = roi_blocks.max(axis=(1, 3), keepdims=True)
			roi_mosaic = np.broadcast_to(block_maxs, roi_blocks.shape).reshape(h_trim, w_trim)

		elif mtype == 3:  # 3번: 최소값 사용
			block_mins = roi_blocks.min(axis=(1, 3), keepdims=True)
			roi_mosaic = np.broadcast_to(block_mins, roi_blocks.shape).reshape(h_trim, w_trim)

		elif mtype == 4:  # 4번: 임의 위치 사용
			nby, nbx   = h_trim // block_h, w_trim // block_w
			roi_blocks = roi.reshape(nby, block_h, nbx, block_w)
			blocks     = roi_blocks.transpose(0, 2, 1, 3, 4)
			rand_y     = np.random.randint(0, block_h, size=(nby, nbx))
			rand_x     = np.random.randint(0, block_w, size=(nby, nbx))

			y_idx, x_idx  = np.meshgrid(np.arange(nby), np.arange(nbx), indexing='ij')
			random_pixels = blocks[y_idx, x_idx, rand_y, rand_x]
			filled_blocks = np.broadcast_to(random_pixels[:, :, None, None, :], blocks.shape)
			roi_mosaic    = filled_blocks.transpose(0, 2, 1, 3, 4).reshape(h_trim, w_trim)

		else:
			print("옵션 값이 범위 내에 없음")
	else:
		print("영상 값 오류")
		return dst

	# 모자이크 된 부분 영상을 원래 위치에 붙이기
	dst[start_y:start_y + h_trim, start_x:start_x + w_trim] = roi_mosaic
	return dst

if __name__ == '__main__' :
	# 명령행 인자 처리
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--image", required = True,
			help = "Path to the input image")
	ap.add_argument("-s", "--start_point", type = int,
 			nargs='+', default=[0, 0],
			help = "Start point of the rectangle")
	ap.add_argument("-e", "--end_point", type = int,
	 		nargs='+', default=[150, 100],
			help = "End point of the rectangle")
	ap.add_argument("-z", "--size", type = int,
	 		nargs='+', default=[15, 15],
			help = "Mosaic Size")
	ap.add_argument("-t", "--type", type = int,
	 		default=1,
			help = "Mosaic Type")
	args = vars(ap.parse_args())

	filename = args["image"]
	sp       = args["start_point"]
	ep       = args["end_point"]
	size     = args["size"]
	mtype    = args["type"]

	# OpenCV를 사용하여 영상 데이터 로딩
	image = cv2.imread(filename, cv2.IMREAD_COLOR)
	if(image is None):
		raise IOError("Cannot open the image")

	(rows, cols, _) = image.shape
	if(sp[0] < 0 or sp[1] < 0 or ep[0] > rows or ep[1] > cols):
		raise ValueError('Invalid Size')

	# list 연결
	rect = sp + ep

	# 모자이크 영상 생성
	result = mosaic_image(image, rect, size, mtype)

	# 영상 출력을 윈도우 생성
	cv2.namedWindow('image', cv2.WINDOW_NORMAL)
	# 윈도우에 영상 출력
	cv2.imshow('image', result)

	# 사용자 입력 대기
	cv2.waitKey(0)
	# 윈도우 파괴
	cv2.destroyAllWindows()