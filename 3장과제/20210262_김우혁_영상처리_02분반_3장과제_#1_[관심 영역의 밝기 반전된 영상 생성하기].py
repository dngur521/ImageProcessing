# 필요한 패키지를 import함
from __future__ import print_function
import argparse
import cv2
import numpy as np
from handle_channel_roi import display_image

def invert_image_m1(img, rect):
	dst = img.copy() # 원본 영상의 복제본을 만들어 사용

	# 원하는 값들 뽑아내기
	start_x = rect[0]
	start_y = rect[1]
	end_x   = rect[2]
	end_y   = rect[3]

	# 그레이스케일 영상은 길이가 2, 컬러 영상은 3
	img_len = len(image.shape) # shape 속성이 길이

	# 컬러 영상인 경우 채널 수가 3인 것을 확인
	if img_len == 3:
		for x in range(start_x, end_x):
			for y in range(start_y, end_y):
				dst[y, x, 2] = 255 - dst[y, x, 2] # Red   채널
				dst[y, x, 1] = 255 - dst[y, x, 1] # Green 채널
				dst[y, x, 0] = 255 - dst[y, x, 0] # Blue  채널
		return dst

	elif img_len == 2: # 그레이스케일 영상의 경우
		for x in range(start_x, end_x):
			for y in range(start_y, end_y):
				dst[y, x] = 255 - dst[y, x]
		return dst

	else:
		print("영상 값 오류")
		return dst

def invert_image_m2(img, rect):
	dst = img.copy() # 원본 영상의 복제본을 만들어 사용
	# 원하는 값들 뽑아내기
	start_x = rect[0]
	start_y = rect[1]
	end_x   = rect[2]
	end_y   = rect[3]

	# Numpy의 슬라이싱 연산과 브로드캐스팅 기법을 사용하여 밝기 반전 연산을 수행
	dst[start_y:end_y, start_x:end_x] = 255 - dst[start_y:end_y, start_x:end_x]

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
	args = vars(ap.parse_args())

	filename = args["image"]
	sp = args["start_point"]
	ep = args["end_point"]

	# OpenCV를 사용하여 영상 데이터 로딩
	image = cv2.imread(filename)
	(rows, cols, _) = image.shape
	if sp[0] < 0 or sp[1] < 0 or ep[0] > rows or ep[1] > cols:
		raise ValueError('Invalid Size')

	# list 연결
	rect = sp + ep

	e1 = cv2.getTickCount()
	inverted = invert_image_m1(image, rect)
	e2 = cv2.getTickCount()
	time = (e2 - e1)/ cv2.getTickFrequency()
	print('[정보]방법 1 소요시간: {}'.format(time))

	e1 = cv2.getTickCount()
	inverted = invert_image_m2(image, rect)
	e2 = cv2.getTickCount()
	time = (e2 - e1)/ cv2.getTickFrequency()
	print('[정보]방법 2 소요시간: {}'.format(time))

	display_image(inverted, 'inverted')