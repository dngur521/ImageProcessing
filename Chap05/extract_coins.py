# 필요한 패키지를 import함
from __future__ import print_function
from random import seed, randint
import argparse
import cv2
import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt

def findContours(img):
	# 타원형의 구조적 요소 생성
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

	# 닫힘 연산 적용
	img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

	# contour 생성
	version = int(cv2.__version__.split(".")[0])
	if version == 2 or version == 4:
		(contours, hierarchy) \
		= cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	elif version == 3:
		(_, contours, hierarchy) \
		 = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	print("Total number of contours: ", len(contours))

	return contours

if __name__ == "__main__" :
	# 명령행 인자 처리
	ap = argparse.ArgumentParser()
	ap.add_argument('-i', '--image', required = True, \
			help = 'Path to the input image')
	args = vars(ap.parse_args())

	filename = args['image']

	# OpenCV를 사용하여 영상 데이터 로딩
	image = cv2.imread(filename)

	# Grayscale 영상으로 변환한 후
	# 가우시안 평활화 및 임계화 수행
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (3, 3), 0)
	binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

	# 연결요소 생성
	contours = findContours(binary)

	# 모든 연결요소의 외곽선을 포함하는 영상 생성
	new_img = np.zeros_like(image, dtype="uint8")
	cntrarray = sorted(contours, key = cv2.contourArea, reverse = True)
	box_img = deepcopy(new_img)  # 필요하다면 반복 전에 복제

	seed(9001)
	for cntr in cntrarray:
		if cv2.contourArea(cntr) > 5:
			r = randint(0, 256)
			g = randint(0, 256)
			b = randint(0, 256)

			# 외곽선: 무작위 색
			cv2.drawContours(new_img, [cntr], 0, (b, g, r), 10)
			# 내부 채우기: 흰색
			hull = cv2.convexHull(cntr)
			cv2.drawContours(new_img, [hull], 0, (255, 255, 255), -1)
			print("면적: ", cv2.contourArea(cntr))

	# 결과 영상 출력
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	plt.subplot(1, 3, 1), plt.imshow(image)
	plt.title('image'), plt.xticks([]), plt.yticks([])
	plt.subplot(1, 3, 2), plt.imshow(binary, cmap='gray')
	plt.title('threshold'), plt.xticks([]), plt.yticks([])
	plt.subplot(1, 3, 3), plt.imshow(new_img)
	plt.title('contour'), plt.xticks([]), plt.yticks([])
	plt.show()