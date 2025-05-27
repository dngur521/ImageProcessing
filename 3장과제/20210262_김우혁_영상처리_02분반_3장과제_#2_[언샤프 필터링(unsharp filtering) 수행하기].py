# 필요한 패키지를 import함
from __future__ import print_function
import argparse
import cv2
import numpy as np
from handle_channel_roi import display_image

def unsharp_image(img, alpha=1.1, beta=0.5):
	dst = img.copy() # 원본 영상의 복제본 만들기

	# 원본 영상에 대한 가우시안 필터링 수행
	ksize = 3 # 마스크 크기 또는 표준편차는 직접 설정
	gaussian = cv2.GaussianBlur(dst, (ksize, ksize), 0)

	# 원본 영상과 가우시안 필터링 수행한 영상을 적절한 비율로 더하기
	dst = cv2.addWeighted(img, alpha, gaussian, -beta, 0)

	# 결과 영상 반환
	return dst

if __name__ == '__main__':
	# 명령행 인자 처리
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--input", required = True,
		help = "Path to the input image")
	ap.add_argument("-o", "--output", required = True,
		help = "Path to the output image")
	args = vars(ap.parse_args())

	infile  = args["input"]
	outfile = args["output"]

	# OpenCV를 사용하여 영상 데이터 로딩
	image = cv2.imread(infile, cv2.IMREAD_UNCHANGED)

	filtered = unsharp_image(image, 1.4, 0.5)
	display_image(filtered, 'filtering')

	print('Saved to {}'.format(outfile))
	cv2.imwrite(outfile, filtered)