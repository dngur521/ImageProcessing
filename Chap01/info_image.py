# 필요한 패키지를 import 함
from __future__ import print_function
import argparse
import cv2
import display_image

def info_image(image):
    print(image.shape)
    print(image.size)
    print(image.dtype)

if __name__ == "__main__":
    image = cv2.imread("nature.jpg")
    if image is None:
        print('{}: reading error'.format("nature.jpg"))
    else:
        # 영상 블러링 및 캐니 에지 연산 수행
        info_image(image)

    pk = cv2.waitKey(0) & 0xFF # in 64bit machine, 하위 바이트 하나의 값만 불러오겠다 (== 아스키 코드 값을 확인하겠다)
    if pk == ord('s'): # chr(99)는 문자 'c' 반환, 입력된 문자가 c면 아래 코드 수행(파일 저장)
        cv2.imwrite('copy.jpg', image)