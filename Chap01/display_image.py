# 필요한 패키지를 import 함
from __future__ import print_function
import argparse
import cv2

def display_image(img):
    # 영상 정보 출력
    print("(rows, cols, ch): {}".format(img.shape)) #cf: size, dtype

    # 영상 출력을 위한 윈도우 생성
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    # cv2.WINDOW_AUTOSIZE: 원본 이미지 크기로 고정하여 윈도우 생성
    # cv2.WINDOW_NORMAL: 원본 이미지 크기로 윈도우를 생성하여 이미지를 나타내지만 사용자가 크기를 조절할 수 있는 윈도우 생성

    # 윈도우에 영상 출력
    cv2.imshow('image', img)

    # 사용자 입력 대기 (인자로 대기시간 설정, 0은 무한대), 아무 키나 입력하면 아래 코드 실행(윈도우 파괴)
    cv2.waitKey(0)
    # 윈도우 파괴
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # 명령행 인자 처리
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Path to the input image")
    args = vars(ap.parse_args())

    filename = args["image"]

    # OpenCV를 사용하여 영상 데이터 로딩
    image = cv2.imread(filename, cv2.IMREAD_COLOR)
    if image is None:
        print('{}: reading error'.format(filename))
    else:
        # 윈도우에 영상 출력
        display_image(image)

# 사용 예: python display_image.py -- image ../images/nature.jpg
# 사용 예: python display_image.py -i ../images/nature.jpg