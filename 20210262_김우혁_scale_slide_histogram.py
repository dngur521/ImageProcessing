from __future__ import print_function
import argparse
import cv2
import numpy as np
from matplotlib import pyplot as plt

def scaleHistogram(img, min_val, max_val):
    # 컬러 영상 처리: 각 채널별로 normalize
    channels = cv2.split(img)
    scaled_channels = [cv2.normalize(ch, None, min_val, max_val, cv2.NORM_MINMAX) for ch in channels]
    return cv2.merge(scaled_channels)

def slideHistogram(img, val):
    # 각 채널에 대해 슬라이딩 수행
    channels = cv2.split(img)
    slided_channels = []
    for ch in channels:
        hist = cv2.calcHist([ch], [0], None, [256], [0, 256]).flatten()
        slidehist = np.roll(hist, val)
        slidehist[:val] = 0

        # 255가 넘어가는 값들에 대한 처리
        if val > 0:
            slidehist[255] += np.sum(slidehist[256:])
            slidehist = slidehist[:256]
        else:
            slidehist = slidehist[:256]

        # 슬라이딩된 히스토그램을 바탕으로 다시 이미지 매핑 (간단하게 LUT 적용)
        lut = np.zeros(256, dtype=np.uint8)
        for i in range(256):
            lut[i] = min(255, i + val) if i + val < 256 else 255
        slided_channel = cv2.LUT(ch, lut)
        slided_channels.append(slided_channel)
        
    return cv2.merge(slided_channels)

if __name__ == "__main__":
    # 명령행 인자 처리
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--image', required=True,
                    help='Path to the input image')
    ap.add_argument('-r', '--range', nargs=2, type=int,
                    default=[50, 150],
                    help='스케일링의 범위 (기본값: [50, 150])')
    ap.add_argument('-s', '--slide', type=int,
                    default=50,
                    help='슬라이딩의 값 (기본값: 50)')
    args = vars(ap.parse_args())

    filename    = args['image']
    scale_range = args['range']
    slide_value = args['slide']
    min_val     = scale_range[0]
    max_val     = scale_range[1]

    image = cv2.imread(filename)

    # 결과 영상 생성
    scaled_image = scaleHistogram(image, min_val, max_val)
    slided_image = slideHistogram(scaled_image, slide_value)

    # 히스토그램 계산
    color = ('b', 'g', 'r')
    orig_hist  = [cv2.calcHist([image],        [i], None, [256], [0, 256]) for i in range(3)]
    scale_hist = [cv2.calcHist([scaled_image], [i], None, [256], [0, 256]) for i in range(3)]
    slide_hist = [cv2.calcHist([slided_image], [i], None, [256], [0, 256]) for i in range(3)]

    # 이미지 및 히스토그램 출력
    plt.figure(figsize=(15, 10))

    # 원본 이미지
    plt.subplot(3, 2, 1)
    plt.title('Original Image')
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    # 원본 히스토그램
    plt.subplot(3, 2, 2)
    plt.title('Original Histogram')
    for i, col in enumerate(color):
        plt.plot(orig_hist[i], color=col)
    plt.xlim([0, 256])

    # 스케일링 이미지
    plt.subplot(3, 2, 3)
    plt.title('Scaled Image')
    plt.imshow(cv2.cvtColor(scaled_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    # 스케일링 히스토그램
    plt.subplot(3, 2, 4)
    plt.title('Scaled Histogram')
    for i, col in enumerate(color):
        plt.plot(scale_hist[i], color=col)
    plt.xlim([0, 256])

    # 슬라이딩 이미지
    plt.subplot(3, 2, 5)
    plt.title('Slided Image')
    plt.imshow(cv2.cvtColor(slided_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    # 슬라이딩 히스토그램
    plt.subplot(3, 2, 6)
    plt.title('Slided Histogram')
    for i, col in enumerate(color):
        plt.plot(slide_hist[i], color=col)
    plt.xlim([0, 256])

    plt.tight_layout()
    plt.show()
