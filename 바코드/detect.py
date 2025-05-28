from __future__ import print_function
import argparse
import cv2
import os
import glob
import matplotlib.pyplot as plt
import time

start_time = time.perf_counter()

def process_gradient(grad_sub):
    grad_sub_abs = cv2.convertScaleAbs(grad_sub)
    blur = cv2.GaussianBlur(grad_sub_abs, (97, 97), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return grad_sub_abs, blur, thresh

def detectBarcode(img, verbose=False):
    stages = {}

    # 그레이 스케일 변환
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 정규화
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    stages["1. Grayscale"] = gray

    # 수평/수직 경계 강도 계산
    gradX = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gradY = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    stages["2. Gradient X"] = cv2.convertScaleAbs(gradX)
    stages["2. Gradient Y"] = cv2.convertScaleAbs(gradY)

    # Y - X
    grad_sub1, blur1, thresh1 = process_gradient(gradY - gradX)
    # X - Y
    grad_sub2, blur2, thresh2 = process_gradient(gradX - gradY)

    # 수평방향 기준과 수직방향 기준의 threshold 값 중 적절한 값 하나 선택하기 (0이 아닌 값이 많은것 선택)
    count1 = cv2.countNonZero(thresh1)
    count2 = cv2.countNonZero(thresh2)
    if count1 > count2:
        grad_sub, grad_blur, thresh = grad_sub1, blur1, thresh1
        stages["3. Subtracted Gradient"] = grad_sub1
        stages["4. Gaussian Blurred"] = blur1
        stages["5. Threshold"] = thresh1
    else:
        grad_sub, grad_blur, thresh = grad_sub2, blur2, thresh2
        stages["3. Subtracted Gradient"] = grad_sub2
        stages["4. Gaussian Blurred"] = blur2
        stages["5. Threshold"] = thresh2

    # 모폴로지 변환 (닫힘 연산)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    stages["6. Morphological Close"] = closed

    # 침식 & 팽창 연산
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilate = cv2.dilate(closed, kernel, iterations=8)
    stages["7. Dilate"] = dilate
    eroded = cv2.erode(dilate, kernel, iterations=16)
    stages["7. Erode"] = eroded

    # 연결 요소 생성
    contours, _ = cv2.findContours(eroded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # verbose = True일 경우 matplot을 이용해서 수행 과정 화면에 출력
    if verbose:
        if contours:
            x, y, w, h = cv2.boundingRect(contours[0])
            debug_img = img.copy()
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), (255, 0, 0), 3)
            stages["8. Largest Contour"] = cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB)

        titles = list(stages.keys())
        n = len(titles)
        plt.figure(figsize=(18, 4))
        for i, key in enumerate(titles):
            plt.subplot(1, n, i + 1)
            plt.title(key, fontsize=8)
            img_to_show = stages[key]
            if len(img_to_show.shape) == 2:
                plt.imshow(img_to_show, cmap='gray')
            else:
                plt.imshow(img_to_show)
            plt.axis('off')
        plt.tight_layout()
        plt.show()

    # 연결 요소 중 가장 크기가 큰 한개의 연결 요소 반환 (최종 바코드 영역 검출)
    return contours[0] if contours else None

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset",
                    required = True, help = "path to the dataset folder")
    ap.add_argument("-r", "--detectset",
                    required = True, help = "path to the detectset folder")
    ap.add_argument("-f", "--detect",
                    required = True, help = "path to the detect file")
    args = vars(ap.parse_args())

    dataset    = args["dataset"]
    detectset  = args["detectset"]
    detectfile = args["detect"]

    # 결과 영상 저장 폴더 존재 여부 확인
    if(not os.path.isdir(detectset)):
        os.mkdir(detectset)

    # 결과 영상 표시 여부
    verbose = True

    # 검출 결과 위치 저장을 위한 파일 생성
    f = open(detectfile, "wt", encoding="UTF-8")  # UT-8로 인코딩

    # 바코드 영상에 대한 바코드 영역 검출
    for imagePath in glob.glob(dataset + "/*.jpg"):
        print(imagePath, '처리중...')

        image = cv2.imread(imagePath)

        # 바코드 검출
        points = detectBarcode(image, verbose=verbose)

        # 바운딩 박스 계산
        x, y, w, h = cv2.boundingRect(points)

        # 결과 영상 저장하기 위한 바코드 영역 표시
        detectimg = cv2.rectangle(image.copy(), (x, y), (x + w, y + h), (0, 255, 0), 10)

        # 결과 영상 저장
        loc1 = imagePath.rfind("\\")
        loc2 = imagePath.rfind(".")
        fname = 'result/' + imagePath[loc1 + 1: loc2] + '_res.jpg'
        cv2.imwrite(fname, detectimg)

        # 검출한 결과 위치 저장
        f.write(imagePath[loc1 + 1: loc2])
        f.write("\t")
        f.write(str(x))
        f.write("\t")
        f.write(str(y))
        f.write("\t")
        f.write(str(x + w))
        f.write("\t")
        f.write(str(y + h))
        f.write("\n")

    end_time = time.perf_counter()
    print(f"총 수행 시간: {end_time - start_time:.2f}초")