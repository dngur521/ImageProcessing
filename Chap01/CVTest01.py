import cv2

def load_image():
    image = cv2.imread("Lenna.png", -1)
    # 1: 컬러 영상(cv2.IMREAD_COLOR) 0: 회색음영 영상(cv2.IMREAD_GRAYSCALE) -1: 원래의 영상(cv2.IMREAD_UNCHANGED)
    if image is None:
        print("Can't open the image")
        return

    cv2.namedWindow('window', cv2.WINDOW_NORMAL)
    cv2.imshow('window', image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    load_image()