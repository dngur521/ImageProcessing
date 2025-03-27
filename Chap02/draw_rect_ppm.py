# 필요한 패키지를 import함
import array
import argparse
import PPM.PPM_P6 as ppm
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input",  required = True,
		        help = "Path to the input image")
ap.add_argument("-o", "--output", required = True,
		        help = "Path to the output image")
ap.add_argument("-l", "--location",
                type = int, nargs='+', default=[0,0],
		        help = "Location of square")
ap.add_argument("-s", "--size",
                type = int, nargs='+', default=[50, 50],
		        help = "Size of the square")
ap.add_argument("-c", "--color",
                type = int, nargs='+', default=[255, 0, 0],
		        help = "Color of each pixel in the square")
args = vars(ap.parse_args())

infile   = args["input"]
outfile  = args["output"]
location = args["location"]
size     = args["size"]
color    = args["color"]

# PPM_P6 객체 생성
ppm_p6 = ppm.PPM_P6()
# PPM_P6 객체를 사용하여 PPM 파일 읽기
(width, height, maxval, bitmap) = ppm_p6.read(infile)

# Bytes 형을 파이썬 array 형으로 변환
image = array.array('B', bitmap)
# 파이썬 array 형을 Numpy array 형으로 변환
image = np.array(image)

# 원하는 값들 추출하기
X = location[1]
Y = location[0]
xsize = size[1]
ysize = size[0]

print(f'Square\'s location: ({Y}, {X}), size: {ysize} * {xsize}, color: {color}')

# Numpy array의 차원을 1차원에서 3차원으로 변경
image = image.reshape((height, width, 3))
# R = color[0]
# G = color[1]
# B = color[2]
# 원하는 곳에 사각형 그리기(array slicing 기능 사용)
image[X:X+xsize, Y:Y+ysize] = color
# image[X:Y, X+xsize:Y+ysize] 가 아님!!

# Numpy array의 차원을 3차원에서 1차원으로 변경
image = image.reshape(height * width * 3)
# numpy.array를 bytes 데이터로 변환
image = bytes(image)

# PPM_P6 객체를 사용하여 PPM 파일 저장
ppm_p6.write(width, height, maxval, image, outfile)