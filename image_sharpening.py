import cv2
import numpy as np
import os

# pic = cv2.imread('E:\\NCHU\computer_version\lenna_512.jpg')
# print(pic.shape)

pic_gray = cv2.imread("/d/computer_vision/hw_image_sharpening/lenna_512.jpg", cv2.IMREAD_GRAYSCALE)
cv2.imwrite("/d/computer_vision/hw_image_sharpening/lenna_512_gray.jpg", pic_gray)
print(pic_gray)

def sobel_gray(image):
    edge = np.zeros(image.shape)
    for y in range(image.shape[0]-2):
        for x in range(image.shape[1]-2):

            subimg = image[y:y+3, x:x+3]
            z1 = subimg[0,0]
            z2 = subimg[0,1]
            z3 = subimg[0,2]
            z4 = subimg[1,0]
            z6 = subimg[1,2]
            z7 = subimg[2,0]
            z8 = subimg[2,1]
            z9 = subimg[2,2]

            sobel = abs((z7 + 2*z8 + z9) - (z1 + 2*z2 +z3)) + abs((z3 + 2*z6 + z9) - (z1 + 2*z4 + z7))
            edge[y+1,x+1] = sobel / 255
            # edge = cv2.normalize(edge, None, 0, 1, cv2.NORM_MINMAX)
            # edge = cv2.normalize(edge,None,0,255,cv2.NORM_MINMAX)
            # edge = edge.astype(np.uint8)
    return edge

def average(image):
    avg = np.zeros(image.shape)
    for y in range(image.shape[0]-2):
        for x in range(image.shape[1]-2):

            subimg = image[y:y+3, x:x+3]
            z1 = subimg[0,0]
            z2 = subimg[0,1]
            z3 = subimg[0,2]
            z4 = subimg[1,0]
            z5 = subimg[1,1]
            z6 = subimg[1,2]
            z7 = subimg[2,0]
            z8 = subimg[2,1]
            z9 = subimg[2,2]

            avg[y+1,x+1] = (z1 + z2 + z3 + z4 + z5 + z6 + z7 + z8 + z9) / 9
    return avg

def isotropic(image):
    temp = np.zeros(image.shape)
    for y in range(image.shape[0]-2):
        for x in range(image.shape[1]-2):

            subimg = image[y:y+3, x:x+3]
            z1 = subimg[0,0]
            z2 = subimg[0,1]
            z3 = subimg[0,2]
            z4 = subimg[1,0]
            z5 = subimg[1,1]
            z6 = subimg[1,2]
            z7 = subimg[2,0]
            z8 = subimg[2,1]
            z9 = subimg[2,2]

            temp[y+1,x+1] = 8*z5 - z1 - z2 - z3 - z4 - z6 - z7 - z8 - z9
    # cv2.imshow("temp",temp)
    temp[temp > 255] = 255
    temp[temp < 0] = 0

    # laplacian_pic = cv2.normalize(temp,None,0,255,cv2.NORM_MINMAX)        
    # laplacian_pic = laplacian_pic.astype(np.uint8)
    # cv2.imshow('la', laplacian_pic)
    return temp
if __name__ == "__main__":
    s = average(sobel_gray(pic_gray))
    print('s\n',s)
    # s = cv2.normalize(s, None, 0, 1, cv2.NORM_MINMAX)
    
    i = isotropic(pic_gray)
    print('i\n',i)
    print('s*i\n',(s * i))

    sharped = (s * i) + pic_gray
    print('sharped\n',sharped)

    # shaped = np.zeros(pic_gray.shape)
    # for y in range(pic_gray.shape[0]):
    #     for x in range(pic_gray.shape[1]):
    #         shaped[y,x] = s[y,x] * i[y,x] + pic_gray[y,x]
    cv2.imwrite("/d/computer_vision/hw_image_sharpening/sharped_sobel.jpg", sharped)
    # cv2.imshow("shaped", sharped)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
  

