import cv2
import numpy as np
import matplotlib.pyplot as plt
from copy import copy, deepcopy

bgr_img = cv2.imread('niemeyer.png')
gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
cv2.imwrite('san_francisco_grayscale.jpg',gray_img)
img_output = np.zeros((gray_img.shape[0]+2,gray_img.shape[1]+2,3), np.uint8)


for i in range(0,img_output.shape[0]):
    for j in range(0,img_output.shape[1]):
        if i == 0:
            if j == 0:
                img_output[i][j] = 0
            elif j == img_output.shape[1]-1:
                img_output[i][j] = 0
            else:
                img_output[i][j] = 0
        elif i == img_output.shape[0]-1:
            if j == 0:
                img_output[i][j] = 0
            elif j == img_output.shape[1]-1:
                img_output[i][j] = 0
            else:
                img_output[i][j] = 0
        else:
            if j == 0:
                img_output[i][j] = 0
            elif j == img_output.shape[1]-1:
                img_output[i][j] = 0
            else:
                img_output[i][j] = gray_img[i-1][j-1]

mask =[[1.0/9.0,1.0/9.0,1.0/9.0],[1.0/9.0,1.0/9.0,1.0/9.0],[1.0/9.0,1.0/9.0,1.0/9.0]]

def mask_covolucao(i,j,mask):
    len_mask = len(mask)/2
    i = i-len_mask
    j = j-len_mask
    for x in range(0,len(mask[0])):
        for y in range(0,len(mask)):
            aux = mask[x][y]*gray_img[i][j]

    return aux
i = 0
j = 0
while i < gray_img.shape[0]:
    while j < gray_img.shape[1]:
        if i == 0:
            if j == 0:
                img_output[i+1][j+1] = mask[1][1]*gray_img[i][j]+mask[1][2]*gray_img[i][j+1]+mask[2][1]*gray_img[i+1][j]+mask[2][2]*gray_img[i+1][j+1]
            elif j == gray_img.shape[1]-1:
                img_output[i+1][j+1] = mask[1][1]*gray_img[i][j]+mask[2][1]*gray_img[i+1][j]+mask[1][0]*gray_img[i][j-1]+mask[2][0]*gray_img[i+1][j-1]
            else:
                img_output[i+1][j+1] = mask[1][1]*gray_img[i][j]+mask[1][2]*gray_img[i][j+1]+mask[2][1]*gray_img[i+1][j]+mask[2][2]*gray_img[i+1][j+1]
                img_output[i+1][j+1] = img_output[i+1][j+1]+mask[1][0]*gray_img[i][j-1]+mask[2][0]*gray_img[i+1][j-1]
                
        elif i == gray_img.shape[0]-1:
            if j == 0:
                img_output[i+1][j+1] = mask[1][1]*gray_img[i][j]+mask[0][1]*gray_img[i-1][j]+mask[1][2]*gray_img[i][j+1]+mask[0][2]*gray_img[i-1][j+1]
            elif j == gray_img.shape[1]-1:
                img_output[i+1][j+1] = mask[1][1]*gray_img[i][j]+mask[1][0]*gray_img[i][j-1]+mask[0][1]*gray_img[i-1][j]+mask[0][0]*gray_img[i-1][j-1]
            else:
                img_output[i+1][j+1] = mask[1][1]*gray_img[i][j]+mask[0][1]*gray_img[i-1][j]+mask[1][2]*gray_img[i][j+1]+mask[0][2]*gray_img[i-1][j+1]
                img_output[i+1][j+1] = img_output[i+1][j+1]+mask[1][0]*gray_img[i][j-1]+mask[0][0]*gray_img[i-1][j-1]
                
        else:
            if j == 0:
                img_output[i+1][j+1] = mask[1][1]*gray_img[i][j]+mask[1][2]*gray_img[i][j+1]+mask[2][1]*gray_img[i+1][j]+mask[2][2]*gray_img[i+1][j+1]
                img_output[i+1][j+1] = img_output[i+1][j+1]+mask[0][1]*gray_img[i-1][j]+mask[0][2]*gray_img[i-1][j+1]
            elif j == gray_img.shape[1]-1:
                img_output[i+1][j+1] = mask[1][1]*gray_img[i][j]+mask[2][1]*gray_img[i+1][j]+mask[1][0]*gray_img[i][j-1]+mask[2][0]*gray_img[i+1][j-1]
                img_output[i+1][j+1] = img_output[i+1][j+1]+mask[0][1]*gray_img[i-1][j]+mask[0][0]*gray_img[i-1][j-1]
            else:
                img_output[i+1][j+1] = mask_covolucao(i,j,mask)
        j = j+1
    i = i+1



plt.imshow(img_output, cmap = plt.get_cmap('gray'))
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()

while True:
    k = cv2.waitKey(0) & 0xFF    # 0xFF? To get the lowest byte.
    if k == 27: break            # Code for the ESC key

cv2.destroyAllWindows()
