import cv2
import numpy as np
import matplotlib.pyplot as plt

bgr_img = cv2.imread('lena.jpg')
gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
#cv2.imwrite('san_francisco_grayscale.jpg',gray_img)
number = 0
for y in range(0, gray_img.shape[0],10): #percorrelinhas
    if number == 0:
        number = 255
    elif number == 255:
        number = 0
    for x in range(0, gray_img.shape[1],10):
        gray_img[y:y+10, x:x+10] = number
        if number == 0:
            number = 255
        elif number == 255:
            number = 0

plt.imshow(gray_img, cmap = plt.get_cmap('gray'))
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()

while True:
    k = cv2.waitKey(0) & 0xFF    # 0xFF? To get the lowest byte.
    if k == 27: break            # Code for the ESC key

cv2.destroyAllWindows()
