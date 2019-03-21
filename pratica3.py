import cv2
import numpy as np
import matplotlib.pyplot as plt

bgr_img = cv2.imread('niemeyer.png')
gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
cv2.imwrite('san_francisco_grayscale.jpg',gray_img)
histograma = range(256)
for i in range(0, 255,1):
    histograma[i] = 0
i = 0
for i in range(0, gray_img.shape[0],1):
    for j in range(0, gray_img.shape[1],1):
        histograma[gray_img[i][j]] = histograma[gray_img[i][j]]+1

plt.plot(cv2.calcHist([gray_img],[0],None,[256],[0,256]))
#plt.plot(histograma)

#plt.imshow(gray_img, cmap = plt.get_cmap('gray'))
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()

while True:
    k = cv2.waitKey(0) & 0xFF    # 0xFF? To get the lowest byte.
    if k == 27: break            # Code for the ESC key

cv2.destroyAllWindows()
