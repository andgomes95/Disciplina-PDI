import cv2
import numpy as np
import matplotlib.pyplot as plt

bgr_img = cv2.imread('lena.jpg')
gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
cv2.imwrite('san_francisco_grayscale.jpg',gray_img)
histograma = range(256)
cores = range(256)
for i in range(0, 256,1):
    histograma[i] = 0
    cores[i] = float(cores[i])/256.0
i = 0
total = 0

for i in range(0, gray_img.shape[0],1):
    for j in range(0, gray_img.shape[1],1):
        total = total +1
        histograma[gray_img[i][j]] = histograma[gray_img[i][j]]+1
histogramaProb = histograma
for i in range(0, 256,1):
    if i == 0:
        histogramaProb[i] = float(histograma[i])/float(total)
    else:
        histogramaProb[i] = histogramaProb[i-1] + float(histograma[i])/float(total)
j = 0
i = 0
while (j < 256):
    if cores[j] > histogramaProb[i]:
        histogramaProb[i] = j
        i=i+1
    elif j == 255 and i < 255:
        histogramaProb[i] = j
        i=i+1
    else:
        histogramaProb[255] = 255
        j = j+1    

for i in range(0, gray_img.shape[0],1):
    for j in range(0, gray_img.shape[1],1):
        x = gray_img[i][j]
        gray_img[i][j] = histogramaProb[x]

#plt.plot(histograma)
plt.imshow(gray_img, cmap = plt.get_cmap('gray'))
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()

while True:
    k = cv2.waitKey(0) & 0xFF    # 0xFF? To get the lowest byte.
    if k == 27: break            # Code for the ESC key

cv2.destroyAllWindows()
