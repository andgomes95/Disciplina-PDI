import cv2
import numpy as np
import matplotlib.pyplot as plt

bgr_img = cv2.imread('lena.jpg')
hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
cv2.imwrite('equalize.jpg',hsv_img)
h, s, v = cv2.split(hsv_img)
histograma_s = range(256)
histograma_v = range(256)
cores = range(256)
for i in range(0, 256,1):
    histograma_s[i] = 0
    histograma_v[i] = 0
    cores[i] = float(cores[i])/256.0
i = 0
total = 0

for i in range(0, hsv_img.shape[0],1):
    for j in range(0, hsv_img.shape[1],1):
        total = total +1
        histograma_s[s[i][j]] = histograma_s[s[i][j]]+1
        histograma_v[v[i][j]] = histograma_v[v[i][j]]+1
histogramaProb_s = histograma_s
histogramaProb_v = histograma_v
for i in range(0, 256,1):
    if i == 0:
        histogramaProb_s[i] = float(histograma_s[i])/float(total)
        histogramaProb_v[i] = float(histograma_v[i])/float(total)
    else:
        histogramaProb_s[i] = histogramaProb_s[i-1] + float(histograma_s[i])/float(total)
        histogramaProb_v[i] = histogramaProb_v[i-1] + float(histograma_v[i])/float(total)
j = 0
i = 0
while (j < 256):
    if cores[j] > histogramaProb_s[i]:
        histogramaProb_s[i] = j
        i=i+1
    elif j == 255 and i < 255:
        histogramaProb_s[i] = j
        i=i+1
    else:
        histogramaProb_s[255] = 255
        j = j+1   
i = 0
j = 0
while (j < 256):
    if cores[j] > histogramaProb_v[i]:
        histogramaProb_v[i] = j
        i=i+1
    elif j == 255 and i < 255:
        histogramaProb_v[i] = j
        i=i+1
    else:
        histogramaProb_v[255] = 255
        j = j+1    

for i in range(0, hsv_img.shape[0],1):
    for j in range(0, hsv_img.shape[1],1):
        x = s[i][j]
        y = v[i][j]
        s[i][j] = histogramaProb_s[x]
        v[i][j] = histogramaProb_v[y]

hsv_img = cv2.merge([h,s,v])

#plt.plot(histograma)
bgr_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)
plt.imshow(bgr_img, cmap = plt.get_cmap('gray'))
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()

while True:
    k = cv2.waitKey(0) & 0xFF    # 0xFF? To get the lowest byte.
    if k == 27: break            # Code for the ESC key

cv2.destroyAllWindows()
