import cv2
import numpy as np
import matplotlib.pyplot as plt

img1 = cv2.imread('left.png',0)
img2 = cv2.imread('right.png',0)

row1,col1 =img1.shape[:2]
row2,col2 =img2.shape[:2]
print (row1,col1,row2,col2)
blur1 = cv2.GaussianBlur(img1,(9,9),0)
blur2 = cv2.GaussianBlur(img2,(9,9),0)
cv2.imwrite("blur_1.png", blur1)
cv2.imwrite("blur_2.png", blur2)

f1 = np.fft.fft2(img1)
fshift1 = np.fft.fftshift(f1)
magnitude_spectrum1 = 20*np.log(np.abs(fshift1))
print(magnitude_spectrum1.shape)

f2 = np.fft.fft2(img2)
fshift2 = np.fft.fftshift(f2)
magnitude_spectrum2 = 20*np.log(np.abs(fshift2))

sigma_2 = np.zeros((row1,col1))
for i in range(row1):
    for j in range(col1):
        if magnitude_spectrum2[i,j] != 0:
            a=np.log(magnitude_spectrum1[i,j]/magnitude_spectrum2[i,j])
            b=1/((i+1)*(i+1)+(j+1)*(j+1))
            sigma_2[i,j] = b*a
            if sigma_2[i,j]<0:
                sigma_2[i,j]=0
                      
sigma = np.sqrt(sigma_2)
mx = np.max(sigma)
mi = np.min(sigma)
sigma = (sigma-mi)/(mx-mi)
sigma*=255
cv2.imwrite("Relative_blur_map.png", sigma)    

plt.subplot(231),plt.imshow(img1, cmap = 'gray')
plt.title('Input Image1'), plt.xticks([]), plt.yticks([])
plt.subplot(232),plt.imshow(blur1, cmap = 'gray')
plt.title('Blur Image1'), plt.xticks([]), plt.yticks([])
plt.subplot(233),plt.imshow(magnitude_spectrum1, cmap = 'gray')
plt.title('Magnitude Spectrum1'), plt.xticks([]), plt.yticks([])
plt.subplot(234),plt.imshow(img2, cmap = 'gray')
plt.title('Input Image2'), plt.xticks([]), plt.yticks([])
plt.subplot(235),plt.imshow(blur2,cmap = 'gray')
plt.title('Blur Image2'), plt.xticks([]), plt.yticks([])
plt.subplot(236),plt.imshow(magnitude_spectrum2, cmap = 'gray')
plt.title('Magnitude Spectrum2'), plt.xticks([]), plt.yticks([])
plt.show()



