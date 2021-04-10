#importing the required modules
import numpy as np
import cv2
from matplotlib import pyplot as plt

#original image
img = cv2.imread("shoe.jpeg", cv2.IMREAD_GRAYSCALE)

#laplacian gradient 
lap=cv2.Laplacian(img, cv2.CV_64F,ksize=3)
lap=np.uint8(np.absolute(lap))

#sobelx and sobely gradient
sobelx=cv2.Sobel(img,cv2.CV_64F,1,0)
sobely=cv2.Sobel(img,cv2.CV_64F,0,1)
sobelx=np.uint8(np.absolute(sobelx))
sobely=np.uint8(np.absolute(sobely))

#sobel combined gradient
sobelcombined=cv2.bitwise_or(sobelx, sobely)

#canny edge
edges = cv2.Canny(img,100,200)


titles=['image','Laplacian','SobelX','SobelY','SobelCombined','Canny']
images=[img,lap,sobelx,sobely,sobelcombined,edges]

#displaying the images
for i in range(6):
    plt.subplot(2, 3, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()