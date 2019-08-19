import numpy as np
import math 
import cv2
from pylab import *
from scipy import linalg
from points import points

#imgleft = cv2.imread('house1.jpg',0)
#imgright = cv2.imread('house2.jpg',0)
imgleft = cv2.imread('library1.jpg',0)
imgright = cv2.imread('library2.jpg',0)

    # normalize image coordinates
#x1,x2 = points()
x1,x2 = points1()

n = x1.shape[1]
x1 = x1 / x1[2]
mean_1 = mean(x1[:2],axis=1)
S1 = sqrt(2) / std(x1[:2])
T1 = array([[S1,0,-S1*mean_1[0]],[0,S1,-S1*mean_1[1]],[0,0,1]])
x1 = dot(T1,x1)
    
x2 = x2 / x2[2]
mean_2 = mean(x2[:2],axis=1)
S2 = sqrt(2) / std(x2[:2])
T2 = array([[S2,0,-S2*mean_2[0]],[0,S2,-S2*mean_2[1]],[0,0,1]])
x2 = dot(T2,x2)

    # compute F with the normalized coordinates

A = zeros((8,9))
for i in range(0,8):
    A[i] = [x1[0,i]*x2[0,i], x1[0,i]*x2[1,i], x1[0,i]*x2[2,i],
            x1[1,i]*x2[0,i], x1[1,i]*x2[1,i], x1[1,i]*x2[2,i],
            x1[2,i]*x2[0,i], x1[2,i]*x2[1,i], x1[2,i]*x2[2,i]]

print(A) 

U,S,V = linalg.svd(A)
F = V[-1].reshape(3,3)
U,S,V = linalg.svd(F)
S[2] = 0
F = dot(U,dot(diag(S),V))
F = F/F[2,2]

# reverse normalization
F = dot(T1.T,dot(F,T2))
F = F/F[2,2]

print(F)

xl = np.array([[320],[79],[1]])
xr = np.array([[77],[141],[1]])

def getLine(m,n,F,x):
    line = dot(F,x)
    
    # epipolar line parameter and values
    t = linspace(0,n,100)
    lt = array([(line[2]+line[0]*tt)/(-line[1]) for tt in t])
    ndx = (lt>=0) & (lt<m)

    return t , lt 

subplot(221),imshow(imgleft),title('Left Image')
m,n = imgleft.shape[:2]
t,lt = getLine(m,n,F,xr)
plot(t,lt,linewidth=2)

subplot(222),imshow(imgright),title('Right Image')
m,n = imgright.shape[:2]
t,lt = getLine(m,n,F.T,xl)
plot(t,lt,linewidth=2)
show()




    
       
        
        

