import numpy as np
import math 
import cv2
from scipy import linalg

B = np.array([[121.474,99.2211,151.215,87.697],[877.978,99.682,491.443,52.799]
,[877.782,603799,608.659,311.531],[121.684,603.938,658.396,510.912]])#[121.907,351.992,1105.87,861.454]
n,m = B.shape

for i in range(0,n):
    C = np.array([[0,0,0,-B[i,0],-B[i,1],-1,B[i,3]*B[i,0],B[i,1]*B[i,3],B[i,3]],
                  [B[i,0],B[i,1],1,0,0,0,-B[i,0]*B[i,2],-B[i,1]*B[i,2],-B[i,2]]])
    if i == 0:
        ys = np.array([], dtype=np.int64).reshape(0,9)
        A = np.vstack([ys, C])
    else :
        A = np.concatenate((A, C), axis=0) 

print(A) 

U,S,V = linalg.svd(A)
H = V[-1].reshape(3,3)
print(S)
U,S,V = linalg.svd(F)
print(S)
S[2] = 0
H = dot(U,dot(diag(S),V))
h = H/H[2,2]
print(H)




    
       
        
        

