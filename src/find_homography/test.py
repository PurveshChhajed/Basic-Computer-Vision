import numpy as np
import math 
import cv2
from scipy import linalg

h = np.array([[1,0.5,0],[0,1,0],[0.001,0.0001,1]])
r = np.array([[121.474],[99.2211],[1]])
x=h.dot(r)
x = x/x[2,0]
print(x)

h = np.array([[1,0.5,0],[0,1,0],[0.001,0.0001,1]])
r = np.array([[877.978],[99.682],[1]])
x=h.dot(r)
x = x/x[2,0]
print(x)

h = np.array([[1,0.5,0],[0,1,0],[0.001,0.0001,1]])
r = np.array([[877.782],[603.799],[1]])
x=h.dot(r)
x = x/x[2,0]
print(x)

h = np.array([[1,0.5,0],[0,1,0],[0.001,0.0001,1]])
r = np.array([[121.684],[603.938],[1]])
x=h.dot(r)
x = x/x[2,0]
print(x)