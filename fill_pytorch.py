from torch import nn
from torch.autograd import Variable
import torch
import cv2 
import sys
import numpy as np

def fill_original_image(original,pooled):
	for y in range(np.shape(original.data)[0]):
		for x in range(np.shape(original.data)[1]):
			if original.data[y,x,0] == 0:
				original.data[y,x,0] = pooled.data[y,x,0]
	return original

imagePath = sys.argv[1]
image_raw=cv2.imread(imagePath,-1)
image=np.zeros((np.shape(image_raw)[0],np.shape(image_raw)[1],1),float)
image[:,:,0]=image_raw
image=Variable(torch.FloatTensor(image))
func = nn.MaxPool2d(3,stride=1,padding=1)
filled = image
while True:
	filled = func(image)
	image = fill_original_image(image,filled)
	if image.data.min()!=0.0:
		break
cv2.imwrite("result.png",np.asarray(image.data).astype("uint16"))


