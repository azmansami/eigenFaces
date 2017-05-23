#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 21 20:53:49 2017

@author: azmansami
"""

import scipy as sp
import numpy as np
from scipy import misc
import sys
from os import listdir, path, makedirs
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from PIL import Image


img_dir = "/Users/azmansami/yalefaces"
out_dir = img_dir+"/_reconstructed"
eface_dir = img_dir+"/_efaces"

print("Using Input Image folder: " + img_dir)

img_names = listdir(img_dir)
img_list = list()
for i in img_names:
    if (not i.startswith('.') and i != "Thumbs.db" 
        and ((".normal" in i) 
            or (".happy" in i) 
            or (".sad" in i )
            or (".glasses" in i)
            )
        ):
        #returns a 2D array  
        img = sp.misc.imread(img_dir+'/'+i,True)        
        img_list.append(img)
    
img_shape = img_list[0].shape

#individual image size 243x320 
#print (img.size)
        
#see an image for debug purpose        
#plt.imshow(img_list[0],cmap="gray")    

#unfold each image from 2D to 1D vector
#individual vector size is 77760x1 (243x320 = 77760) 
#Transposed to make 1D image vectors into columns 

imgs_mtrx=np.array([img.flatten() for img in img_list]).T
mean_img = np.sum(imgs_mtrx,axis=1)/len(imgs_mtrx[0,])

#debug, unfold mean_image to make 2D image and plot to see
mean_img_2d=mean_img.reshape(img_shape)
plt.imshow(mean_img_2d,cmap="gray")


#subtracting mean from image matrix
for c_idx in range(imgs_mtrx.shape[1]):
    imgs_mtrx[:,c_idx] = imgs_mtrx[:,c_idx] - mean_img

#"A" is image matrix minus mean image
A=imgs_mtrx

#find eigenValues and eigenVectors for C
U,s,V=np.linalg.svd(A, full_matrices=False)

#see eigenValue plot to find whcih eigenVectors are most contributing
plt.plot(s)
efaces = U #[:,0:57] #if you want to limit no. of efaces, do it here

#calculate weights for each training image.
#projet the faces in eigenVector space
weights=np.dot(efaces.T,A)

#reconstructed faces with the given weights and eigenfaces
recons_imgs = list()
for c_idx in range(imgs_mtrx.shape[1]):
    ri = mean_img + np.dot(efaces,weights[:,c_idx])
    recons_imgs.append(ri.reshape(img_shape))


#debug: see an output
#plt.imshow(recons_imgs[10],cmap="gray")

#save mean and reconstructued images
if not path.exists(out_dir): makedirs(out_dir)
if not path.exists(eface_dir): makedirs(eface_dir)
for idx, img in enumerate(recons_imgs):
    sp.misc.imsave(out_dir+"/img_"+str(idx)+".jpg",img)
sp.misc.imsave(out_dir+"/mean.jpg",mean_img_2d)

for idx in range(efaces.shape[1]):
    sp.misc.imsave(eface_dir+"/eface"+str(idx)+".jpg",efaces[:,idx].reshape(img_shape))

print("Please check " + eface_dir + " for reconstructed images")  
print("Please check " + out_dir + " for reconstructed images and mean image!")  
    
