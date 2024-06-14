# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 22:55:40 2024

@author: Dell
"""

from pystackreg import StackReg
from skimage import io
from matplotlib import pyplot as plt




#load reference and "moved" image
ref_img = io.imread('F:/Germany_2022/TU Illmenau/hiwi/DataSets/histology/NDp/FLM-005_J-13-2235_1564_HE.ndpi')
offset_img = io.imread('F:/Germany_2022/TU Illmenau/hiwi/DataSets/histology/NDp/FLM-005_J-13-2235_1568_HE.ndpi')




#Rigid Body transformation
sr = StackReg(StackReg.RIGID_BODY)
out_rot = sr.register_transform(ref_img, offset_img)



fig = plt.figure(figsize=(10, 10))
ax1 = fig.add_subplot(2,2,1)
ax1.imshow(ref_img, cmap='gray')
ax1.title.set_text('Input Image')

ax3 = fig.add_subplot(2,2,3)
ax3.imshow(out_rot, cmap='gray')
ax3.title.set_text('Rigid Body')

plt.show()