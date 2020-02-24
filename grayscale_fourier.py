#!/usr/bin/env python3.6
import cv2
import sys
import numpy as np
from matplotlib import pyplot as plt

# To get rid of those pesky 'RuntimeWarning'
np.seterr(divide = 'ignore') 

if len(sys.argv) != 2:
    print('Usage: %s <grayscale file>'%(sys.argv[0]))
    sys.exit(1)

plt.style.use('dark_background')
fig, axes = plt.subplots(nrows=3, ncols=3, sharex=True, sharey=True, figsize = (12,6))

# Load image
img = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)

# Picture dimensions
rows, cols = img.shape
# Get best/optimal picture size to speedup Fourier Transforms by a factor of 4!
best_rows = cv2.getOptimalDFTSize(rows)
best_cols = cv2.getOptimalDFTSize(cols)
# Zero out everything extra we are adding
nimg = np.zeros((best_rows, best_cols))
# Replace original with optimal size
img[:rows,:cols] = img

# Now start
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))

# Plot numbers are Row, Cols, Index
plt.subplot(331), plt.imshow(img, cmap = 'gray')
plt.title('Input Image', position=(0.17,0.03), color='white'), plt.xticks([]), plt.yticks([])
plt.subplot(332), plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum', position=(0.3,0.03), color='white'), plt.xticks([]), plt.yticks([])
plt.subplot(333), plt.imshow(img)
plt.title('JET', position=(0.06,0.03), color='white'), plt.xticks([]), plt.yticks([])

# Center of picture
crow, ccol = int(rows/2.0+.5) , int(cols/2.0+.5)
print('rows [%s], cols[%s], crow[%s], ccol[%s]'%(rows, cols, crow, ccol))
'''
rows [183], cols[275], crow[92], ccol[138]
Assuming a radius of "2", for the following 11x11 matrices:

  Matrix for Full Image       Matrix for High Pass        Matrix for Low Pass    
  0 1 2 3 4 5 6 7 8 9 0       0 1 2 3 4 5 6 7 8 9 0       0 1 2 3 4 5 6 7 8 9 0   
0 1 1 1 1 1 1 1 1 1 1 1     0 1 1 1 1 1 1 1 1 1 1 1     0 0 0 0 0 0 0 0 0 0 0 0   
1 1 1 1 1 1 1 1 1 1 1 1     1 1 1 1 1 1 1 1 1 1 1 1     1 0 0 0 0 0 0 0 0 0 0 0   
2 1 1 1 1 1 1 1 1 1 1 1     2 1 1 1 1 1 1 1 1 1 1 1     2 0 0 0 0 0 0 0 0 0 0 0   
3 1 1 1 1 1 1 1 1 1 1 1     3 1 1 1 0 0 0 0 0 1 1 1     3 0 0 0 1 1 1 1 1 0 0 0   
4 1 1 1 1 1 1 1 1 1 1 1     4 1 1 1 0 0 0 0 0 1 1 1     4 0 0 0 1 1 1 1 1 0 0 0   
5 1 1 1 1 1 1 1 1 1 1 1     5 1 1 1 0 0 X 0 0 1 1 1     5 0 0 0 1 1 X 1 1 0 0 0   
6 1 1 1 1 1 1 1 1 1 1 1     6 1 1 1 0 0 0 0 0 1 1 1     6 0 0 0 1 1 1 1 1 0 0 0   
7 1 1 1 1 1 1 1 1 1 1 1     7 1 1 1 0 0 0 0 0 1 1 1     7 0 0 0 1 1 1 1 1 0 0 0   
8 1 1 1 1 1 1 1 1 1 1 1     8 1 1 1 1 1 1 1 1 1 1 1     8 0 0 0 0 0 0 0 0 0 0 0   
9 1 1 1 1 1 1 1 1 1 1 1     9 1 1 1 1 1 1 1 1 1 1 1     9 0 0 0 0 0 0 0 0 0 0 0   
0 1 1 1 1 1 1 1 1 1 1 1     0 1 1 1 1 1 1 1 1 1 1 1     0 0 0 0 0 0 0 0 0 0 0 0   
'''
radius = int(max(rows, cols) * .05)

# High Pass Filtering
highpass = np.fft.fftshift(f)
highpass[crow-radius:crow+radius, ccol-radius:ccol+radius] = 0
magnitude_spectrum_hp = 20*np.log(np.abs(highpass))
magnitude_spectrum_hp = 20*np.log(np.abs(magnitude_spectrum_hp))
h_ishift = np.fft.ifftshift(highpass)
img_highpass = np.fft.ifft2(h_ishift)
img_highpass = np.abs(img_highpass)

plt.subplot(334), plt.imshow(img_highpass, cmap = 'gray')
plt.title('Image after HPF', position=(0.21,0.03), color='white'), plt.xticks([]), plt.yticks([])
plt.subplot(335), plt.imshow(magnitude_spectrum_hp, cmap = 'gray')
#plt.title('Magnitude Spectrum', position=(0.3,0.03), color='white'), plt.xticks([]), plt.yticks([])
plt.xticks([]), plt.yticks([])
plt.subplot(336), plt.imshow(img_highpass)
#plt.title('JET', position=(0.06,0.03), color='white'), plt.xticks([]), plt.yticks([])
plt.xticks([]), plt.yticks([])

# Low Pass Filtering
'''
# You can also do it like this, but I prefer my way below, seems more intuitive?
dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
# mask created where center radius is 1, everything else is zeros
mask = np.zeros((rows,cols,2),np.uint8)
mask[crow-radius:crow+radius, ccol-radius:ccol+radius] = 1
# apply mask and inverse DFT
lp_fshift = dft_shift*mask
lpf_ishift = np.fft.ifftshift(lp_fshift)
img_lowpass = cv2.idft(lpf_ishift)
img_lowpass = cv2.magnitude(img_lowpass[:,:,0],img_lowpass[:,:,1])
'''

# old way?
lowpass = np.fft.fftshift(f)
lowpass[:] = 0
lowpass[crow-radius:crow+radius, ccol-radius:ccol+radius] = np.fft.fftshift(f)[crow-radius:crow+radius, ccol-radius:ccol+radius]
l_ishift = np.fft.ifftshift(lowpass)
img_lowpass = np.abs(np.fft.ifft2(l_ishift))
magnitude_spectrum_lp = 20*np.log(np.abs(lowpass))

plt.subplot(337), plt.imshow(img_lowpass, cmap = 'gray')
plt.title('Image after LPF', position=(0.21,0.03), color='white'), plt.xticks([]), plt.yticks([])
plt.subplot(338), plt.imshow(magnitude_spectrum_lp, cmap = 'gray')
#plt.title('Magnitude Spectrum', position=(0.3,0.03), color='white'), plt.xticks([]), plt.yticks([])
plt.xticks([]), plt.yticks([])
plt.subplot(339), plt.imshow(img_lowpass)
#plt.title('JET', position=(0.06,0.03), color='white'), plt.xticks([]), plt.yticks([])
plt.xticks([]), plt.yticks([])

plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.01, hspace=0.1)

plt.show()
