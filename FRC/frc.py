import numpy as np
from scipy.ndimage.filters import gaussian_filter1d as gf 
from EM import *

def radial_profile(array):
    a, b = np.shape(array)
    xgrid, ygrid = np.meshgrid(range(a), range(b))
    xgrid = xgrid-a/2
    ygrid = ygrid-b/2
    rad = np.transpose(np.sqrt(xgrid**2+ygrid**2))
    maxind = int(a/2)
    indices = range(0, maxind)
    profile = []
    sigma = []
    number = []
    for i in indices:
        n = np.count_nonzero((i<rad)*(rad<=i+1))
        number.append(n)
        profile.append(np.sum(array[(i<rad)*(rad<=i+1)])/n)
        sigma.append(1/np.sqrt(n/2))
    return np.array(indices), np.array(profile), np.array(sigma)

def symmetrize(im):
    im = functions.scale(im)
    x,y = np.shape(im)
    symarr = np.zeros((2*x, 2*y))
    symarr[:x, :y] = im
    symarr[x:, :y] = np.flipud(im)
    symarr[:x, y:] = np.fliplr(im)
    symarr[x:,y:] = np.flipud(np.fliplr(im))
    #functions.show(symarr)
    return symarr

def FSC(im1, im2, edge=None):
    '''
    Inputs
    edge:   "symmetrize", "periodic", "window", or "None"

    '''

    if edge=='periodic':
        I1 = functions.pfft(im1)
        I2 = functions.pfft(im2)

        print("Image 1")
        functions.show(np.abs(np.fft.ifft2(I1)))

        showi1 = np.log(np.abs(I1))
        functions.show(np.clip(showi1, np.mean(showi1)-np.std(showi1), np.percentile(showi1, 99.9)))
        print("Image 2")
        functions.show(np.abs(np.fft.ifft2(I2)))
        showi1 = np.log(np.abs(I2))
        functions.show(np.clip(showi1, np.mean(showi1)-np.std(showi1), np.percentile(showi1, 99.9)))


    elif edge=='symmetrize':
        i1 = symmetrize(im1)
        i2 = symmetrize(im2)

        I1 = functions.rebin2d(np.fft.fftshift(np.fft.fft2(i1)),2)
        I2 = functions.rebin2d(np.fft.fftshift(np.fft.fft2(i2)),2)

        print("Image 1")
        functions.show(i1)
        showi1 = np.log(np.abs(I1))
        functions.show(np.clip(showi1, np.mean(showi1)-2*np.std(showi1), np.percentile(showi1, 99.9)))
        print("Image 2")
        functions.show(i2)
        showi1 = np.log(np.abs(I2))
        functions.show(np.clip(showi1, np.mean(showi1)-2*np.std(showi1), np.percentile(showi1, 99.9)))

    
    elif edge=='window':
        i1 = functions.hamming_window(im1)
        i2 = functions.hamming_window(im2)

        I1 = np.fft.fftshift(np.fft.fft2(i1))
        I2 = np.fft.fftshift(np.fft.fft2(i2))

        print("Image 1")
        functions.show(i1)
        showi1 = np.log(np.abs(I1))
        functions.show(np.clip(showi1, np.mean(showi1)-2*np.std(showi1), np.percentile(showi1, 99.9)))
        print("Image 2")
        functions.show(i2)
        showi1 = np.log(np.abs(I2))
        functions.show(np.clip(showi1, np.mean(showi1)-2*np.std(showi1), np.percentile(showi1, 99.9)))

    elif edge==None:
        I1 = np.fft.fftshift(np.fft.fft2(im1))
        I2 = np.fft.fftshift(np.fft.fft2(im2))
        
        print("Image 1")
        functions.show(im1)
        showi1 = np.log(np.abs(I1))
        functions.show(np.clip(showi1, np.mean(showi1)-2*np.std(showi1), np.percentile(showi1, 99.9)))
        print("Image 2")
        functions.show(im2)
        showi1 = np.log(np.abs(I2))
        functions.show(np.clip(showi1, np.mean(showi1)-2*np.std(showi1), np.percentile(showi1, 99.9)))



    else:
        print('Specify something correct for "edge" input: symmetrize, periodic, window, or None')    
    
    ind, C, sigma = radial_profile(I1*np.conj(I2))
    ind, C1, sigma = radial_profile(I1*np.conj(I1))
    ind, C2, sigma = radial_profile(I2*np.conj(I2))
    
    return np.abs(C)/np.sqrt(np.abs(C1*C2)), sigma

def split_image(array):
    x, y = np.shape(array)
    im1 = np.zeros((x//2, y//2))
    im2 = np.zeros((x//2, y//2))
    
    for i in range(x):
        for j in range(y):
            if i%2 == 0:
                if j%2 == 0:
                    im1[i//2,j//2] = array[i,j]
                elif j%2 == 1:
                    im2[i//2,j//2] = array[i,j]
    return im1, im2

def crop_square(array):
    m = min(array.shape)
    return array[:m, :m]

def calculate_freq(array, pixelsize):
    m = min(array.shape)
    ind = np.arange(0,int(m/2),1)
    return ind*1/(m)*1/pixelsize