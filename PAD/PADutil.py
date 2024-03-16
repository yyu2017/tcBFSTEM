"""
PADutil.py: helper functions and RigidRegistration usage for PAD.py

This version: September 2018, Katherine Spoth
"""

from __future__ import division, print_function
import numpy as np
from scipy.optimize import minimize, curve_fit
from scipy.linalg import svd
import scipy.fftpack as sfft
from os import getcwd, mkdir, path
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import tifffile
import math

from rigidregistration import stackregistration as stackreg
#Ben Savitzky; https://github.com/bsavitzky/rigidRegistration/

def readPAD(filename,EMPAD,dim,dtype):
    """
    Function to read in data from the EMPAD.
    Returns the data in 4D array (cbed x, cbed y, scan x, scan y)
    of 32-bit floats.
    """
    #A clumsy way to get the scan dimensions from the defualt filenames:
    # scanx = int(filename.rstrip('.raw').split('_')[-1].lstrip('xy'))
    # scany = int(filename.rstrip('.raw').split('_')[-2].lstrip('xy'))
    scanx = dim
    scany = dim

    #open the file
    contents = np.fromfile(filename, dtype = dtype)
    #reshape and return
    if EMPAD == 2:
        return np.reshape(contents, (128, 128, scany, scanx), order = 'F')
    if EMPAD == 1:
        return np.reshape(contents, (128, 130, scany, scanx), order = 'F')
    
def line(x, slope, intercept):
    """
    A line, used to find misalignments in CBED as function of scan
    position 
    (This happens often with very large fields of view - scan pivot 
    point alignment)
    """
    return slope*x + intercept

def fftshift(array, xshift, yshift):
    """
    A function to apply sub-pixel shifts to an image, using fourier 
    space.

    inputs: 
    array: the 2D image
    xshift, yshift: desired shift amount in fracitonal pixel value
    Returns: the shifted array
    """
    #create image coordinate grid
    rx, ry = np.meshgrid(np.arange(np.shape(array)[0]), 
                         np.arange(np.shape(array)[1]))
    x, y = float(np.shape(array)[0]), float(np.shape(array)[1])
    #shift in Fourier space:
    w = -np.exp(-(2j*np.pi)*(xshift*rx/x+yshift*ry/y))
    shifted_fft = np.fft.fftshift(np.fft.fft2(array))*w.T
    # inverse transform back to real space and return
    return np.abs(np.fft.ifft2(np.fft.ifftshift(shifted_fft)))

def rebin2D(array, factor):
    """
    Rebin array by specified factor. (x, y, z)
    Inputs
    array   Data to bin
    factor  Amount to bin by, such that the new array shape is the 
            original's shape modulo the factor.
    Returns:
    binned array
    """
    shape = np.shape(array)
    #for 3 dimensional array, bin each z-slice
    if len(shape) == 3:
        # #ensure even dimension
        newshape = (shape[0]//factor), (shape[1]//factor), shape[2]
        #remove pixels that we won't use
        arraycrop = array[0:(newshape[0]*factor),0:(newshape[1]*factor), :]
        #define array to write output into
        binned = np.zeros(newshape)
        #bin each slice and save
        for i in range(shape[2]):
            binned[:,:,i] = arraycrop[:,:,i].reshape([newshape[0], factor, newshape[1], factor]).sum(-1).sum(1)
    #2 dimensional array: same handling, but one slice
    else:
        newshape = shape[0]//factor, shape[1]//factor
        arraycrop = array[0:(newshape[0]*factor),0:(newshape[1]*factor)]
        binned = arraycrop.reshape([newshape[0], factor, newshape[1], factor]).sum(-1).sum(1)
   
    return binned


def rebin2D_2(array, factor):
    """
    Rebin array by specified factor.
    Inputs
    array   Data to bin. (z, x, y)
    factor  Amount to bin by, such that the new array shape is the 
            original's shape modulo the factor.
    Returns:
    binned array
    """
    shape = np.shape(array)
    #for 3 dimensional array, bin each z-slice
    if len(shape) == 3:
        #ensure even dimension
        newshape = shape[0], (shape[1]//factor), (shape[2]//factor)
        #remove pixels that we won't use
        arraycrop = array[:, 0:(newshape[1]*factor),0:(newshape[2]*factor)]
        #define array to write output into
        binned = np.zeros(newshape)
        #bin each slice and save
        for i in range(shape[0]):
            binned[i,:,:] = arraycrop[i,:,:].reshape([newshape[1], factor, newshape[2], factor]).sum(-1).sum(1)
    #2 dimensional array: same handling, but one slice
    else:
        newshape = shape[0]//factor, shape[1]//factor
        arraycrop = array[0:(newshape[0]*factor),0:(newshape[1]*factor)]
        binned = arraycrop.reshape([newshape[0], factor, newshape[1], factor]).sum(-1).sum(1)
   
    return binned

def radius_grid(array):
    """
    Function to make an array the shape of input array; each pixel is valued its radius from the center.
    """
    a, b = np.shape(array)
    xgrid, ygrid = np.meshgrid(range(a), range(b))
    xgrid = xgrid-a/2
    ygrid = ygrid-b/2
    rad = np.transpose(np.sqrt(xgrid**2+ygrid**2))
    return rad

def low_pass_window(array, b):
    """
    Defines window that low-pass filters an array when applied in Fourier space
    b is a real space size in pixels, b/FOV describes the 1/e fall-off in Fourier space.
    """
    r = max(np.shape(array))
    rad = radius_grid(array)
    weight = 10**(-rad**2/(r/b)**2)
    return weight

def low_pass_filter(array, b):
    """
    Actually filters the array, see low_pass_window for math
    """
    def lp(array, b):
        window = low_pass_window(array, b)
        return abs(np.fft.ifft2(window*np.fft.fftshift(np.fft.fft2(array))))

    if array.ndim == 2:
        return lp(array, b)
    elif array.ndim == 3:
        filtered = np.zeros_like(array)
        for i in range(np.shape(array)[-1]):
            filtered[:,:,i] = lp(array[:,:,i], b)
        return filtered

######
# Registration functions for tilt corrected BF. Utilize Ben's RigidRegistration
# code.

def make_stack(array):
    #make imstack object
    return stackreg.imstack(array)

def apply_shifts_expand(imstack, expand):
    """
    Upsample images by expand using nearest-neighbor; apply the shifts after
    expanding.

    Inputs:
    imstack     Registration.imstack object. Needs to have average shifts already
                calculated
    expand      Integer, how much by which to expand the image.
    """
    imstack.stack_registered=np.zeros((expand*imstack.nx, 
                                       expand*imstack.ny,imstack.nz))
    for z in range(imstack.nz):
        im = imstack.imstack[:,:,z]
        expim = np.zeros((expand*imstack.nx, expand*imstack.ny))
        for i in range(imstack.nx*expand):
            for j in range(imstack.ny*expand):
                expim[i,j] = im[i//expand, j//expand]
        imstack.stack_registered[:,:,z] = stackreg.generateShiftedImage(expim, 
                                   imstack.shifts_x[z]*expand, imstack.shifts_y[z]*expand) 
    imstack.average_image = np.sum(imstack.stack_registered,axis=2)
    return imstack.average_image

def show_sample_ccs(imstack, correlationType):
    if correlationType=="cc":
        getSingleCorrelation = imstack.getSingleCrossCorrelation
    elif correlationType=="mc":
        getSingleCorrelation = imstack.getSingleMutualCorrelation
    elif correlationType=="pc":
        getSingleCorrelation = imstack.getSinglePhaseCorrelation
    fft1 = imstack.fftstack[:,:,0]
    fft2 = imstack.fftstack[:,:,imstack.nz//2]
    fft3 = imstack.fftstack[:,:,-1]
    print("A few cross-correlations:")
    show(np.abs(np.fft.fftshift((getSingleCorrelation(fft1, fft2)))), cmap='jet')
    show(np.abs(np.fft.fftshift((getSingleCorrelation(fft1, fft3)))), cmap='jet')
    show(np.abs(np.fft.fftshift((getSingleCorrelation(fft2, fft3)))), cmap='jet')
    return

######
#Handy functions for displaying, so you don't have to remember
#lines and lines of pyplot code

def show(data, cmap='gray', **kwargs):
    """
    Show an image using pyplot. Removes frames, ticks, etc so plot looks
    like a normal image. Very useful for jupyter notebooks.
    Inputs:
    Data    2D image array to show
    cmap    The colormap for displaying the data. greyscale by default.
    **kwargs Passed to the figure() function - useful for changing size, 
             dpi, etc
    """
    fig = plt.figure(**kwargs)
    plt.matshow(data, fignum = False, cmap = cmap)
    a=fig.gca()
    a.set_frame_on(False)
    a.set_xticks([]); a.set_yticks([])
    plt.axis('off')
    plt.show()
    return

def show_map(array, cmap='bwr', v=False, vmin=False, vmax=False, **kwargs):
    """
    Similar to show above, but more useful for data other than images.
    I use it for the shift matrices and maps; colormap defaults to be
    symmetric about zero (eg Most red (postive) shifts are equivalent to most 
    blue (negative) shifts).
    Inputs:
    array   2D array to display
    cmap    colormap to display the data - default is bwr, any white-centered
            map is sensible for shift data
    v       cmap limits: set max as +v and min as -v automatically
    vmax, vmin  useful if user wants asymmetric data limits
    **kwargsPassed to plt.figure, useful are figsize=(), etc.
    """
    fig = plt.figure(**kwargs)
    if vmin==False and vmax==False and v==False:
        v = np.max(abs(array), axis=None)
        vmin = -v
        vmax = v
    if v!= False:
        vmin = -v
        vmax = v
    plt.matshow(array, fignum = False, cmap = cmap, vmin = vmin, vmax = vmax)
    a=fig.gca()
    a.set_frame_on(False)
    a.set_xticks([]); a.set_yticks([])
    plt.axis('off')
    plt.colorbar(shrink = 0.65)
    plt.show()
    return fig



def saveTiff(array, filename, bits=16):
    t = 'uint{}'.format(bits)

    scalearr = (imarr - np.min(imarr))/(np.max(imarr)-np.min(imarr))

    imarr = ((2**bits-1)*scalearr).astype(t)

    if len(np.shape(imarr))>2:
        imarr = np.swapaxes(imarr,0,2)
    tifffile.imsave(filename+'.tif', imarr)


##########################################################upsample##############################################

def normal(im):
    return (im-np.min(im))/(np.max(im)-np.min(im))

def interpolation(img,scale,method='null',noise=False,snr=10):
    x,y = np.shape(img)
    
    if method=='null':
        up = np.zeros((x*scale,y*scale))
        up[0::scale, 0::scale] = img[0::1, 0::1]
        
    if method=='nn':
        up = np.repeat(np.repeat(img, scale, axis=0), scale,axis=1)
            
    return up

def generateShiftedImage_real(im, xshift, yshift):

    # Define real space meshgrids nx, ny = np.shape(im)
    nx, ny = np.shape(im)
    rx,ry = np.meshgrid(np.arange(-nx/2,nx/2,1),np.arange(-ny/2,ny/2,1))
    
  #  rx,ry = rxT.T,ryT.T
    nx,ny = float(nx),float(ny)

    w = np.exp(-(2j*np.pi)*(xshift*rx/nx+yshift*ry/ny))
   
#     shifted_im = np.real(np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(np.fft.fft2(im))*w)))
#     shifted_im = np.abs(np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(np.fft.fft2(im))*w)))
#     shifted_im = np.imag(np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(np.fft.fft2(im))*w)))
    shifted_im = np.real(np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(np.fft.fft2(im))*w)))

    return shifted_im

def generateShiftedImage_abs(im, xshift, yshift):

    # Define real space meshgrids nx, ny = np.shape(im)
    nx, ny = np.shape(im)
    rx,ry = np.meshgrid(np.arange(-nx/2,nx/2,1),np.arange(-ny/2,ny/2,1))
    
  #  rx,ry = rxT.T,ryT.T
    nx,ny = float(nx),float(ny)

    w = np.exp(-(2j*np.pi)*(xshift*rx/nx+yshift*ry/ny))
   
    shifted_im = np.abs(np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(np.fft.fft2(im))*w)))
#     shifted_im = np.imag(np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(np.fft.fft2(im))*w)))
#     shifted_im = np.real(np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(np.fft.fft2(im))*w)))

    return shifted_im

def generateShiftedImage(im, xshift, yshift):

    # Define real space meshgrids nx, ny = np.shape(im)
    nx, ny = np.shape(im)
    rx,ry = np.meshgrid(np.arange(-nx/2,nx/2,1),np.arange(-ny/2,ny/2,1))
    
  #  rx,ry = rxT.T,ryT.T
    nx,ny = float(nx),float(ny)

    w = np.exp(-(2j*np.pi)*(xshift*rx/nx+yshift*ry/ny))
   
    shifted_im = np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(np.fft.fft2(im))*w))
#     shifted_im = np.imag(np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(np.fft.fft2(im))*w)))
#     shifted_im = np.real(np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(np.fft.fft2(im))*w)))

    return shifted_im


def shiftfunc(im,xs,ys,scale,stype='real',zeros=None):
    xdim,ydim = np.shape(im)
    xdim,ydim = int(xdim/scale),int(ydim/scale)
    
    tile = np.zeros((scale,scale))
        
    if stype == 'real':
        xsub,ysub = xs%scale,ys%scale
        xint,yint = int(math.floor(xs)),int(math.floor(ys))
        
        v_xd = xsub - math.floor(xsub)
        v_xu = 1-v_xd

        v_yr = ysub - math.floor(ysub)  
        v_yl = 1-v_yr
        
        c_xu,c_xd = int(math.floor(xsub)%scale),int(math.floor(xsub+1)%scale)
        c_yl,c_yr = int(math.floor(ysub)%scale),int(math.floor(ysub+1)%scale)
        
        # top left, right
        tile[c_xu,c_yl],tile[c_xu,c_yr] = (v_xu*v_yl),(v_xu*v_yr)
        # bottom left, right
        tile[c_xd,c_yl],tile[c_xd,c_yr] = (v_xd*v_yl),(v_xd*v_yr)
        
        upshift = np.roll(im,(xint,yint),axis=(0,1))

        tile_full = np.tile(tile,(xdim, ydim))
        
        shift_im = upshift*tile_full
        
        norm = tile_full
        
    if stype == 'phase':
#         norm =  stackreg.generateShiftedImage(zeros,xs,ys)
#         norm =  stackreg.generateShiftedImage(im,xs,ys)
#         shift_im =  stackreg.generateShiftedImage(im,xs,ys)
  
        norm =  generateShiftedImage_real(im,xs,ys)
        shift_im =  generateShiftedImage_real(im,xs,ys)
        
    return shift_im,norm

#takes in a stack, xy image shifts and upsample

def apply_shift_expand(stack,xshifts,yshifts,scale,stype='real',noise=False,snr=10):
    x,y,z = np.shape(stack)
    stack_reg = np.zeros((x*scale,y*scale,z))
    norm = np.zeros((x*scale,y*scale))
    
    if scale>1:
#         pbar = tqdm(total = z,desc = "registering")
        for i in range(z):
            if stype == 'phase':
                upstack = interpolation(stack[:,:,i],scale,method='null',noise=noise,snr=snr)
                zeros = np.zeros((x*scale,y*scale))
                ones = np.ones((x,y))
                zeros[0::scale, 0::scale] = ones[0::1, 0::1]

            else:
                upstack = interpolation(stack[:,:,i],scale,method='nn',noise=noise,snr=snr)
                zeros = None
                
            stack_reg[:,:,i],temp = shiftfunc(upstack, xshifts[i]*scale, yshifts[i]*scale,scale,stype=stype,zeros=zeros)
            norm += temp
#             pbar.update(1)

    else:
#         pbar = tqdm(total = z,desc = "registering")
        for i in range(z):
            stack_reg[:,:,i] = stackreg.generateShiftedImage(stack[:,:,i],xshifts[i],yshifts[i])
#             pbar.update(1)
        
        
    ave_im = np.sum(stack_reg,axis=2)
    
    if scale>1:
        return ave_im/norm
    else:
        return ave_im


def get_FFT(im):
    im_fft = sfft.fftshift(sfft.fft2(im))
    return(np.abs(im_fft))
