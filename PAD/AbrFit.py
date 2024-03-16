"""
AbrFit.py: aberration fitting functions 

This version: September 2018, Katherine Spoth
"""

from __future__ import division, print_function
import numpy as np
from scipy.optimize import minimize, curve_fit
from scipy.linalg import svd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import tifffile
import os
import multiprocessing as mp

from rigidregistration import stackregistration as stackreg
#Ben Savitzky; https://github.com/bsavitzky/rigidRegistration/

def ddx_1stOrder(tx, ty, C01a, C10, C12a, C12b, theta):
    tx2=tx*np.cos(theta)-ty*np.sin(theta)
    ty2=tx*np.sin(theta)+ty*np.cos(theta)
    return C01a+C10*tx2+C12a*tx2+C12b*ty2

def ddy_1stOrder(tx, ty, C01b, C10, C12a, C12b, theta):
    tx2=tx*np.cos(theta)-ty*np.sin(theta)
    ty2=tx*np.sin(theta)+ty*np.cos(theta)                          
    return C01b+C10*ty2+C12a*(-ty2)+C12b*tx2

def abr_1stOrder(tx, ty, C01a, C01b, C10, C12a, C12b):
    return C01a*tx+C01b*ty+C10*(tx**2+ty**2)/2+C12a*(tx**2-ty**2)/2+C12b*tx*ty

def abr_1stOrder_rotation(tx, ty, C01a, C01b, C10, C12a, C12b, theta):
    tx2=tx*np.cos(theta)-ty*np.sin(theta)
    ty2=tx*np.sin(theta)+ty*np.cos(theta)
    return C01a*tx2+C01b*ty2+C10*(tx2**2+ty2**2)/2+C12a*(tx2**2-ty2**2)/2+C12b*tx2*ty2
                             
def fit_aberrations_1stOrder(pad, plot=True):

    coords = np.argwhere(pad.tcmask!=0)
    #convert the coordinates to radians with convergence angle over cbedrad
    xcoords = (coords[:, 1]-pad.cbedcenterx)*(pad.ca/pad.cbedrad)
    ycoords = (coords[:, 0]-pad.cbedcentery)*(pad.ca/pad.cbedrad)
        
    #function to minimize for aberration coefficients
#     C01a,C01b,C10,C12a,C12b
    err =  lambda C : np.sum((pad.xshiftmap[np.nonzero(pad.tcmask)]*pad.pixelsize-ddx_1stOrder(xcoords, ycoords, C[0], C[2], C[3], C[4], C[5]))**2)+np.sum((pad.yshiftmap[np.nonzero(pad.tcmask)]*pad.pixelsize-ddy_1stOrder(xcoords, ycoords, C[1], C[2], C[3], C[4], C[5]))**2)
    
    guess = (10, 10, -1000, 10, 10, 1.6*np.pi)
    methods = ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'Newton-CG', 'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP', 'dogleg', 'trust-ncg']

    minerr = 1000
#     minerr = 3000
    minmeth = ''
    minres = []

    for method in methods:
         try:
            print(method)
            res = minimize(err, guess, method=method)
            #print res.x
            print(err(res.x))
            if err(res.x) < minerr:
                minerr = err(res.x)
                minmeth = method
                minres = res.x
                print('succeed!')
            else:
                print('fail!')
         except:
            print("there was an issue with this one")

    if plot:
        zerosx = np.zeros_like(pad.xshiftmap)
        pad.xFit = np.zeros_like(pad.xshiftmap)
        zerosx[np.nonzero(pad.tcmask)] = ddx_1stOrder(xcoords, ycoords, minres[0], minres[2], minres[3], minres[4],minres[5])
        pad.xFit = zerosx
        functions.show_map_lower(zerosx)
        
        
        zerosy = np.zeros_like(pad.yshiftmap)
        pad.yFit = np.zeros_like(pad.yshiftmap)
        zerosy[np.nonzero(pad.tcmask)] = ddy_1stOrder(xcoords, ycoords, minres[1], minres[2], minres[3], minres[4],minres[5])
        pad.yFit = zerosy
        functions.show_map_lower(zerosy)

        zeroAbr = np.zeros_like(pad.yshiftmap)
        pad.abrFit = np.zeros_like(pad.yshiftmap)
        zeroAbr[np.nonzero(pad.tcmask)] = abr_1stOrder(xcoords, ycoords, minres[0],minres[1], minres[2], minres[3], minres[4])
        pad.abrFit = zeroAbr
        functions.show_map_lower(zeroAbr)
                
        zeroAbr = np.zeros_like(pad.yshiftmap)
        pad.abrFit = np.zeros_like(pad.yshiftmap)
        zeroAbr[np.nonzero(pad.tcmask)] = abr_1stOrder_rotation(xcoords, ycoords, minres[0],minres[1], minres[2], minres[3], minres[4], minres[5])
        pad.abrFit = zeroAbr
        functions.show_map_lower(zeroAbr)
        
    return minres

def ddx_2ndOrder(tx, ty, C01a, C10, C12a, C12b, C21a, C21b, C23a, C23b, theta):
    tx2=tx*np.cos(theta)-ty*np.sin(theta)
    ty2=tx*np.sin(theta)+ty*np.cos(theta)
    return C01a+C10*tx2+C12a*tx2+C12b*ty2+C21a*(tx2**2+(ty2**2)/3)+C21b*2/3*tx2*ty2+C23a*(tx2**2-ty2**2)+C23b*2*tx2*ty2

def ddy_2ndOrder(tx, ty, C01b, C10, C12a, C12b, C21a, C21b, C23a, C23b, theta):
    tx2=tx*np.cos(theta)-ty*np.sin(theta)
    ty2=tx*np.sin(theta)+ty*np.cos(theta)
    return C01b+C10*ty2+C12a*(-ty2)+C12b*tx2+C21a*(2/3*tx2*ty2)+C21b*((tx2**2)/3+ty2**2)+C23a*(-2*tx2*ty2)+C23b*(tx2**2-ty2**2)


def abr_2ndOrder(tx, ty, C01a, C01b, C10, C12a, C12b, C21a, C21b, C23a, C23b):

    return C01a*tx+C01b*ty+C10*(tx**2+ty**2)/2+C12a*(tx**2-ty**2)/2+C12b*tx*ty+C21a*(tx**3+tx*ty**2)/3+C21b*(tx**2*ty+ty**3)/3+C23a*(tx**3-3*tx*ty**2)/3+C23b*(3*tx**2*ty-ty**3)/3


def abr_2ndOrder_rotation(tx, ty, C01a, C01b, C10, C12a, C12b, C21a, C21b, C23a, C23b,theta):
    tx2=tx*np.cos(theta)-ty*np.sin(theta)
    ty2=tx*np.sin(theta)+ty*np.cos(theta)
    return C01a*tx2+C01b*ty2+C10*(tx2**2+ty2**2)/2+C12a*(tx2**2-ty2**2)/2+C12b*tx2*ty2+C21a*(tx2**3+tx2*ty2**2)/3+C21b*(tx2**2*ty2+ty2**3)/3+C23a*(tx2**3-3*tx2*ty2**2)/3+C23b*(3*tx2**2*ty2-ty2**3)/3



def fit_aberrations_2ndOrder(pad, plot=True):

#     if not hasattr(pad, 'ca'):
#         pad.ca = float(raw_input('Enter convergence angle in mrad: '))/1000
# #     if not hasattr(pad, "bf_recon"):
#         print "must get good BF recon first!"
#     else:
    coords = np.argwhere(pad.tcmask!=0)
    #convert the coordinates to radians with convergence angle over cbedrad
    xcoords = (coords[:, 1]-pad.cbedcenterx)*(pad.ca/pad.cbedrad)
    ycoords = (coords[:, 0]-pad.cbedcentery)*(pad.ca/pad.cbedrad)
        
    #function to minimize for aberration coefficients
#     C01a,C01b,C10,C12a,C12b,C21a,C21b,C23a,C23b
    err =  lambda C : np.sum((pad.xshiftmap[np.nonzero(pad.tcmask)]*pad.pixelsize-ddx_2ndOrder(xcoords, ycoords, C[0], C[2], C[3], C[4], C[5], C[6], C[7], C[8],C[9]))**2)+np.sum((pad.yshiftmap[np.nonzero(pad.tcmask)]*pad.pixelsize-ddy_2ndOrder(xcoords, ycoords, C[1], C[2], C[3], C[4], C[5], C[6], C[7], C[8], C[9]))**2)
    
    guess = (10, 10, 1000, 10, 10, 10, 10, 10, 10, 1.6*np.pi)
    methods = ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'Newton-CG', 'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP', 'dogleg', 'trust-ncg']

#     minerr = 600
    minerr = 1000

    minmeth = ''
    minres = []

    for method in methods:
         try:
            print(method)
            res = minimize(err, guess,method=method)
            #print res.x
            print(err(res.x))
            if err(res.x) < minerr:
                minerr = err(res.x)
                minmeth = method
                minres = res.x
                print('succeed!')
            else:
                print('fail!')
         except:
            print("there was an issue with this one")

    if plot:
        zerosx = np.zeros_like(pad.xshiftmap)
        pad.xFit = np.zeros_like(pad.xshiftmap)
        zerosx[np.nonzero(pad.tcmask)] = ddx_2ndOrder(xcoords, ycoords, minres[0], minres[2], minres[3], minres[4], minres[5], minres[6], minres[7], minres[8],minres[9])
        pad.xFit = zerosx
        functions.show_map_lower(zerosx)
        
        zerosy = np.zeros_like(pad.yshiftmap)
        pad.yFit = np.zeros_like(pad.yshiftmap)
        zerosy[np.nonzero(pad.tcmask)] = ddy_2ndOrder(xcoords, ycoords, minres[1], minres[2], minres[3], minres[4], minres[5], minres[6], minres[7], minres[8],minres[9])
        pad.yFit = zerosy
        functions.show_map_lower(zerosy)
        
        zeroAbr = np.zeros_like(pad.yshiftmap)
        pad.abrFit = np.zeros_like(pad.yshiftmap)
        zeroAbr[np.nonzero(pad.tcmask)] = abr_2ndOrder(xcoords, ycoords, minres[0],minres[1], minres[2], minres[3], minres[4], minres[5], minres[6], minres[7], minres[8])
        pad.abrFit = zeroAbr
        functions.show_map_lower(zeroAbr)

        zeroAbr = np.zeros_like(pad.yshiftmap)
        pad.abrFit = np.zeros_like(pad.yshiftmap)
        zeroAbr[np.nonzero(pad.tcmask)] = abr_2ndOrder_rotation(xcoords, ycoords, minres[0],minres[1], minres[2], minres[3], minres[4], minres[5], minres[6], minres[7], minres[8],minres[9])
        pad.abrFit = zeroAbr
        functions.show_map_lower(zeroAbr)


    return minres

def ddx_3rdOrder(tx, ty, C01a, C10, C12a, C12b, C21a, C21b, C23a, C23b, C30, C32a, C32b, C34a, C34b, theta):
    tx2=tx*np.cos(theta)-ty*np.sin(theta)
    ty2=tx*np.sin(theta)+ty*np.cos(theta)
    return C01a+C10*tx2+C12a*tx2+C12b*ty2+C21a*(tx2**2+(ty2**2)/3)+C21b*2/3*tx2*ty2+C23a*(tx2**2-ty2**2)+C23b*2*tx2*ty2+C30*(tx2**3+tx2*ty2**2)+C32a*(tx2**3)+C32b*(3/2*tx2**2*ty2+1/2*ty2**3)+C34a*(tx2**3-3*tx2*ty2**2)+C34b*(3*tx2**2*ty2 - ty2**3)/2

def ddy_3rdOrder(tx, ty, C01b, C10, C12a, C12b, C21a, C21b, C23a, C23b, C30, C32a, C32b, C34a, C34b, theta):
    tx2=tx*np.cos(theta)-ty*np.sin(theta)
    ty2=tx*np.sin(theta)+ty*np.cos(theta)
    return C01b+C10*ty2+C12a*(-ty2)+C12b*tx2+C21a*(2/3*tx2*ty2)+C21b*((tx2**2)/3+ty2**2)+C23a*(-2*tx2*ty2)+C23b*(tx2**2-ty2**2)+C30*(tx2**2*ty2+ty2**3)+C32a*(-ty2**3)+C32b*(1/2*tx2**3+3/2*tx2*ty2**2)+C34a*(-3*tx2**2*ty2+ty2**3)+C34b*(tx2**3-3*tx2*ty2**2)/2

def abr_3rdOrder(tx, ty, C01a, C01b, C10, C12a, C12b, C21a, C21b, C23a, C23b, C30, C32a, C32b, C34a, C34b):
    return C01a*tx+C01b*ty+C10*(tx**2+ty**2)/2+C12a*(tx**2-ty**2)/2+C12b*tx*ty+C21a*(tx**3+tx*ty**2)/3+C21b*(tx**2*ty+ty**3)/3+C23a*(tx**3-3*tx*ty**2)/3+C23b*(3*tx**2*ty-ty**3)/3+C30*(tx**4+2*tx**2*ty**2+ty**4)/4+C32a*(tx**4-ty**4)/4+C32b*(tx**3*ty+tx*ty**3)/2+C34a*(tx**4-6*tx**2*ty**2+ty**4)/4+C34b*(tx**3*ty-tx*ty**3)/2
       
def abr_3rdOrder_rotation(tx, ty, C01a, C01b, C10, C12a, C12b, C21a, C21b, C23a, C23b, C30, C32a, C32b, C34a, C34b, theta):
    tx2=tx*np.cos(theta)-ty*np.sin(theta)
    ty2=tx*np.sin(theta)+ty*np.cos(theta)
    return C01a*tx2+C01b*ty2+C10*(tx2**2+ty2**2)/2+C12a*(tx2**2-ty2**2)/2+C12b*tx2*ty2+C21a*(tx2**3+tx2*ty2**2)/3+C21b*(tx2**2*ty2+ty2**3)/3+C23a*(tx2**3-3*tx2*ty2**2)/3+C23b*(3*tx2**2*ty2-ty2**3)/3+C30*(tx2**4+2*tx2**2*ty2**2+ty2**4)/4+C32a*(tx2**4-ty2**4)/4+C32b*(tx2**3*ty2+tx2*ty2**3)/2+C34a*(tx2**4-6*tx2**2*ty2**2+ty2**4)/4+C34b*(tx2**3*ty2-tx2*ty2**3)/2
       

    
def fit_aberrations_3rdOrder(pad, plot=True):

#     if not hasattr(pad, 'ca'):
#         pad.ca = float(raw_input('Enter convergence angle in mrad: '))/1000
# #     if not hasattr(pad, "bf_recon"):
#         print "must get good BF recon first!"
#     else:
    coords = np.argwhere(pad.tcmask!=0)
    #convert the coordinates to radians with convergence angle over cbedrad
    xcoords = (coords[:, 1]-pad.cbedcenterx)*(pad.ca/pad.cbedrad)
    ycoords = (coords[:, 0]-pad.cbedcentery)*(pad.ca/pad.cbedrad)
        
    #function to minimize for aberration coefficients
#     C01a,C01b,C10,C12a,C12b,C21a,C21b,C23a,C23b,C30,C32a,C32b,C34a,C34b
    err =  lambda C : np.sum((pad.xshiftmap[np.nonzero(pad.tcmask)]*pad.pixelsize-ddx_3rdOrder(xcoords, ycoords, C[0], C[2], C[3], C[4], C[5], C[6], C[7], C[8], C[9],C[10],C[11],C[12],C[13],C[14]))**2)+np.sum((pad.yshiftmap[np.nonzero(pad.tcmask)]*pad.pixelsize-ddy_3rdOrder(xcoords, ycoords, C[1], C[2], C[3], C[4], C[5], C[6], C[7], C[8], C[9],C[10],C[11],C[12],C[13],C[14]))**2)
    
    guess = (10, 10, 1000, 10, 10, 10, 10, 10, 10, 100000, 100000, 100000, 100000, 100000, 0)
    methods = ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'Newton-CG', 'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP', 'dogleg', 'trust-ncg']

#     minerr = 300
    minerr = 1000

    minmeth = ''
    minres = []

    for method in methods:
         try:
            print(method)
            res = minimize(err, guess,method=method)
            #print res.x
            print(err(res.x))
            if err(res.x) < minerr:
                minerr = err(res.x)
                minmeth = method
                minres = res.x
                print('succeed!')
            else:                                                            
                print('fail!')
         except:
            print("there was an issue with this one")

    if plot:
        zerosx = np.zeros_like(pad.xshiftmap)
        pad.xFit = np.zeros_like(pad.xshiftmap)
        zerosx[np.nonzero(pad.tcmask)] = ddx_3rdOrder(xcoords, ycoords, minres[0], minres[2], minres[3], minres[4], minres[5], minres[6], minres[7], minres[8], minres[9], minres[10], minres[11], minres[12], minres[13], minres[14])
        pad.xFit = zerosx
        pad.x_1d_fit_3rd = ddx_3rdOrder(xcoords, ycoords, minres[0], minres[2], minres[3], minres[4], minres[5], minres[6], minres[7], minres[8], minres[9], minres[10], minres[11], minres[12], minres[13], minres[14])
        functions.show_map_lower(zerosx)
        
        zerosy = np.zeros_like(pad.yshiftmap)
        pad.yFit = np.zeros_like(pad.yshiftmap)
        zerosy[np.nonzero(pad.tcmask)] = ddy_3rdOrder(xcoords, ycoords, minres[1], minres[2], minres[3], minres[4], minres[5], minres[6], minres[7], minres[8], minres[9], minres[10], minres[11], minres[12], minres[13], minres[14])
        pad.yFit = zerosy
        pad.y_1d_fit_3rd = ddy_3rdOrder(xcoords, ycoords, minres[1], minres[2], minres[3], minres[4], minres[5], minres[6], minres[7], minres[8], minres[9], minres[10], minres[11], minres[12], minres[13], minres[14])
        functions.show_map_lower(zerosy)
        
        zeroAbr = np.zeros_like(pad.yshiftmap)
        pad.abrFit = np.zeros_like(pad.yshiftmap)
        zeroAbr[np.nonzero(pad.tcmask)] = abr_3rdOrder_rotation(xcoords, ycoords, minres[0],minres[1], minres[2], minres[3], minres[4], minres[5], minres[6], minres[7], minres[8], minres[9], minres[10], minres[11], minres[12], minres[13], minres[14])
        pad.abrFit = zeroAbr
        functions.show_map_lower(zeroAbr)
    
    return minres