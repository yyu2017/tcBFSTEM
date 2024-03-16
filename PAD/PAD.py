"""
PAD.py: Module for reading, manipulation, analysis of 4D-STEM data from
        the EMPAD. 

This version: September 2018, Katherine Spoth
Thie version: Yue Editted, 1029 December
"""


from __future__ import division, print_function

import numpy as np
from scipy.optimize import minimize, curve_fit
from scipy.linalg import svd
from skimage.filters import threshold_isodata
from os import getcwd, mkdir, path
import matplotlib.pyplot as plt
from PAD import PADutil as util
from PAD import AbrFit
import math



class PADdata(object):
    """
    object for utils on 4D self data. 
    """
    
    def __init__(self, filename, EMPAD, dim, dtype='float32', voltage=300, 
                 exposure=1, savepath=getcwd()):
        """
        Initializes PAD data object by reading in the .raw datafile 
        located at path given by filename string.

        Inputs: 
            filename    string indicating path to .raw file from self. 
                        Suggest convention with number as beginning of 
                        filename, as anything before the first 
                        underscore is used as an identifier
            voltage     The microscope voltage during data acquisition: 
                        used to set counts per electron in the code.
            exposure    The exposure time in ms, deaults to 1 becuase 
                        when this code was written that was the only 
                        option available
            savepath    Where to put output files if not the current 
                        directory. Useful if you want to save in a 
                        different directory.
        """

        if EMPAD == 2:
            self.data = util.readPAD(filename, EMPAD = EMPAD, dim = dim, dtype = dtype)[:, :, :, :]
        if EMPAD == 1:
            self.data = util.readPAD(filename, EMPAD = EMPAD, dim = dim, dtype = dtype)[1:-2, 3:-2, :, :]

        #assign dimenstions as attributes for easy reference
        self.cbedx, self.cbedy, self.scanx, self.scany = np.shape(self.data)

        #define the counts per electron for determining dose rates
        if voltage == 300:
            self.countspere = 579
        else:
            self.countspere = input("Please input the counts per electron at "
                "the accelerating voltage used. ")

        self.exposure = exposure/1000 #convert ms to s

        #keep filename without extension for saving later, if necessary
        self.f = filename.split('/')[-1]
        #assign the dataset a "number" from the filename, for identification in
        #methods where data is autosaved
        self.number = self.f.split('_')[0]
        self.savepath = savepath + '/'
        self.pdfpath = savepath + '/pdf/'
        #save this for later
        # if not path.exists(self.pdfpath):
        #     mkdir(self.pdfpath)

        # self.cbedsum_orig = np.sum(np.sum(self.data, axis=2), axis=2)
        # self.cbedslice_orig = self.data[:,:,self.scanx//2, self.scany//2]
        return 


    def scan_crop(self, xmin, xmax, ymin, ymax):
        """ Crop the data in the scan dimension """
        self.data = self.data[:,:,xmin:xmax, ymin:ymax]
        #redefine the data dimensions
        self.cbedx, self.cbedy, self.scanx, self.scany = np.shape(self.data)
        return

    def cbed_crop(self, xmin, xmax, ymin, ymax):
        """ Crop the data in the CBED dimension """
        self.data = self.data[xmin:xmax, ymin:ymax, :, :]
        #redefine the data dimensions
        self.cbedx, self.cbedy, self.scanx, self.scany = np.shape(self.data)
        return

    def get_cbedsum(self):
        """Finds sum of all the CBED patterns in the dataset."""
        self.cbedsum = np.sum(np.sum(self.data, axis=2), axis=2)
        return

    def get_cbedmean(self):
        """Finds mean of all the CBED patterns in the dataset."""
        self.cbedmean = np.mean(np.mean(self.data, axis=2), axis=2)
        return

    def get_center(self, show=False):
        """
        Finds center of CBED pattern by thresholding intensity, then assuming 
        circular shape. Works well for amorphous bio data, not tested on 
        anything crystalline.
        Option "show" allows the user to verify that the circular region 
        defined by thresholding is reasonable.
        """
        def center(input, threshold):
            """ Function to minimize to find center (x,y) of thresholded region"""
            sigma = sum([(x-input[0])**2+(y-input[1])**2 for x,y in np.swapaxes(np.nonzero(threshold), 0, 1)])
            return sigma

        #Find sum of CBEDs and set threshold for bright region:
        self.get_cbedsum()
        cbedsum = self.cbedsum
        threshold = threshold_isodata(cbedsum)
        #threshold = np.mean(cbedsum) + 3*np.std(cbedsum)
        mask = cbedsum > threshold

        if show:
            util.show(mask)
            util.show(cbedsum)

        #Find thecenter (the point within the region at minimum distance to
        #all other points)
        self.cbedcenterx, self.cbedcentery = minimize(center, [self.cbedx/2, self.cbedy/2], args = mask).x
        
        #Find the radius (by assuming the area of the region is a circle)
        self.cbedrad = np.sqrt(np.sum(mask.astype('int'), axis=None)/np.pi)


        # #Handle the case for which there is no bright region (extremely low-dose
        # #data)
        # b = 1
        # while self.cbedrad==0:
        #     #low-pass filter the data
        #     if show:
        #         print('filtering low pass with b = {}'.format(b))
        #     cbedsum = util.low_pass_filter(self.cbedsum, b)
        #     if show:
        #         util.show(cbedsum)
        #     #repeat thresholding with filtered data
        #     threshold = np.mean(cbedsum) + 1*np.std(cbedsum)
        #     mask = cbedsum > threshold
        #     if show:
        #         util.show(mask)
        #     #find center and radius as above
        #     self.cbedcenterx, self.cbedcentery = minimize(center, [self.cbedx/2, 
        #                                          self.cbedy/2], args = mask).x
        #     self.cbedrad = np.sqrt(np.sum(mask.astype('int'), axis=None)/np.pi)
        #     b += 1
        #     #repeats until a bright region is found

        return



    def auto_scan_crop(self, image='incobf'):
        """
        Looks for dark columns caused by blanking the beam in cryo-imaging.
        inputs:
        image   Specify whether to use the ADF or the BF image for cropping. 
                Usually bf works, but sometimes if it fails ADF might be better.
        """

        #get a scanned image to use to find dark columns
        if image=='incobf':
            self.get_incobf()
            imarr = self.incobf
        elif image=='adf':
            self.get_adf()
            imarr = self.adf
        elif image=='bf':
            self.get_bf()
            imarr = self.bf
        else:
            print("Specify image type 'incobf', 'bf', or 'adf'.")
            return

        threshold = np.min(imarr)+np.std(imarr)
        #find columns where many pixels are dark
        #colsnum is a 1D array giving number of dark pixels per column
        colsnum = np.sum(imarr<threshold, axis=0)
        colsmask = colsnum > 0
        #defining columns to discard in cols
        cols = np.zeros_like(colsmask)

        #start from column zero - we'll only discard if columns touch the edges
        i = 0
        while colsmask[i]:
            cols[i] = True
            i = i+1

        #and the same from the rightmost column:
        i = self.scany-1
        while colsmask[i]:
            cols[i] = True
            i = i-1

        self.data = self.data[:,:,:,~cols]
        self.cbedx, self.cbedy, self.scanx, self.scany = np.shape(self.data)
        
        print("Cropped {} columns".format(np.count_nonzero(cols)))

        "Regenerate original image with cropped data"
        if image=='incobf':
            self.get_incobf()
        elif image=='adf':
            self.get_adf()
        elif image=='bf':
            self.get_bf()
        return

    def bin_cbeds(self, factor):
        """
        bin dataset in CBED space. Useful for registering extremely low dose
        data when using tilt-corrected bright field, or simple reduction in data
        size that preserves real-space pixel size.

        Inputs:
        factor  factor by which to bin the CBEDs

        Returns:
        binned array

        """

        #shape array to be 3D stack of CBEDs (can then pass to rebin2D)
        cbedstack = np.reshape(self.data, (self.cbedx, self.cbedy, 
                                           self.scanx*self.scany))
        binnedstack = util.rebin2D(cbedstack, factor)
        #reshape to 4D
        self.data = np.reshape(binnedstack, (self.cbedx//factor, 
                                             self.cbedy//factor, 
                                             self.scanx, self.scany))
        #update the data dimensions, and the center position as both have changed
        self.cbedx, self.cbedy, self.scanx, self.scany = np.shape(self.data)
        self.get_center()

        return
    
    def bin_cbedsTW4D(self, factor):
        shape = np.shape(self.data)
        newshape = (shape[0]//factor), (shape[1]//factor), shape[2], shape[3]
        binned = np.zeros(newshape)
        
        for i in range(shape[2]):
            for j in range(shape[3]):
                binned[:,:,i,j] = self.data[:newshape[0]*factor,:newshape[1]*factor,i,j].reshape([newshape[0], factor, newshape[1], factor]).sum(-1).sum(1)
        
        self.data = binned
        #update the data dimensions, and the center position as both have changed
        self.cbedx, self.cbedy, self.scanx, self.scany = np.shape(self.data)
        self.get_center()

        return
    
    def bin_realSpace(self, factor):
        """
        bin dataset in real space. Useful for obtaining shift matrix for large scan pixels for the sake of time.

        Inputs:
        factor  factor by which to bin the real space

        Returns:
        binned array

        """

        #shape array to be 3D stack of real-space images (can then pass to rebin2D)
        realSpaceStack = np.reshape(self.data, (self.cbedx*self.cbedy, self.scanx, self.scany))
        binnedstack = util.rebin2D_2(realSpaceStack, factor)
        #reshape to 4D
        self.data = np.reshape(binnedstack, (self.cbedx, 
                                             self.cbedy, 
                                             self.scanx//factor, self.scany//factor))
        #update the data dimensions, and the center position as both have changed
        self.cbedx, self.cbedy, self.scanx, self.scany = np.shape(self.data)
        self.get_center()

        return

    def make_mask(self, radius=1):
        """
        Make a circular mask in CBED space, for defining bright field image
        regions.

        Inputs:
        radius  float/integer/number multiplying the CBED radius 
                from get_center()        
        """

        xgrid, ygrid = np.meshgrid(range(self.cbedx), range(self.cbedy))
        xgrid = xgrid-self.cbedcenterx
        ygrid = ygrid-self.cbedcentery
        rad = np.transpose(np.sqrt(ygrid**2+xgrid**2))
        outer = self.cbedrad*radius
        mask = rad < outer
        return mask

    def make_singlePixel_mask(self, radius=1, angle=180):
        """
        Make a single mask in CBED space, for pulling out single CBED pixel to 
        form image and for comparing shifts between images.

        Inputs:
        radius  if not specified, assuming to be 1(the CBED radius). 
        angle in degree. 0 is in positive x direction. 

        """
        if not hasattr(self, 'cbedrad'):
            self.get_center()

        mask = np.zeros((self.cbedx,self.cbedy),dtype='bool')
        pixel_x = self.cbedcenterx + radius * self.cbedrad * np.cos(np.pi*angle/180)
        pixel_y = self.cbedcentery + radius * self.cbedrad * np.sin(np.pi*angle/180)
        for i in range(0,self.cbedx):
            for j in range(0,self.cbedy):
                mask[i][j] = i==np.floor(pixel_x) and j==np.floor(pixel_y)
        return mask


    def show_singlePixel_CBED(self, radius=1, angle=180, **kwargs):
        """
        Show a single pixel in the CBED space. The single pixel is plotted with cmap = 'gray', and the
        single pixel is plotted with cmap = 'copper'.

        Inputs: 
        radius  if not specified, assuming to be 1(the CBED radius). 
        angle in degree. 0 is in positive x direction.

        The CBED

        """
        
        if not hasattr(self, 'cbedsum'):
            self.get_cbedsum()

        mask = self.make_singlePixel_mask(radius, angle)
        util.show(self.cbedsum)
        util.show(mask, cmap='copper')

        return 


        
    def find_binning(self, min_electrons=250000):
        """
        Determines the amount by which to bin in order to retain a minimum
        number of electrons in each bright field pixel. Useful for automating
        tcBF-STEM on low-dose samples.

        Inputs:
        The data array
        min_electrons   integer, desctibing the minimum amount of electrons
                        (not counts!) desired in each frame
        Returns:
        binf            integer, describing by how much the data should be bin

        """

        self.get_center()
        squaremask = self.make_mask(radius = 0.75)
        mean = np.mean(np.sum(np.sum(self.data[squaremask.astype(bool)],
                       axis=-1), axis=-1))/579
        if min_electrons < mean:
            return 1
        else:
            binf = int(np.sqrt(min_electrons/mean))+1
            return binf

    def find_frameCounts(self):
        """
        Return the number of electron counts in each bright field pixel.
    
        """
        self.get_center()
        squaremask = self.make_mask(radius = 0.75)
        mean = np.mean(np.sum(np.sum(self.data[squaremask.astype(bool)],
                   axis=-1), axis=-1))/579
        return mean

    def center_cbeds(self):
        """
        This function corrects for large shifts in the CBEDs as a function of 
        probe position (linear in scanx, scany coordinates.) This occurs 
        with misalignments of the scan pivot points, or is simply more 
        noticeable when scanning over large fields of view.
        """
        self.get_center()
        #Find center-of-mass image
        self.get_com()

        #get the mean of COM componenets: x and y relative to x, y coords
        #Fit the means as a function of coord to a line
        xymeans = np.mean(self.comx, axis=0)
        xypopt, xypcov = curve_fit(util.line, range(self.scany), xymeans)

        xxmeans = np.mean(self.comx, axis=1)
        xxpopt, xxpcov = curve_fit(util.line, range(self.scanx), xxmeans)

        yxmeans = np.mean(self.comy, axis=1)
        yxpopt, yxpcov = curve_fit(util.line, range(self.scanx), yxmeans)

        yymeans = np.mean(self.comy, axis=0)
        yypopt, yypcov = curve_fit(util.line, range(self.scany), yymeans)


        #Define a list of x, y, shifts that will correct any linear dependence
        #found
        self.cbedshiftsx = np.zeros_like(self.com)
        self.cbedshiftsy = np.zeros_like(self.com)

        for i in range(self.scanx):
            for j in range(self.scany):
                
                self.cbedshiftsx[i,j] = (xypopt[0]*j+xypopt[1]
                                        + xxpopt[0]*i+xxpopt[1])
                self.cbedshiftsy[i,j] = (yxpopt[0]*i+yxpopt[1] 
                                        + yypopt[0]*j+yypopt[1])
                #Apply the shifts to subpixel accuracy
                self.data[:,:,i,j] = util.fftshift(self.data[:,:,i,j], 
                                              -self.cbedshiftsx[i,j], 
                                              -self.cbedshiftsy[i,j])

        #Crop the CBEDs to prevent any wraparound from shifting
        xcrop = int(np.max(abs(self.cbedshiftsx)))+1
        ycrop = int(np.max(abs(self.cbedshiftsy)))+1

        self.cbed_crop(xcrop, -xcrop, ycrop, -ycrop)

        #Get the following attributes with the shifted CBEDs:
        self.get_cbedsum()
        self.get_cbedmean()
        self.get_center()
        self.get_com()

        return


    def subtract_background(self, threshold=20):
        """
    	Removes intensity offset of each CBED pattern. Iterating over scan 
        pixels, histogram is fit to Gaussian near zero. The center of the
        distribution is subtracted.

        Inputs: 
        threshold defines maxiumum intensity up to which data histogram is fit

        Returns:
        Array with data at each scan pixel shifted to make them have the same
        zero value
    	"""

        def gauss1d(x, amplitude, x0, sigma_x, offset):
            # Returns result as a 1D array that can be passed to 
            # scipy.optimize.curve_fit
            x0 = float(x0)
            g = offset+amplitude*np.exp(-1/(2*sigma_x**2)*(x-x0)**2)
            return g

        self.background = np.zeros((self.scanx, self.scany))

        #iterate over image pixels
        for i in range(self.scanx):
            for j in range(self.scany):
                try:
                    try:
                        #Fit the data below the threshold to a gaussian
                        hist, bins = np.histogram(self.data[:,:,i,j][self.data[:,:,i,j]<threshold], bins=128)
                        popt, pcov = curve_fit(gauss1d, bins[:-1], hist, p0=[np.max(hist), 
                                               np.mean(self.data[:,:,i,j][self.data[:,:,i,j]<threshold]), 
                                               np.std(self.data[:,:,i,j][self.data[:,:,i,j]<threshold]), 0])
                        #subtract the offset of the data
                        self.data[:,:,i,j] = self.data[:,:,i,j] - popt[1]
                        self.background[i,j] = popt[1]
                    except:
                        #If this doesn't work, try a higher threshold
                        hist, bins = np.histogram(self.data[:,:,i,j][self.data[:,:,i,j]<100], bins=128)
                        popt, pcov = curve_fit(gauss1d, bins[:-1], hist, p0=[np.max(hist), 
                                               np.mean(self.data[:,:,i,j][self.data[:,:,i,j]<10]), 
                                               np.std(self.data[:,:,i,j][self.data[:,:,i,j]<10]), 0])
                        self.data[:,:,i,j] = self.data[:,:,i,j] - popt[1]
                        self.background[i,j] = popt[1]
                except:
                    #If this doesn't work, forget the fitting and use the mean
                    print("fitting failed at pixel {}, {}; using mean of data < 10".format(i,j))
                    self.data[:,:,i,j] = (self.data[:,:,i,j] 
                                         - np.mean(self.data[:,:,i,j][self.data[:,:,i,j]<10]))
                    self.background[i,j] = np.mean(self.data[:,:,i,j][self.data[:,:,i,j]<10])       
        return

    def subtract_background_image(self, background):
        """
        A function which you hopefully do not have to use.
        If for some reason, the detector's automatic background subtraction
        has failed, this function will subtract a background image from
        every CBED image.

        Good ways to get the background are to save a small scan with the 
        beam blanked, or to look for regions of blanked data (often happens
        in cryo imaging) where an average image can be extracted.

        Inputs:
        background  A 2D array of size cbedx, cbedy to subtract from each 
                    CBED image 
        """

        for i in range(self.scanx):
            for j in range(self.scany):
                self.data[:,:,i,j] = self.data[:,:,i,j] - background
        return

    def threshold_dark(self, threshold=15):
        #removes small/negative counts from dataset
        mask = self.data < threshold
        self.data[mask] = 0.0

        #get the sum again with these fixed
        self.get_cbedsum()

        #Total dose in number of electrons
        self.dose = np.sum(self.data, axis=None)/self.countspere
        #Probe current, in Amperes
        self.current = self.dose * 1.60217646E-19 / (self.scanx*self.scany*self.exposure)
        return


    def get_adf(self,inner_factor=3):
        self.get_cbedsum()
        if not hasattr(self, 'cbedrad'):
            self.get_center()
        self.adf = np.zeros((self.scanx, self.scany))

        xgrid, ygrid = np.meshgrid(range(self.cbedx), range(self.cbedy))
        xgrid = xgrid-self.cbedcenterx
        ygrid = ygrid-self.cbedcentery
        rad = np.transpose(np.sqrt(ygrid**2+xgrid**2))
        inner = self.cbedrad * inner_factor
        outer = min(self.cbedx-self.cbedcenterx, self.cbedcenterx, self.cbedy-self.cbedcentery, self.cbedcentery)
        mask = (rad >= inner) * (rad < outer)

        for i in range(self.scanx):
            for j in range(self.scany):
                self.adf[i,j] = np.sum(self.data[:,:,i,j][mask])

        self.adfinner = inner
        self.adfouter = outer
        return
    
    def get_adf_radius(self,inner,outer):
        self.get_cbedsum()
        if not hasattr(self, 'cbedrad'):
            self.get_center()
        self.adf_new = np.zeros((self.scanx, self.scany))
        xgrid, ygrid = np.meshgrid(range(self.cbedx), range(self.cbedy))
        xgrid = xgrid-self.cbedcenterx
        ygrid = ygrid-self.cbedcentery
        rad = np.transpose(np.sqrt(ygrid**2+xgrid**2))
        mask = (rad >= inner) * (rad < outer)
        for i in range(self.scanx):
            for j in range(self.scany):
                self.adf_new[i,j] = np.sum(self.data[:,:,i,j][mask])
        self.adfinner = inner
        self.adfouter = outer


    def get_bf(self):
        """
        Generate a bright-field image based on a detector subtending 1/3
        the convergence angle in CBED space, centered on the BF disk
        """
        if not hasattr(self, 'cbedrad'):
            self.get_center()
        mask = self.make_mask(radius=1/3)
        self.bfrad = 1/3*self.cbedrad
        self.bf = np.sum(self.data[mask, :, :], axis=0)
        return


    def get_singlePixel_bf(self, radius=1, angle=180):
        """
        Generate a bright-field image using a single pixel in CBED. 
        Useful for comparing image shifts for tcBF.
        """
        mask = self.make_singlePixel_mask(radius, angle)
        self.singlePixel_bf = np.sum(self.data[mask, :, :], axis=0)
        return


    def get_incobf(self):
        """
        Generate an incoherent bright-field image based on a detector 
        subtending the full convergence angle in CBED space, centered 
        on the BF disk
        """
        if not hasattr(self, 'cbedrad'):
            self.get_center()
        mask = self.make_mask(radius=1)
        self.incobf = np.sum(self.data[mask, :, :], axis=0)
        return


    def get_com(self):
        """
        Generate a center of mass image.

        For each scan pixel, we find the position of the center of mass relative
        to the center of the CBED patterns found in get_center.

        Returns:
        self.comx   x offset from center, 2D array with scan dimensions
        self.comy   y offset from center, 2D array with scan dimensions
        self.com    magnitude of shifts from x, y images
        """

        self.get_cbedsum()
        self.get_center()
        self.comx = np.zeros((self.scanx, self.scany))
        self.comy = np.zeros((self.scanx, self.scany))
        self.com = np.zeros((self.scanx, self.scany))

        xgrid, ygrid = np.meshgrid(range(self.cbedx), range(self.cbedy))
        xgrid = xgrid-self.cbedcenterx
        ygrid = ygrid-self.cbedcentery

        for i in range(self.scanx):
            #print "scan row {}".format(i)
            for j in range(self.scany):
                tot = np.sum(self.data[:,:,i,j])

                if tot==0:
                    self.comx[i,j] = 0
                    self.comy[i,j] = 0

                else:
                    self.comx[i,j] = np.sum(self.data[:,:,i,j]*xgrid.T)/tot
                    self.comy[i,j] = np.sum(self.data[:,:,i,j]*ygrid.T)/tot
                self.com[i,j] = np.sqrt(self.comx[i,j]**2+self.comy[i,j]**2)
        return


    def get_dpc(self, angle=0):
        """
        Create a DPC image from the BF disk, by subtracting one half from
        the other.
        Halves are specified by angle input.
        Input:
        angle   integer, in degrees. Specifies at which angle to slice the 
                central disk (so unlike fixed detector where we have left/right 
                or up/down, we can slice at any angle we want)
        """

        self.dpcangle = np.pi / 180 * angle

        #helper function so we can easily get corresponding perpendicular image
        #ie if we input zero degrees, we get up/down DPC, but also dpcperp 
        #output is left/right image

        def dpc_maker(self, angle):
            angle = np.pi / 180 * angle
            if not hasattr(self, 'cbedrad'):
                self.get_center()

            outer = self.cbedrad

            #define one array based giving radius from center, and one
            #giving angle from positive x-axis in radians
            xgrid, ygrid = np.meshgrid(range(self.cbedx), range(self.cbedy))
            xgrid = xgrid-self.cbedcenterx
            ygrid = ygrid-self.cbedcentery
            radius = np.transpose(np.sqrt(ygrid**2+xgrid**2))
            anglegrid = np.transpose(np.arctan2(ygrid,xgrid))

            #Create masks defining the two halves of the BF disk
            mask1 = (radius < outer) * (anglegrid < angle) * (angle-np.pi < anglegrid)
            mask2 = (radius < outer) * (anglegrid > angle) * (angle-np.pi > anglegrid)

            #subtract the two masks from each other
            imarray = (self.data[mask1, :, :].sum(axis=0)-self.data[mask2, :, :].sum(axis=0))/np.sum(self.data)

            return imarray, outer

        self.dpc, self.dpcouter = dpc_maker(self, angle)
        self.dpcperp, self.dpcouter = dpc_maker(self, angle+90)
        self.dpcmag = np.sqrt(self.dpc**2 + self.dpcperp**2)

        return

    def get_firstmoment(self):
        """
        Generates a first-moment image: each pixel in returned image is first
        moment (in x, y) of intensity distribution of CBED at that scan pixel.
        """
        self.firstmomx = np.zeros((self.scanx, self.scany))
        self.firstmomy = np.zeros((self.scanx, self.scany))

        #coordinates by which to weight the intensity
        vx = np.arange(0,self.cbedx)
        vy = np.arange(0,self.cbedy)

        #calculate first moment at each scan pixel.
        for i in range(self.scanx):
            for j in range(self.scany):
                cbed = self.data[:,:,i,j]
                pnorm = np.sum(cbed)
                if pnorm == 0:
                    self.firstmomx[i,j] = 0
                    self.firstmomy[i,j] = 0
                else:
                    self.firstmomx[i,j] = np.sum(vx * np.sum(cbed, axis=0), axis=None)/pnorm
                    self.firstmomy[i,j] = np.sum(vy * np.sum(cbed, axis=1), axis=None)/pnorm
        #combine components. Can easily also output a phase map
        self.firstmommag = np.sqrt(self.firstmomx**2 + self.firstmomy**2)

        return

    def get_secondmoment(self):
        """
        Generates a second-moment image: each pixel in returned image is first
        moment (in x, y) of intensity distribution of CBED at that scan pixel.
        """

        self.secondmomx = np.zeros((self.scanx, self.scany))
        self.secondmomy = np.zeros((self.scanx, self.scany))

        if not hasattr(self, 'firstmomx'):
                self.get_firstmoment()

        #coordinates by which to weight intensity
        vx = np.arange(0,self.cbedx)
        vy = np.arange(0,self.cbedy)        

        #calculate second moment at each pixel using result of first moment
        for i in range(self.scanx):
            for j in range(self.scany):
                cbed = self.data[:,:,i,j]
                pnorm = np.sum(cbed)
                if pnorm == 0:
                    self.secondmomx[i,j] = 0
                    self.secondmomy[i,j] = 0
                else:
                    self.secondmomx[i,j] = np.sum((vx - self.firstmomx[i,j])**2 
                                        * np.sum(cbed, axis=0), axis=None)/pnorm
                    self.secondmomy[i,j] = np.sum((vy - self.firstmomy[i,j])**2
                                        * np.sum(cbed, axis=1), axis=None)/pnorm

        self.secondmommag = np.sqrt(self.secondmomx**2 + self.secondmomy**2)

        return

    def make_stack(self, radius=0.8):
     #Initialize imstack object from RigidRegistraion using BF CBED region
        if not hasattr(self, 'cbedrad'):
            self.get_center()
        self.tcmask = self.make_mask(radius = radius)
        stack = util.make_stack(np.rollaxis(np.reshape(self.data, 
                               (self.cbedx*self.cbedy, self.scanx, self.scany)),
                               0, 3)[:,:,(self.tcmask).flatten()])
        self.stack = stack.imstack
        return 

    def get_tcBF(self, radius = 0.8, expand=4, n = 0, correlationType='cc', 
                 findMaxima = 'pixel', correction = False, threshold=8, show=False, crop=True):
        """
        Function to get tilt-corrected bright field image by cross correlating
        images from each BF CBED pixel

        Uses RigidRegistration code by Ben Savitzky:
        https://github.com/bsavitzky/rigidRegistration/
        
        Inputs: 
        radius      fraction of cbedrad to use for the reconstructed image
        expand      amount by which to upsample (1 scan pixel becomes 4x4 default)
        n           Integer, describing falloff of low-pass filter. if zero, 
                    no filtering used
        correlationType passed to findImageShifts; cc, pc, or mc. cc is regular 
                        cross-correlation, pc is phase correlation, mc is mutual
                        correlation. See Ben's publication for details.
        findMaxima  Method/precision with which to find maximum in cross-
                    correlations. Pixel presicion is usually sufficient, with 
                    subpixel precision coming from averages of the many images in
                    our BF disk/stack. Other options are "gf" for Gaussian
                    fitting of the cross correlation peak, or "com" for center of
                    mass.
        show        Boolean: with True some intermediate images are output as 
                    well as verbose correlations
        crop        Boolean .Crops regions of wraparound from image shifting. 
                    There shouldn't usually be a reason to turn this off.

        """

        self.data = self.data[:,:,:2*(self.scanx//2), :2*(self.scany//2)]
        self.cbedx, self.cbedy, self.scanx, self.scany = self.data.shape

        #Find the center, and define the region of CBED pixels used for the image
        if not hasattr(self, 'cbedrad'):
            self.get_center()
        self.tcmask = self.make_mask(radius = radius)
        self.bfreconrad = radius*self.cbedrad

        #Initialize imstack object from RigidRegistraion using BF CBED region
        stack = util.make_stack(np.swapaxes(np.reshape(self.data, 
                               (self.cbedx*self.cbedy, self.scanx, self.scany)),
                               0, 2)[:,:,(self.tcmask).flatten()])

        #calculate all FFTs. The code applies sin^2 window in real space
        stack.getFFTs()

        #Apply low-pass filter, if desired
        if n==0:
            stack.makeFourierMask(mask='none')
        else:
            stack.makeFourierMask(mask='lowpass', n=n)

        #Save the dose, for easy comparison to other imaging methods
        self.bfrecondose = np.sum(stack.imstack, axis=None)

        #Get all the cross correlations and find the maxima by method
        #specified in function call
        stack.findImageShifts(correlationType=correlationType, 
                              findMaxima=findMaxima, verbose=False)
        
        if show:
            util.show_sample_ccs(stack, correlationType)

        #We are not using any corrections to the shift matrix here, but check
        #the rigidregistration code because this can be useful for low-dose data
        #I've in the past discarded shifts that are outliers of the full ensemble
        #of entries in the Xij, Yij matrix
        self.Rij_mask = np.ones((stack.nz,stack.nz))

        #Find the average shifts
        stack.get_imshifts()

        #assign the shift matrices to attributes of PADdata for easy reference
        self.xshiftmatrix = stack.X_ij
        self.yshiftmatrix = stack.Y_ij

        if correction:
                stack.get_outliers(threshold=threshold)
                stack.make_corrected_Rij()
                self.xshiftmatrix = stack.X_ij_c
                self.yshiftmatrix = stack.Y_ij_c
                stack.get_imshifts()

        self.stack = stack.imstack

        self.tcbf = util.apply_shifts_expand(stack, expand)
        
        xshiftmap = np.copy(self.tcmask).astype('float')
        yshiftmap = np.copy(self.tcmask).astype('float')
        xshiftmap[np.nonzero(xshiftmap)] = stack.shifts_x
        yshiftmap[np.nonzero(yshiftmap)] = stack.shifts_y
        self.xshiftmap = xshiftmap
        self.yshiftmap = yshiftmap

        if show:
            print("X shift matrix")
            util.show_map(self.xshiftmatrix)
            print("Y shift matrix")
            util.show_map(self.yshiftmatrix)
            print("X shift map")
            util.show_map(self.xshiftmap)
            print("Y shift map")
            util.show_map(self.yshiftmap)
            util.show(self.tcbf, figsize=(10,10))
        return


    def show_tcBF_arrowShifts(self, figsize=(10,10), scale=150, width=0.0035):
        fig = plt.figure(figsize=(10, 10))
        plt.quiver(self.xshiftmap,self.yshiftmap,scale=150,width=0.0035)
        plt.matshow(self.cbedsum,fignum=0)
        a=fig.gca()
        a.set_frame_on(False)
        a.set_xticks([]); a.set_yticks([])
        a.set_aspect('equal')
        plt.axis('off')
        # plt.show()
        return fig



    def PCA(self):
        """
        Code to perform PCA on the 4D dataset - looks at scan images relative
        to CBED components.

        Can give some cool results but I haven't used it much on bio images - KAS
        """

        #reshape data into 2D array, with each row being one CBED and each column
        #one scan image



        SI = self.data.reshape(self.cbedx*self.cbedy, self.scanx*self.scany)
        meanim = np.mean(SI, axis=1)
        meancbed = np.mean(SI, axis=0)

        gain = 1

        #normalization factors calculation

        g = 1/np.sqrt(meanim)
        h = 1/np.sqrt(meancbed+meancbed**2*2*gain**2)

        gg = g/np.sum(g, axis=None)
        hh = h/np.sum(h, axis=None)

        SI = (hh*(gg*SI.T).T)

        #perform SVD using scipy.linalg.svd
        print("doing singular value decomposition")
        u,s,v = svd(SI, full_matrices=False)

        #recombine to get images per component

        scoresweight = np.dot(np.diag(np.sqrt(1/gg)),np.dot(u, np.diag(s)))
        loadsweight = np.dot(np.dot(np.diag(s), v), np.diag(np.sqrt(1/hh)))

        #Sscores are CBEDs, loads are scan images.
        self.scores = scoresweight.reshape((self.cbedx, self.cbedy, -1), order ='F')
        self.loads = np.swapaxes(loadsweight.reshape((-1, self.scany, self.scanx), order='F'), 0, 2)

        return



    # def subtract_background_for_one(self, threshold=20):
    #     """
    #     Removes intensity offset of each CBED pattern. Iterating over scan 
    #     pixels, histogram is fit to Gaussian near zero. The center of the
    #     distribution is subtracted.

    #     Inputs: 
    #     threshold defines maxiumum intensity up to which data histogram is fit

    #     Returns:
    #     Array with data at each scan pixel shifted to make them have the same
    #     zero value
    #     """

    #     i=self.scanx//2
    #     j=self.scany//2

    #     def gauss1d(x, amplitude, x0, sigma_x, offset):
    #         # Returns result as a 1D array that can be passed to 
    #         # scipy.optimize.curve_fit
    #         x0 = float(x0)
    #         g = offset+amplitude*np.exp(-1/(2*sigma_x**2)*(x-x0)**2)
    #         return g

    #         self.background = np.zeros((self.scanx, self.scany)

    #             try:
    #                 try:
    #                     #Fit the data below the threshold to a gaussian
    #                     hist, bins = np.histogram(self.data[:,:,i,j][self.data[:,:,i,j]<threshold], bins=128)
    #                     popt, pcov = curve_fit(gauss1d, bins[:-1], hist, p0=[np.max(hist), 
    #                                            np.mean(self.data[:,:,i,j][self.data[:,:,i,j]<threshold]), 
    #                                            np.std(self.data[:,:,i,j][self.data[:,:,i,j]<threshold]), 0])
    #                     #subtract the offset of the data
    #                     self.data[:,:,i,j] = self.data[:,:,i,j] - popt[1]
    #                     self.background[i,j] = popt[1]
    #                 except:
    #                     #If this doesn't work, try a higher threshold
    #                     hist, bins = np.histogram(self.data[:,:,i,j][self.data[:,:,i,j]<100], bins=128)
    #                     popt, pcov = curve_fit(gauss1d, bins[:-1], hist, p0=[np.max(hist), 
    #                                            np.mean(self.data[:,:,i,j][self.data[:,:,i,j]<10]), 
    #                                            np.std(self.data[:,:,i,j][self.data[:,:,i,j]<10]), 0])
    #                     self.data[:,:,i,j] = self.data[:,:,i,j] - popt[1]
    #                     self.background[i,j] = popt[1]
    #             except:
    #                 #If this doesn't work, forget the fitting and use the mean
    #                 print("fitting failed at pixel {}, {}; using mean of data < 10".format(i,j))
    #                 self.data[:,:,i,j] = (self.data[:,:,i,j] 
    #                                      - np.mean(self.data[:,:,i,j][self.data[:,:,i,j]<10]))
    #                 self.background[i,j] = np.mean(self.data[:,:,i,j][self.data[:,:,i,j]<10])       
    #     return


