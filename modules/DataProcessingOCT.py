'''
Author = JdeWit
Date: 2019-08-15
email: J.deWit-1@tudelft.nl

This set of functions can be used for processing of raw OCT data obtained 
with the ThorImage software and the THORLABS ganymede II OCT setup. 
It contains functions for Ascans, Bscans, dispersion compensation, resolution 
and SNR calculation and making dB compressed images of Bscans. 
'''
import pdb#breakpoint: pdb.set_trace() 
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.signal import hilbert


# mathematical functions used to fit
def linear(x,offset, slope):
    return offset+slope*x

def phasefie(x, offset, center, a1, a2, a3):
    return 2*np.pi*(offset + a1*(x-center)**2 + a2*(x-center)**3 + a3*(x-center)**4)

def phasefie2(x, offset, center, a1):
    return 2*np.pi*(offset + a1*(x-center)**2)

def gauss(x, ampl, x0, sigma):
    return ampl*np.exp(-(x-x0)**2/(2*sigma**2))



# Functions to calculate Ascans and Bscans
def calc_Ascan(raw0mean,spectrum,dechirp=0,popt=0,filters='none',objective='LK4',apodization='hanning'):
    '''
    This function calculates the Ascan from the raw data. The input is a single interference spectrum at each time.
    Input parameters:
        raw0mean: interference signal
        spectum: background spectrum from the reference arm
        dechirp: dechirp vector to interpolate the spectrum to a linear basis in k space
        popt: this can be a list of 5 parameters for dispersion correction according to the function phasefie.
            if the length of popt is not equal to 5, a dispersion correction is taken from the function
            getSavedDispersionCorrection, based on the filter and objective.
        filter: this is whether there is a filter inserted between the objective lens and the sample. 
            objective: which objective is used (choice between LK2 and LK4, LK4 is default)
        apodization: apodization of the spectrum. In case this is 'none', the original spectrum is used as apodization.
            The input can be 'hanning', 'hamming', or a custom apodization with the same length as the spectrum.
    output parameters:
        Ascan_z: the complex values Ascan
#    '''
    interference_signal=raw0mean-spectrum
    
    # interpolate data on linear grid in k space
    # if the dechirp data is not given it will be loaded from the file Chirp.data
    if np.size(dechirp)==1:
        dechirp = np.fromfile('modules\\Chirp.data',np.float32)
    f = interp1d(dechirp, interference_signal, kind='cubic')
    m = np.linspace(0,len(raw0mean)-1, len(raw0mean))
    int_interp = f(m)
        
    # calculate the apodization vector
    
    # if a spectrum value is close to 0 we can get problems with apodization, 
    # so we want to change these points
    # calculate the apodization vector
    
    # interpolate spectrum to linear grid
    f = interp1d(dechirp, spectrum, kind='cubic')
    spectrum_kspace =   f(m)
    # if a spectrum value is close to 0 we can get problems with apodization, 
    # so we want to change these points
    spectrum_singularity=spectrum_kspace<np.mean(spectrum_kspace)*1e-6
    if np.sum(spectrum_singularity)>0:
        indices=np.where(spectrum_singularity)[0]
        for i in range(indices.size):
            if indices[i]==0:
                spectrum_kspace[indices[i]]=spectrum_kspace[indices[i]+1]
            elif indices[i]==spectrum_kspace.size-1:
                spectrum_kspace[indices[i]]=spectrum_kspace[indices[i]-1]
            else:
                spectrum_kspace[indices[i]]=np.min([spectrum_kspace[indices[i]-1],spectrum_kspace[indices[i]+1]])
    # calculate the apodization vector based on the mode
    if np.size(apodization)==np.size(spectrum_kspace):
        apodization_vector=apodization/spectrum_kspace
    elif apodization=='hanning':
        apodization_vector=np.hanning(np.size(spectrum_kspace))/spectrum_kspace
    elif apodization=='none':
        apodization_vector=1
    elif apodization=='hamming':
        apodization_vector=np.hamming(np.size(spectrum_kspace))/spectrum_kspace
    # apply the apodization vector to the signal    
    signal_apod=int_interp*apodization_vector
    # do dispersion correction using a complex exponent with precalculated coefficients
    if np.size(popt)==5:
        popt0=popt
    else:
        popt0=getSavedDispersionCorrection(filters,objective)
    int0meanphase=signal_apod*np.exp(1j*phasefie(m, popt0[0], popt0[1], popt0[2], popt0[3], popt0[4]))
    # calculate the Ascan by taking an inverse FFT
    Ascan_z=np.fft.fftshift(np.fft.ifft(int0meanphase))
    
    return Ascan_z

def calc_Bscan(rawdata,spectrum,dechirp=0,Ascanav=1,apodization='hanning',popt=0,objective='LK4',filters='none',disp_cor=False,averaging='complex'):
    '''
    This function calculates the Bscan from the raw data with the processing 
    steps done simultaneously on a full matrix of many Ascans.
    If there is an Ascan average given, the Ascans will be averaged after they are calculated
    Input parameters:
        rawdata: interference signal 
        spectum: background spectrum from the reference arm
        dechirp: dechirp vector to interpolate the spectrum to a linear basis in k space
        Ascanav: the amount of averaging per Ascan
        apodization: apodization of the spectrum. In case this is 'none', the original spectrum is used as apodization.
            The input can be 'hanning', 'hamming', or a custom apodization with the same length as the spectrum.
        popt: this can be a list of 5 parameters for dispersion correction according to the function phasefie.
            if the length of popt is not equal to 5, a dispersion correction is taken from the function
            getSavedDispersionCorrection, based on the filter and objective.
        objective: select which objective is used to automatically select the right dispersion 
            compensation (LK4 is default, LK2 is the other option.)
        filter: this is whether there is a filter inserted between the objective lens and the sample. 
            objective: which objective is used (choice between LK2 and LK4, LK4 is default)
        averaging: way of averaging in case of Ascan averaging. Choice between ['complex','spectrum','amplitude','intensity']
            of which the first is default (same as the second).
    output parameters:
        Bscan: the complex valued Bscan
    '''
    # subtract the background spectrum
    #print(rawdata.shape)
    if (averaging=='complex')|(averaging=='spectrum'):
        if Ascanav>1:
            N=int(rawdata.shape[0]/Ascanav)
            rawdata_av=np.empty([N,int(rawdata.shape[1])])
            for i in range(N):
                rawdata_av[i,:]=np.average(rawdata[i*Ascanav:(i+1)*Ascanav,:],axis=0)
            rawdata=rawdata_av
    interference_signal=rawdata-spectrum
   # print(interference_signal.shape)
    # load the dechirp vector if it has not been given as input
    if np.size(dechirp)==1:
        dechirp=np.fromfile('modules\\Chirp.data',np.float32)
    # interpolate the signal to a linear grid in k space
    f = interp1d(dechirp, interference_signal, kind='cubic')
    m=np.linspace(0,len(dechirp)-1, len(dechirp)*2)
    int_interp = f(m)
    
    # calculate the apodization vector
    
    # interpolate spectrum to linear grid
    f = interp1d(dechirp, spectrum, kind='cubic')
    spectrum_kspace=f(m)
    # if a spectrum value is close to 0 we can get problems with apodization, 
    # so we want to change these points
    spectrum_singularity=spectrum_kspace<np.mean(spectrum_kspace)*1e-6
    if np.sum(spectrum_singularity)>0:
        indices=np.where(spectrum_singularity)[0]
        for i in range(indices.size):
            if indices[i]==0:
                spectrum_kspace[indices[i]]=spectrum_kspace[indices[i]+1]
            elif indices[i]==spectrum_kspace.size-1:
                spectrum_kspace[indices[i]]=spectrum_kspace[indices[i]-1]
            else:
                spectrum_kspace[indices[i]]=np.min([spectrum_kspace[indices[i]-1],spectrum_kspace[indices[i]+1]])
    # calculate the apodization vector based on the mode
    if np.size(apodization)==np.size(spectrum_kspace):
        apodization_vector=apodization/spectrum_kspace
    elif apodization=='hanning':
        apodization_vector=np.hanning(np.size(spectrum_kspace))/(spectrum_kspace/np.max(spectrum_kspace))
        #avoid amplifying noise
        apodization_vector[(apodization_vector>1)&(spectrum_kspace<(np.max(spectrum_kspace)*0.2))]=1
    elif apodization=='none':
        apodization_vector=1
    elif apodization=='hamming':
        apodization_vector=np.hamming(np.size(spectrum_kspace))/(spectrum_kspace/np.max(spectrum_kspace))
        #avoid amplifying noise
        apodization_vector[(apodization_vector>1)&(spectrum_kspace<(np.max(spectrum_kspace)*0.2))]=1
        
    
    # apply the apodization vector to the signal    
    signal_apod=int_interp*apodization_vector
    # apply dispersion correction with a 5th order polynomial. The coefficients 
    # can be given as option into this function or loaded based on filter and objective
    if disp_cor:
        popt = getDispersionCorrection(signal_apod,dechirp=dechirp)
    if np.size(popt)==5:
        popt0=popt
    else:
        popt0=getSavedDispersionCorrection(filters,objective)
    int0meanphase = signal_apod*np.exp(1j*phasefie(m, popt0[0], popt0[1], popt0[2], popt0[3], popt0[4]))
    Bscan = np.fft.ifft(int0meanphase,axis=1)
    
    # if Ascans are averaged the Alines in the Bscan calculated above are averaged
    # to obtain the final Bscan.
    if (averaging=='amplitude')|(averaging=='intensity'):
        Bscan=np.abs(Bscan)
        if (averaging=='intensity'):
            Bscan=Bscan**2
        if Ascanav>1:
            N=int(rawdata.shape[0]/Ascanav)
            Bscan2=np.empty([N,int(rawdata.shape[1])])
            for i in range(N):
                Bscan2[i,:]=np.average(Bscan[i*Ascanav:(i+1)*Ascanav,:],axis=0)
            Bscan=Bscan2
        if (averaging=='intensity'):
            Bscan=Bscan**(1/2) # to give still amplitude data back
    # in the output the Bscan is transposed to make the z-axis the first dimension of the output image. 
    # Moreover only one side of the image is given, This is the positive z domain 
   
    return np.matrix.transpose(Bscan[:,0:int(Bscan.shape[1]/2)])

# process data into an image
def interference_Bscan(rawdata,spectrum,dechirp=0,Ascanav=1,apodization='hanning',popt=0,objective='LK4',filters='none',disp_cor=False,averaging='complex'):
    '''
    This function calculates the Bscan from the raw data with the processing 
    steps done simultaneously on a full matrix of many Ascans.
    If there is an Ascan average given, the Ascans will be averaged after they are calculated
    Input parameters:
        rawdata: interference signal 
        spectum: background spectrum from the reference arm
        dechirp: dechirp vector to interpolate the spectrum to a linear basis in k space
        Ascanav: the amount of averaging per Ascan
        apodization: apodization of the spectrum. In case this is 'none', the original spectrum is used as apodization.
            The input can be 'hanning', 'hamming', or a custom apodization with the same length as the spectrum.
        popt: this can be a list of 5 parameters for dispersion correction according to the function phasefie.
            if the length of popt is not equal to 5, a dispersion correction is taken from the function
            getSavedDispersionCorrection, based on the filter and objective.
        objective: select which objective is used to automatically select the right dispersion 
            compensation (LK4 is default, LK2 is the other option.)
        filter: this is whether there is a filter inserted between the objective lens and the sample. 
            objective: which objective is used (choice between LK2 and LK4, LK4 is default)
        averaging: way of averaging in case of Ascan averaging. Choice between ['complex','spectrum','amplitude','intensity']
            of which the first is default (same as the second).
    output parameters:
        Bscan: the complex valued Bscan
    '''
    # subtract the background spectrum
    if (averaging=='complex')|(averaging=='spectrum'):
        if Ascanav>1:
            N=int(rawdata.shape[0]/Ascanav)
            rawdata_av=np.empty([N,int(rawdata.shape[1])])
            for i in range(N):
                rawdata_av[i,:]=np.average(rawdata[i*Ascanav:(i+1)*Ascanav,:],axis=0)
            rawdata=rawdata_av
    interference_signal=rawdata-spectrum
    
    # load the dechirp vector if it has not been given as input
    if np.size(dechirp)==1:
        dechirp=np.fromfile('modules\\Chirp.data',np.float32)
    # interpolate the signal to a linear grid in k space
    f = interp1d(dechirp, interference_signal, kind='cubic')
    m=np.linspace(0,len(dechirp)-1, len(dechirp))
    int_interp = f(m)
    
    # calculate the apodization vector
    
    # interpolate spectrum to linear grid
    f = interp1d(dechirp, spectrum, kind='cubic')
    spectrum_kspace=f(m)
    # if a spectrum value is close to 0 we can get problems with apodization, 
    # so we want to change these points
    spectrum_singularity=spectrum_kspace<np.mean(spectrum_kspace)*1e-6
    if np.sum(spectrum_singularity)>0:
        indices=np.where(spectrum_singularity)[0]
        for i in range(indices.size):
            if indices[i]==0:
                spectrum_kspace[indices[i]]=spectrum_kspace[indices[i]+1]
            elif indices[i]==spectrum_kspace.size-1:
                spectrum_kspace[indices[i]]=spectrum_kspace[indices[i]-1]
            else:
                spectrum_kspace[indices[i]]=np.min([spectrum_kspace[indices[i]-1],spectrum_kspace[indices[i]+1]])
    # calculate the apodization vector based on the mode
    if np.size(apodization)==np.size(spectrum_kspace):
        apodization_vector=apodization/spectrum_kspace
    elif apodization=='hanning':
        apodization_vector=np.hanning(np.size(spectrum_kspace))/(spectrum_kspace/np.max(spectrum_kspace))
        #avoid amplifying noise
        apodization_vector[(apodization_vector>1)&(spectrum_kspace<(np.max(spectrum_kspace)*0.2))]=1
    elif apodization=='none':
        apodization_vector=1
    elif apodization=='hamming':
        apodization_vector=np.hamming(np.size(spectrum_kspace))/(spectrum_kspace/np.max(spectrum_kspace))
        #avoid amplifying noise
        apodization_vector[(apodization_vector>1)&(spectrum_kspace<(np.max(spectrum_kspace)*0.2))]=1
        
    
    # apply the apodization vector to the signal    
    signal_apod=int_interp*apodization_vector
    # apply dispersion correction with a 5th order polynomial. The coefficients 
    # can be given as option into this function or loaded based on filter and objective
    # if disp_cor:
    #     popt = getDispersionCorrection(signal_apod,dechirp=dechirp)
    # if np.size(popt)==5:
    #     popt0=popt
    # else:
    #     popt0=getSavedDispersionCorrection(filters,objective)
    # int0meanphase=signal_apod*np.exp(1j*phasefie(m, popt0[0], popt0[1], popt0[2], popt0[3], popt0[4]))
    # Bscan=np.fft.ifft(int0meanphase,axis=1)
    
    # if Ascans are averaged the Alines in the Bscan calculated above are averaged
    # to obtain the final Bscan.
    # if (averaging=='amplitude')|(averaging=='intensity'):
    #     Bscan=np.abs(Bscan)
    #     if (averaging=='intensity'):
    #         Bscan=Bscan**2
    #     if Ascanav>1:
    #         N=int(rawdata.shape[0]/Ascanav)
    #         Bscan2=np.empty([N,int(rawdata.shape[1])])
    #         for i in range(N):
    #             Bscan2[i,:]=np.average(Bscan[i*Ascanav:(i+1)*Ascanav,:],axis=0)
    #         Bscan=Bscan2
    #     if (averaging=='intensity'):
    #         Bscan=Bscan**(1/2) # to give still amplitude data back
    # # in the output the Bscan is transposed to make the z-axis the first dimension of the output image. 
    # # Moreover only one side of the image is given, This is the positive z domain 
    return signal_apod#np.matrix.transpose(signal_apod[:,0:int(signal_apod.shape[1]/2)])
    

def plot_Bscan_image(image_data,dBlevel=80,FOV=0,title='image (dB)'):
    '''
    This function gives a plot of the Bscan image. The input is the image 
    data before log compressing. The image data may be complex values as 
    first the absolute value is taken and then the data is log compressed.
    input parameters:
        image_data: a matrix with the Bscan image before log compression. It may be complex valued or not.
        dBlevel: dB level of the image
        FOV: the field of view. This can be a list with the FOV in [z,x,y] respectively
            if FOV[1]==0 or FOV is a scalar, the image will be plot with pixels on the axis rather than mm
        title: the title of the image
        plot_no: the figure number. If it is 0, a new figure will be opened.
        
    output parameters
        fig: the figure as object
        ax: the plot as object to be manipulated further
    '''
    # log compress the data
    dB_im=20*np.log(np.abs(image_data))/np.log(10)
    dB_im=dB_im-np.max(dB_im)
    # open the figure and plot the data
    fig, ax = plt.subplots()
    clim=[-dBlevel,0]
    if np.size(FOV)==1 or FOV[1]==0:
        ax.imshow(dB_im,clim=clim)
        plt.xlabel('x (pixel)')
        plt.ylabel('z (pixel)')
    else:
        FOVmm=FOV*1e3
        extent=(0,FOVmm[1], FOVmm[0],0)
        ax.imshow(dB_im,extent=extent,clim=clim)
        plt.xlabel('x (mm)')
        plt.ylabel('z (mm)')
    plt.title(title)
    cbarticks=list(range(clim[0],clim[1]+1,10))
    cbartickslabels=list(str(i) for i in cbarticks)
    cbartickslabels[-1]=cbartickslabels[-1]+' dB'
    axins = inset_axes(ax,
                   width=0.15,  # "5%"width = 5% of parent_bbox width
                   height="100%",  # height : 50%
                   loc='center left',
                   bbox_to_anchor=(1.02, 0., 1, 1),
                   bbox_transform=ax.transAxes,
                   borderpad=0,
                       )
    a=ax.get_images()
    cbar = fig.colorbar(a[0],cax=axins, ticks=cbarticks,aspect=30)
    cbar.ax.set_yticklabels(cbartickslabels)
    return fig,ax
        
# analysis of the images/Ascans
def calc_resolution(Ascan_z,FOVz,start=0,end=-1):
    '''
    This function has as input an Ascan (which can be complex), the FOV as 
    as well as the start and end positions of the region where the fit is
    done (in pixels). In this region a gaussian fit is done around the highest position 
    and the output is the maximum value in the fit region, the amplitude of 
    the Gaussian, the position of the gaussian peak and the FWHM respectively.
    If the fit doesn't work The peak amplitude and position of the gaussian 
    are replaced by the value and position of the maximum within the fit area.
    then FWHM is set to 0, which can easily be filtered out later.
    '''
    length=np.size(Ascan_z)
    pix_size=FOVz/length
    if end==-1:
        end=length-1
    z=np.linspace(-FOVz,FOVz-pix_size,length)
    zfit=z[int(start):int(end)]
    yfit=np.abs(Ascan_z[int(start):int(end),])
    peak2=np.amax(yfit)
    sigma_guess=pix_size
    try:
        poptga,pcov = curve_fit(gauss, zfit, yfit, p0=[peak2,zfit[np.argmax(yfit)],sigma_guess])
    except RuntimeError:
            poptga=[peak2,zfit[np.argmax(yfit)],0]
    # if the peak position of the fit is outside the FOV, take just the maximum
    # and the position of the maximum as position. Set FWHM to 0
    if (poptga[1]>FOVz) or (poptga[1]<0):
            poptga=[peak2,zfit[np.argmax(yfit)],0]
    FWHM=2*np.sqrt(2*np.log(2))*np.abs(poptga[2])
    
    return [peak2,poptga[0],poptga[1],FWHM]



def calc_resolution_old(Ascan_z,FOVz,start=1024,end=-1):
    '''
    This function has as input an Ascan (which can be complex), the FOV as 
    as well as the start and end positions of the region where the fit is
    done (in pixels). In this region a gaussian fit is done around the highest position 
    and the output is the maximum value in the fit region, the amplitude of 
    the Gaussian, the position of the gaussian peak and the FWHM respectively.
    If the fit doesn't work The peak amplitude and position of the gaussian 
    are replaced by the value and position of the maximum within the fit area.
    then FWHM is set to 0, which can easily be filtered out later.
    '''
    length=np.size(Ascan_z)
    pix_size=2*FOVz/length
    if end==-1:
        end=length-1
    z=np.linspace(-FOVz,FOVz-pix_size,length)
    zfit=z[int(start):int(end)]
    yfit=np.abs(Ascan_z[int(start):int(end),])
    peak2=np.amax(yfit)
    sigma_guess=pix_size
    try:
        poptga,pcov = curve_fit(gauss, zfit, yfit, p0=[peak2,zfit[np.argmax(yfit)],sigma_guess])
    except RuntimeError:
            poptga=[peak2,zfit[np.argmax(yfit)],0]
    # if the peak position of the fit is outside the FOV, take just the maximum
    # and the position of the maximum as position. Set FWHM to 0
    if (poptga[1]>FOVz) or (poptga[1]<0):
            poptga=[peak2,zfit[np.argmax(yfit)],0]
    FWHM=2*np.sqrt(2*np.log(2))*np.abs(poptga[2])
    
    return [peak2,poptga[0],poptga[1],FWHM]

def calc_SNR(Ascan,FOVz,peak_z_position,peak_power=0):
    '''
    This function calculates the SNR of a peak at peak_z_position. The 
    variance of the noise is calculated over the area peak +50 to peak + 250 
    pixels and the SNR is obtained as 10*log(peak_power**2/var(noise area))
    '''
    if peak_power==0:
        peak_power=np.max(np.abs(Ascan))
    SNRleft=min(int((peak_z_position/FOVz*1024+1024)+50),np.size(Ascan)-3)
    SNRright=min(int((peak_z_position/FOVz*1024+1024)+250),np.size(Ascan)-1)
    SNR=10*np.log((peak_power)**2/np.var(np.abs(Ascan[SNRleft:SNRright])))/np.log(10)
    return SNR

# dispersion compensation
def getSavedDispersionCorrection(filter='none',objective='LK4'):
    '''
    This function stores the dispersion correction coefficients that can be 
    used in the function phasefie for different filters before the reflector.
    The values are calculated from measurements on July 25th 2019
    custom dispersion corrections can be added.
    '''
    if objective=='LK4':
        if filter=='none':
            popt=np.array([1.89661316e-01,9.03297521e+02,-1.26040492e-06,3.47924225e-10,2.18852572e-13])
        elif filter=='ND1':
            popt=np.array([])
        elif filter=='ND1ND2' or filter=='ND2ND1':
            popt=np.array([4.88499293e-01,8.99406248e+02,-3.01384740e-06,1.04388175e-09,4.76105923e-13])
        elif filter=='ND2':
            popt=np.array([3.40201240e-01,8.86972885e+02,-2.34963330e-06,8.34319018e-10,5.20372323e-13])
        elif filter=='biofilmbox':
            popt=[ 3.60740013e-02,  1.29951000e+03, -2.70536409e-06,  0.00000000e+00,0.00000000e+00]
        elif filter=='cuvet':
            popt=np.array([ +3.42786936e-02,  1.49950667e+03, -1.14258821e-06,  0.00000000e+00, 0.00000000e+00])
        elif filter=='cuvet2':
            popt=np.array([ 2.48173937e-03,  1.39950998e+03, -1.86117657e-07,  0.00000000e+00,  0.00000000e+00])
    elif objective=='LK2':
        if filter=='none':
            popt=[-4.96060075e-02,1.35930305e+03,1.08000468e-06,1.67645752e-09,5.97355918e-13]
        if filter == 'custom':
            popt =   np.array([-4.96060075e-02,1.35930305e+03,1.08000468e-06,1.67645752e-09,5.97355918e-13]) + \
                np.array([ +3.42786936e-02,  0, -1.14258821e-06,  0.00000000e+00, 0.00000000e+00]) -\
                np.array([1.89661316e-01,0,-1.26040492e-06,3.47924225e-10,2.18852572e-13])
                #np.array([ -0.000, 0, (3.931*1e-5)*3e-3, (9.167*1e-6)*3e-3, (-1.1332*1e-6)*3e-3])/np.pi
    elif objective == None:
        popt = [0,0,0,0,0]
    return popt

def getDispersionCorrection(int0mean, dechirp=0,fit_range=[100,500],plot=1):
    '''
     This function gives as output the coeficients used in phasefie that best
     correct for the dispersion. 
     Input parameters:
         int0mean: the net interference spectrum, for the fit to make sense, 
         there must be a single point/planar reflector in the experiment.
         dechirp: the vector that is used to interpolate the spectrum to a base
         that is linear in k
         fit_range: is the range where the phase fits are limited to. This is 
             needed because with low intensities at the edges the phase 
             information is not that trustworthy.
         plot: this determines whether the phase and the fit will be plot 
         (plot=1 means plot) it will be plot in figure 1
     if no optimal parameters are found, it will return a 0 array and print:
     'fail phasefie fit'. else it returns the parameters to be used in phasefie
     Output parameters:
         popt: fit parameters for dispersion compensation, to be used in phasefie (see above).
    '''
    # get dechirp vector if not given as input
    if np.size(dechirp)==1:
        dechirp=np.fromfile('modules\\Chirp.data',np.float32)
    # interpolate to linear grid in k space    
    length=int0mean.size
    m=np.linspace(0,length-1, length)#np.linspace(0,length-1, length)
    f = interp1d(dechirp, int0mean, kind='cubic')
    int0meaninterp = f(m)
    # obtain phase information by using the hilbert transform and unwrapping the phase
    int0analytical=hilbert(int0meaninterp)
    phaseint0=np.unwrap(np.angle(int0analytical))
    
    # do a linear fit (corresponding to a single reflector as object)
    start=fit_range[0]
    end=fit_range[1]
    fitrange=np.linspace(start, end, end-start, dtype=int)
       
    poptlin,pcov = curve_fit(linear, m[fitrange], phaseint0[fitrange], p0=[0, 1])
    if plot==1:
        plt.figure(1, figsize=(8,10))
        plt.subplot(211)
        plt.plot(m[start:end], phaseint0[start:end], '-b')
        plt.plot(m, linear(m, poptlin[0], poptlin[1]), '--r')
        plt.grid()
    # subtract the fitted linear phase to obtain the residu
    phasedev=phaseint0-linear(m, poptlin[0], poptlin[1])
    
    # fit a 4th order polynomial on the residu
    try:
        popt,pcov = curve_fit(phasefie, m[fitrange], phasedev[fitrange], p0=[-0.4, 1024, 1e-6, 0, 0], ftol=1e-20)
    except:
        # if the 4th order polynomial is not fittable, a second order polynomial is tried
        try:
            popt,pcov = curve_fit(phasefie2, m[fitrange], phasedev[fitrange], p0=[-0.4, 1024, 1e-6], ftol=1e-20)
            popt=[popt[0],popt[1],popt[2],0,0]
            print('power 2 polynome has been fit')
        except:            
            popt=np.zeros(5)
            print('fail phasefie fit')
    if plot==1:
        plt.subplot(212)
        plt.plot(m[fitrange], phasedev[fitrange], '-b')
        plt.plot(m, phasefie(m, popt[0], popt[1], popt[2], popt[3], popt[4]) , '-r')
        plt.grid()
    return popt

