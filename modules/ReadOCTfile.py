# -*- coding: utf-8 -*-
"""
Author = JdeWit
Date: 2019-08-15
email: J.deWit-1@tudelft.nl

This is a series of functions to read a .oct files from Thorlabs OCT software
and extract the raw data as well as imaging parameters from these files.
"""

import numpy as np
import zipfile
from bs4 import BeautifulSoup

def OCTgetDataCombined(filepath,spectrumindex=0):
    '''
    This function combines the functions below and gives the header, raw data, 
    spectrum and the field of view from the filepath.
    input parameters:
        filepath: path where the data file can be found including the file name
        spectrumindex:determine which Bscan to load the data from. This is 
            especially relevant for recordings with multiple Bscans or 3D datasets.
    output parameters:
        header: header containing the metadata of the measurement
        rawdata: file with the raw data (2D) from the OCT spectrometer 
            (compensated for the camera offset)
        spectrum: spectrum of the reference arm
        FOV: field of view, list with order [FOVz,FOVx,FOVy] in the 1D and 2D 
            mode it is a list with 2 items
    '''
    directory=OCTfileOpen(filepath)
    header=OCTreadHeader(directory)
    rawdata,spectrum=OCTgetRawData(directory,header,spectrumindex)
    FOV=OCTgetFOV(header)
    return [header,rawdata,spectrum,FOV]

def OCTfileOpen(filepath):
    '''
    this function creates pointer into the zip file where the header and the 
    data can be extracted from
    '''
    zip_ref = zipfile.ZipFile(filepath,'r')
    return zip_ref

def OCTreadHeader(directory):
    '''
    this function reads the header from the xml file in the directory that 
    is given as input
    '''
    file=directory.open('Header.xml')
    header=BeautifulSoup(file,'xml')
    return header

def OCTgetRawData(directory,header,spectrumindex=0):
    ''' 
    This function obtains the raw interference data with the indicated 
    spectrum index as well as the apodization spectrum for the measurement.
    Both the raw data and the apodization spectrum are corrected with the 
    offset
    input parameters:
        directory: pointer into the zip file 
        header: header corresponding to the measurements
        spectrumindex: determine which Bscan to load the data from. This is 
            especially relevant for recordings with multiple Bscans or 3D datasets.
    output parameters:
        raw2: the interference spectrum compensated for the camera offset.
        ApodizationSpectrum: the spectrum from the reference arm compensated 
            for the camera offset
    '''    
    # get offset
    offset_obj= directory.open('data/OffsetErrors.data')
    offset = offset_obj.read()
    offset = np.frombuffer(offset,dtype=np.float32)
    
    # get raw data
    bbPixel = int(header.Ocity.Instrument.BytesPerPixel.string)
    isSigned = header.Ocity.Instrument.RawDataIsSigned.string;
    if bbPixel==2:
        if isSigned=='False':
            dtype = np.uint16
        else:
            dtype = np.int16
    BinaryToElectronCountScaling=np.double(header.Ocity.Instrument.BinaryToElectronCountScaling.string)
    
    Raw_Data_File=header.find("DataFile",Type='Raw',string="data\Spectral"+str(spectrumindex)+".data")
    if not Raw_Data_File:
        print('Error: the desired spectrum is not found')
    else: 
        size=[int(Raw_Data_File['SizeX']),int(Raw_Data_File['SizeZ'])]
        ScanRegionStart0=int(Raw_Data_File['ScanRegionStart0'])
        try: 
            NumApos=int(Raw_Data_File['ApoRegionEnd0'])
        except:
            NumApos=0
             
        raw_data_obj=directory.open('data/Spectral'+str(spectrumindex)+'.data')
        raw_data=raw_data_obj.read()
        raw=np.frombuffer(raw_data,dtype)
        raw2=np.reshape(raw,size)*BinaryToElectronCountScaling
        for i in range(size[0]):
            raw2[i,:]=raw2[i,:]-offset
                    
    # get reference spectrum
    if NumApos==0:
        ApodizationSpectrum_obj=directory.open('data/ApodizationSpectrum.data')
        ApodizationSpectrum=ApodizationSpectrum_obj.read()
        ApodizationSpectrum=np.frombuffer(ApodizationSpectrum,dtype=np.float32)-offset
    else:
        ApodizationSpectrum=np.mean(raw2[0:NumApos,:],axis=0)
            
    # select raw data that makes up the scan
    if ScanRegionStart0>0:
        raw2=raw2[ScanRegionStart0:,:]
    
    return [raw2, ApodizationSpectrum]

def OCTgetVideoImage(filepath,imtype='RGB'):
    '''
    This function takes the VideoImage which is displayed with the probe camera
    and make this into an RGB, RGBA or alpha image file. 
    input parameters:
        filepath: the path to the .oct file where the image is stored in
        imtype: shows which channels are returned. Default is RGB. Other options 
        are RGBA and alpha.
    '''
    directory=OCTfileOpen(filepath)
    probeim_obj=directory.open('data/VideoImage.data')
    probeim=probeim_obj.read()
    probeim=np.frombuffer(probeim,dtype=np.uint8)
    
    header=OCTreadHeader(directory)
    Videoim_Data_File=header.find("DataFile",Type="Colored",string="data\VideoImage.data")
    SizeX=int(Videoim_Data_File['SizeX'])
    SizeZ=int(Videoim_Data_File['SizeZ'])
    
    probeim=probeim.reshape([SizeX,SizeZ,4])
    probeim=probeim[:,:,[2,1,0,3]]
    if imtype=='RGB':
        probeim=probeim[:,:,0:3]
    elif imtype=='alpha':
        probeim=probeim[:,:,3]
    elif imtype=='RGBA':
        probeim=probeim
    else:
        print('imtype '+str(imtype)+' is not recognized. RGBA is returned.')            
    return probeim.astype(np.uint8)
    
def OCTgetPreviewImage(filepath,imtype='RGB'):
    '''
    This function takes the PreviewImage which is displayed in Thorimage
    and make this into an RGB, RGBA or alpha image file. This only works when 
    processed data is stored in Thorimage
    input parameters:
        filepath: the path to the .oct file where the image is stored in
        imtype: shows which channels are returned. Default is RGB. Other options 
        are RGBA and alpha.
    '''
    directory=OCTfileOpen(filepath)
    probeim_obj=directory.open('data/PreviewImage.data')
    probeim=probeim_obj.read()
    probeim=np.frombuffer(probeim,dtype=np.uint8)
    
    header=OCTreadHeader(directory)
    Preview_im_Data_File=header.find("DataFile",Type="Colored",string="data\PreviewImage.data")
    SizeX=int(Preview_im_Data_File['SizeX'])
    SizeZ=int(Preview_im_Data_File['SizeZ'])
    
    probeim=probeim.reshape([SizeX,SizeZ,4])
    probeim=probeim[:,:,[2,1,0,3]]
    if imtype=='RGB':
        probeim=probeim[:,:,0:3]
    elif imtype=='alpha':
        probeim=probeim[:,:,3]
    elif imtype=='RGBA':
        probeim=probeim
    else:
        print('imtype '+str(imtype)+' is not recognized. RGBA is returned.')            
    return probeim.astype(np.uint8)

def OCTgetProcessedImage(filepath):
    '''
    This function takes the Processed image (Intensity.data) which is displayed 
    in Thorimage and make this into an RGB, RGBA or alpha image file. This only
    works when processed data is stored in Thorimage.
    input parameters:
        filepath: the path to the .oct file where the image is stored in
        imtype: shows which channels are returned. Default is RGB. Other options 
        are RGBA and alpha.
    '''
    directory=OCTfileOpen(filepath)
    print(directory.namelist())
    probeim_obj = directory.open('data\Intensity.data')
    probeim=probeim_obj.read()
    probeim=np.frombuffer(probeim,dtype=np.single)
    
    header=OCTreadHeader(directory)
    mode = header.Ocity.MetaInfo.AcquisitionMode.string
    Processed_im_Data_File=header.find("DataFile",Type="Real",string="data\Intensity.data")
    SizeX=int(Processed_im_Data_File['SizeX'])
    SizeZ=int(Processed_im_Data_File['SizeZ'])
    
    if mode =='Mode3D':
        SizeY=int(Processed_im_Data_File['SizeY'])
        probeim = probeim.reshape([SizeY,SizeX,SizeZ])
    elif mode == 'Mode2D':
        probeim = probeim.reshape([SizeX,SizeZ])
        probeim = probeim.T
    else:
        probeim = 0
        print('acquisition mode not available in acquiring processed image')
        
    return probeim

def OCTgetFOV(header):
    '''
    This file extract the Field of View from the header and converts it into
    meters. The field of view is a list in order [FOVz,FOVx,FOVy]
    '''
    FOV=np.zeros(3)
    FOV[0]=np.double(header.Ocity.Image.SizeReal.SizeZ.string)
    if header.Ocity.MetaInfo.AcquisitionMode.string=="Mode2D":
        FOV[1]=np.double(header.Ocity.Image.SizeReal.SizeX.string)
        FOV=FOV[0:2]
    elif header.Ocity.MetaInfo.AcquisitionMode.string=="Mode3D":
        FOV[1]=np.double(header.Ocity.Image.SizeReal.SizeX.string)
        FOV[2]=np.double(header.Ocity.Image.SizeReal.SizeY.string)
    else:
        FOV=FOV[0:2]
    FOV=FOV*1e-3
    return FOV


