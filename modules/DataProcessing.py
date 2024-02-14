#%% Import packages
import numpy as np # Numpy
import ReadOCTfile # Function package for reading Ganymede OCT files
from scipy import signal # Signal processing package
from scipy import interpolate # Interpolation package

#%% Defined functions
def OCTgetCorrelation(file_name, Chirp, dt, n_kc, apo, N_b, N_a):
    ''' 
    This function calculates correlation matrix and Doppler phase from a raw
    interferometric file. Input:
        file_name: full file name with path of raw spectra    
        Chirp: Chirp vector for interpolation, common for all files
        dt: integration time, inverse of sampling frequency
        n_kc: sample refractive index
        apo: Gaussian appodization width, use 250 as a reference
        N_b: Number of M-scan averages
        N_a: Number of time points
    Output:
        g2: magnitude correlation matrix (function of depth and time)
        g1: complex field correlation matrix (function of depth and time)
        time: correlation time lag vector
        q: Scattering wavenumber 
        z: Resolved OPL vector 
        Iz: Average intensity image vs. OPL (optical depth)
        Sk: Reference (apodized) spectrum
    '''
    Sk = 0 # Apodized spectrum
    Iz = 0 # Magnitude image
    g1 = np.zeros((N_a, 1024, N_b)) # g1 autocovariance
    g2 = np.zeros((N_a, 1024, N_b)) # g1 autocovariance    
    for i in range(N_b):
        header,rawdata,spectrum,FOV=ReadOCTfile.\
            OCTgetDataCombined(file_name,spectrumindex=i) # Read data
        # Dechirp and k-interpolate for linear k-sampling
        rawdata = np.flip(OCTSampleToK(Chirp, rawdata), axis=1) 
        spectrum = np.flip(OCTSampleToK(Chirp, spectrum))
        [ks, z, dk] = OCTGetWavenumbers() # Spectral parameters
        # Calculate interference data, apodize using Gaussian window
        [interdata,dc]=OCTApodSpectral(rawdata-spectrum,spectrum,2048,apo) 
        # Perform inverse FFT for obtaining complex field vs. OPL and time.
        [z,field_h,N_i] = OCTInvFFT(interdata, ks, dk) 
        # g1 and g2 autocovariance, DLS-OCT 
        g2[:, :, i] = OCTAutocov(np.abs(field_h), N_a, 2)
        g1[:, :, i] = OCTAutocov(field_h, N_a, 1)
        Iz = Iz + np.abs(field_h)**2 # Image intensity
        Sk = Sk + dc # Apodized spectral shape
    Iz = np.mean(Iz, axis=0) # Normalize magnitude image
    Sk = Sk[0]/np.max(Sk[0]) # Average spectral shape Sk
    time = np.linspace(0, (N_a-1)*dt, N_a) # Time lag for autocorrelation 
    q = 2*n_kc*ks[N_i] # Scattering wavenumber
    return [g1, g2, time, q, z, Iz, Sk]

def OCTGetWavenumbers(): 
    ''' 
    This function is used to obtain spectral parameters for the Ganymede setup. 
    Input:
        No input is needed
    Output:
        ks: Wavenumbers corresponding to pixels
        z: Axial OPL (depth) vector corresponding to OCT system
        dk: Pixel spectral width
    '''
    lambda_min = 792.7e-9; # Min. wavelength for Ganymede
    lambda_max = 1012.4e-9; # Max. wavelength for Ganymede   
    k_min = 2*np.pi/lambda_max; # Max. wavenumber for Ganymede
    k_max = 2*np.pi/lambda_min; # Min. wavenumber for Ganymede
    ks = np.linspace(k_min, k_max, 2048) # Wavenumber vector   
    dk = (k_max-k_min)/(2047) # Wavenumber resolution
    z = np.linspace(0, np.pi/2/dk-np.pi/dk/2048, 1024)*1e3 # OPL vector
    return [ks, z, dk]

def OCTSampleToK(Chirp, Spectra): 
    ''' 
    This function is used to interpolate raw spectra for k-linearization. 
    Input:
        Chirp: Chirp vector
        Spectra: spectra to be interpolated
    Output:
        spectra_k: k-linearized spectra
    '''    
    query = np.arange(2048) # Query points for interpolation
    f = interpolate.interp1d(Chirp, Spectra, kind ='cubic') # k-interpolation
    spectra_k = f(query) # Interpolated spectra   
    return spectra_k

def OCTApodSpectral(interdata, dc, N_pix, std): 
    ''' 
    This function is used to apodize interference spectra with Gaussian window. 
    Input:
        interdata: interference spectra
        dc: reference spectrum
        N_pix: number of axial pixels (2048 for this data)
        std: Gaussian apodization window width
    Output:
        interdata_apo: apodized interference spectra
        dc_apo: apodized reference spectrum
    '''  
    apod = signal.windows.gaussian(N_pix, std)[None,:]/dc 
    interdata_apo = interdata*apod #
    dc_apo = dc*apod # Apodize dc term        
    return [interdata_apo, dc_apo]

def OCTInvFFT(interdata, ks, dk):
    ''' 
    This function is used to compute inverse FFT of interference spectra to
    obtain dept-resolved complex OCT data. Input:
        interdata: interference spectra
        ks: wavenumbers corresponding to interference spectra
        dk: spectral resolutio, pixel depth
    Output:
        z: OPL vector 
        field_h: OCT OPL-resolved data, complex field
        N_i: index for center k (kc) for scattering q calculation
    '''  
    N = ks.size # Vector size
    if (np.remainder(N, 2)==0): # Calculations for even N
        N_i = int(N/2) # Index for central wavenumber
        z = np.linspace(0, np.pi/(2*dk)- np.pi/(dk*N), N_i) # OPL vector
        e0 = np.exp(2j*np.pi*np.arange(N)/N*ks[N_i]/dk) # Phase factor
        # Calculate inverse FFT to obtain complex OCT data
        field = np.fft.fftshift(e0[None, :]*np.fft.ifft(np.fft.ifftshift\
            (interdata, axes=1), axis=1), axes=1) # Inverse FFT   
        field_h = field[:, N_i:] # Take right hand side (positive OPL)
    else: # Calculations for odd N
        N_i = int((N+1)/2) #  # Index for central wavenumber
        z = np.linspace(0, np.pi/(2*dk)- np.pi/(dk*N*2), N_i)  # OPL vector
        e0 = np.exp(2j*np.pi*np.arange(N)/N*np.round(ks[N_i-1]/dk)) # Phase
        # Calculate inverse FFT to obtain complex OCT data
        field = np.fft.fftshift(e0[None, :]*np.fft.ifft(np.fft.ifftshift\
            (interdata, axes=1), axis=1), axes=1) # Inverse FFT   
        field_h = field[:, N_i-1:] # Take right hand side (positive OPL)
    return [z, field_h, N_i] 

def OCTAutocov(signal, N_a, opt): 
    ''' 
    This function is used to compute depth-dependent autocovariance function
    Input:
        opt: option for field or magnitude autocovariance
        signal: signal to be correlated with itself
        N_a: signal length in time (number of points)
    Output:
        g: g2 autocovariance
        field_h: OCT OPL-resolved data, complex field
        N_i: index for center k (kc) for scattering q calculation
    '''  
    if (opt==2):
        signal = signal - np.mean(signal, axis=0) # Subtract mean from signal
    N_pix = signal.shape[1] # Number of pixels
    norm = np.linspace(N_a,1,N_a) # Normalization for unbiased correlation
    g = np.zeros((N_a, N_pix)) # Preallocate correlation
    signal_FFT = np.fft.fft(np.concatenate((signal, np.zeros((N_a-1, N_pix))),\
        axis=0), axis=0) # FFT of signal
    G = np.fft.ifft(np.conj(signal_FFT)*signal_FFT, axis=0) # Autocorrelation
    G = np.real(G[:N_a, :])/norm[:, None] # Truncate autocorrelation
    g = G/G[0,:] # Normalization
    return g
