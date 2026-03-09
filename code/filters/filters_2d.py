"""FROM SIHAO CHENG"""
import numpy as np
import torch
import matplotlib.pyplot as plt


class FiltersSet(object):
    def __init__(self, M, N, J, L):
        self.M = M
        self.N = N
        self.J = J
        self.L = L

    # Morlet Wavelets
    def generate_morlet(self, if_save=False, save_dir=None, precision='single'):
        if precision=='double':
            psi = torch.zeros((self.J, self.L, self.M, self.N), dtype=torch.float64)
        if precision=='single':
            psi = torch.zeros((self.J, self.L, self.M, self.N), dtype=torch.float32)

        for j in range(self.J):
            for theta in range(self.L):
                wavelet = self.morlet_2d(
                    M=self.M,
                    N=self.N,
                    sigma = 0.7 * 2**j,#0.7 * 2**j,
                    theta = (int(self.L-self.L/2-1)-theta) * np.pi / self.L,
                    xi= 1.4* 3.0 / 4.0 * np.pi /2**j,#1.4 * 
                    slant=4.0/self.L,
                )
                wavelet_Fourier = np.fft.fft2(wavelet)
                wavelet_Fourier[0,0] = 0
                if precision=='double':
                    psi[j, theta] = torch.from_numpy(wavelet_Fourier.real)#/np.sqrt(112.25202*2**(-2*j))
                if precision=='single':
                    psi[j, theta] = torch.from_numpy(wavelet_Fourier.real.astype(np.float32))#/np.sqrt(112.25202*2**(-2*j))

        #high frequency completion
        #psi = torch.cat([torch.zeros_like(psi)[:1],psi]) #(J+1,L,M,N)

        psi_high = []
        
        theta = [0,1,3]
        signs = torch.stack([torch.ones((self.M,self.N))*(-1)**torch.arange(self.N),
                           torch.ones((self.M,self.N))*(-1)**torch.arange(self.M)[:,None]*(-1)**torch.arange(self.N),
                           torch.ones((self.M,self.N))*(-1)**torch.arange(self.M)[:,None]]).numpy() #(3,M,N)

        for index in range(len(theta)):
          wavelet = self.HighFreqs_2d(
                    M=self.M,
                    N=self.N,
                    sigma= 0.8,
                    theta=(int(self.L-self.L/2-1)-theta[index]) * np.pi / self.L,
                    slant=4.0/self.L,
                )
          #sign
          wavelet = wavelet*signs[index]
          wavelet_Fourier = np.fft.fft2(wavelet)


          #wavelet_Fourier[0,0] = 0
          if precision=='double':
              psi_high.append(torch.from_numpy(wavelet_Fourier))#/np.sqrt(128.24548))
          if precision=='single':
               psi_high.append( torch.from_numpy(wavelet_Fourier).to(torch.complex64))#/np.sqrt(128.24548))
            

        psi_high =torch.stack(psi_high,dim=0) #(5,M,N)

        #Phi Filter
        if precision=='double':
            phi = torch.from_numpy(
                self.gabor_2d_mycode(self.M, self.N, 0.8 * 2**(self.J-1), 0, 0).real
            ) * (self.M * self.N)**0.5#/np.sqrt(2.004822)
        if precision=='single':
            phi = torch.from_numpy(
                self.gabor_2d_mycode(self.M, self.N, 0.8 * 2**(self.J-1), 0, 0).real.astype(np.float32)
            ) * (self.M * self.N)**0.5#/np.sqrt(2.004822)

        filters_set = {'psi':psi, 'psi_high':psi_high, 'phi':phi/torch.abs(phi).sum()}
        if if_save:
            np.save(
                save_dir + 'filters_set_mycode_M' + str(self.M) + 'N' + str(self.N)
                + 'J' + str(self.J) + 'L' + str(self.L) + '_' + precision + '.npy',
                np.array([{'filters_set': filters_set}])
            )
        return filters_set
    
    def HighFreqs_2d(self, M, N, sigma, theta, slant=0.5, offset=0, fft_shift=False):
        wv_modulus = self.gabor_2d_mycode(M, N, sigma, theta, 0, slant, offset, fft_shift)
        return wv_modulus

    def morlet_2d(self, M, N, sigma, theta, xi, slant=0.5, offset=0, fft_shift=False):
        """
            (from kymatio package)
            Computes a 2D Morlet filter.
            A Morlet filter is the sum of a Gabor filter and a low-pass filter
            to ensure that the sum has exactly zero mean in the temporal domain.
            It is defined by the following formula in space:
            psi(u) = g_{sigma}(u) (e^(i xi^T u) - beta)
            where g_{sigma} is a Gaussian envelope, xi is a frequency and beta is
            the cancelling parameter.
            Parameters
            ----------
            M, N : int
                spatial sizes
            sigma : float
                bandwidth parameter
            xi : float
                central frequency (in [0, 1])
            theta : float
                angle in [0, pi]
            slant : float, optional
                parameter which guides the elipsoidal shape of the morlet
            offset : int, optional
                offset by which the signal starts
            fft_shift : boolean
                if true, shift the signal in a numpy style
            Returns
            -------
            morlet_fft : ndarray
                numpy array of size (M, N)
        """
        wv = self.gabor_2d_mycode(M, N, sigma, theta, xi, slant, offset, fft_shift)
        wv_modulus = self.gabor_2d_mycode(M, N, sigma, theta, 0, slant, offset, fft_shift)
        K = np.sum(wv) / np.sum(wv_modulus)

        mor = wv - K * wv_modulus
        return mor

    def gabor_2d_mycode(self, M, N, sigma, theta, xi, slant=1.0, offset=0, fft_shift=False):
        """
            (partly from kymatio package)
            Computes a 2D Gabor filter.
            A Gabor filter is defined by the following formula in space:
            psi(u) = g_{sigma}(u) e^(i xi^T u)
            where g_{sigma} is a Gaussian envelope and xi is a frequency.
            Parameters
            ----------
            M, N : int
                spatial sizes
            sigma : float
                bandwidth parameter
            xi : float
                central frequency (in [0, 1])
            theta : float
                angle in [0, pi]
            slant : float, optional
                parameter which guides the elipsoidal shape of the morlet
            offset : int, optional
                offset by which the signal starts
            fft_shift : boolean
                if true, shift the signal in a numpy style
            Returns
            -------
            morlet_fft : ndarray
                numpy array of size (M, N)
        """
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], np.float64)
        R_inv = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]], np.float64)
        D = np.array([[1, 0], [0, slant * slant]])
        curv = np.matmul(R, np.matmul(D, R_inv)) / ( 2 * sigma * sigma)

        gab = np.zeros((M, N), np.complex128)
        xx = np.empty((2,2, M, N))
        yy = np.empty((2,2, M, N))

        for ii, ex in enumerate([-1, 0]):
            for jj, ey in enumerate([-1, 0]):
                xx[ii,jj], yy[ii,jj] = np.mgrid[
                    offset + ex * M : offset + M + ex * M,
                    offset + ey * N : offset + N + ey * N
                ]

        arg = -(curv[0, 0] * xx * xx + (curv[0, 1] + curv[1, 0]) * xx * yy + curv[1, 1] * yy * yy) +\
            1.j * (xx * xi * np.cos(theta) + yy * xi * np.sin(theta))
        gab = np.exp(arg).sum((0,1))

        norm_factor = 2 * np.pi * sigma * sigma / slant
        gab = gab / norm_factor

        if fft_shift:
            gab = np.fft.fftshift(gab, axes=(0, 1))
        return gab
