import torch
import numpy as np
import pywt

from torchvision.transforms import v2
from scipy.special import erfcx, erf, erfinv
from scipy.ndimage import gaussian_filter
from scipy import io


# -------- 1d signals --------

def load_SNP(nb_copy=1):
    return torch.load('../../data/data_files/SNP')[None, None].repeat(nb_copy,1,1)

# -------- 2d signals --------

def load_quijote(fact=0):

    if fact%2 != 0:
        print("Parameter 'fact' should be a power of 2.")
        return
    
    data_quijote = np.load('../../data/data_files/Quijote_Fidu_15000_256.npy')[:1000]
    
    data_quijote -= data_quijote.min()
    data_quijote = np.log(data_quijote+1e-1)

    if fact != 0:
        data_quijote_downsampled = torch.zeros((data_quijote.shape[0],1,data_quijote.shape[-2]//fact,data_quijote.shape[-1]//fact))

        for i in range(data_quijote.shape[0]):
            data_quijote_downsampled[i] = torch.from_numpy(pywt.wavedec2(data_quijote[i], 'db4', mode='periodization', level=int(np.log2(fact)))[0])

        data_quijote = data_quijote_downsampled
    else:
        data_quijote = torch.from_numpy(data_quijote)[:,None]
    
    return data_quijote#torch.log(data_quijote+1e-2)"""


def load_turbulence_2D(fact=0):

    if fact%2 != 0:
        print("Parameter 'fact' should be a power of 2.")
        return

    data_turbulence = io.loadmat("../../data/data_files/ns_randn4_N256_c1.mat")['imgs'].T[:,None]

    if fact != 0:
        DATA_ = data_turbulence

        DATA = torch.zeros(DATA_.shape[0],1,DATA_.shape[-2]//fact,DATA_.shape[-1]//fact)
        for i in range(DATA_.shape[0]):
            DATA[i,0] = torch.from_numpy(pywt.wavedec2(DATA_[i,0], 'db4', mode='periodization', level=int(np.log2(fact)))[0])
    else:
        DATA = torch.from_numpy(data_turbulence).float()
        
    return DATA