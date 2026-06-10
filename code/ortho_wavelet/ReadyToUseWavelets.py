import numpy as np
import torch
from .Wavelet import Wavelet

def DefineWavelet(Family,m=None,device='cpu',mode='Periodic'):
    """Create Wavelet Object

    Parameters:
    Family (string): 'Db','BM','Haar'
    m (int): Which Wavelet in the family
    device (string): 'cpu' or 'cuda'
    mode (string):  padding, "Symmetric" or "Periodic" border conditions 
    
    Returns:
        Wavelet

    """
    if Family == 'Haar':
        if m is not None:
            print('There is only one Haar Wavelet')
        else:
          return(Haar_wavelets(device,mode))
    elif Family == 'Db':
        return(Db_wavelets(m,device,mode))
    elif Family == 'Sym':
        return(Sym_wavelets(m,device,mode))
    elif Family == 'BM':
        if mode == 'Periodic':
            print('mode switched to Symmetric')
        return(BM_wavelets(m,device='cpu'))
    
    else:
        print('Family is not implemented yet')

"""Haar_Wavelet"""
def Haar_wavelets(device = 'cuda',mode = 'Periodic'):
    """Haar Wavelet

    Parameters:
    device (string): 'cpu' or 'cuda'
    mode (string):  padding, "Symmetric" or "Periodic" border conditions
    
    Returns:
        Haar Wavelet

    """
    Haar = torch.tensor(np.array([1/np.sqrt(2),1/np.sqrt(2)]),dtype=torch.float32)[None,None]
    m1,m2 =0,1
    
    #Define the Wavelet
   
    W=Wavelet(Haar,m1,m2,mode = mode,device =device)
    return(W)

def Db_wavelets(m=2,device = 'cuda',mode = 'Periodic'):
    """Debauchies Wavelets

    Parameters:
    m (int) : Which Wavelet we would like
    device (string): 'cpu' or 'cuda'
    mode (string):  padding, "Symmetric" or "Periodic" border conditions
    
    Returns:
        Dbm Wavelet

    """
    
    if m == 2: 
      Db = torch.tensor(np.array([0.482962913145,0.836516303738,0.224143868042,-0.129409522551]),dtype=torch.float32)[None,None]
      m1,m2 =0,3
    elif m == 3 :
      Db = torch.tensor(np.array([0.332670552950,0.806891509311, 0.459877502118, -0.135011020010, -0.085441273882, 0.035226291882]),dtype=torch.float32)[None,None]
      m1,m2 =0,5
    elif m ==4 :
      Db = torch.tensor(np.array([0.230377813309, 0.714846570553, 0.630880767930, -0.027983769417, -0.187034811719, 0.030841381836, 0.032883011667, -0.010597401785]),dtype=torch.float32)[None,None]
      m1,m2 =0,7
    elif m ==10:
        Db = torch.tensor(np.array([0.026670057901, 0.188176800078,0.527201188932,0.688459039454, 0.281172343661, -0.249846424327, -0.195946274377, 0.127369340336, 0.093057364604, -0.071394147166, -0.029457536822,0.033212674059, 0.003606553567, -0.010733175483, 0.001395351747, 0.001992405295,-0.000685856695, -0.000116466855, 0.000093588670,  0.000013264203]),dtype=torch.float32)[None,None]
        m1,m2 =0,19
    elif m ==45:
        Db = [0.000000087528502, 0.000002587801411,0.000036269133572,0.000319681340506,0.001981100366811 ,0.009134051436457 ,0.032293741822651 ,0.088736055270969,
              0.189230576160889 ,  0.306707992105235 ,  0.355724313258375 ,  0.242227292059185 , -0.012322488769674 , -0.213274445567216 , -0.162902597551331 ,  0.069492679810949,
              0.177424151444634 ,  0.023729903281501 , -0.141814194205502 , -0.063876401837722 ,  0.104368814877092  , 0.073459968673990 , -0.076917150796878 , -0.068364100241049,
              0.058908504010517 ,  0.057006038384415 , -0.047097637540825 , -0.043627761028379 ,  0.038556082346835 ,  0.030550446539303 , -0.031371239228880 , -0.019180466340882,
              0.024636091077332 ,  0.010279885814356 , -0.018299108739417 , -0.004200493553835 ,  0.012541827912341 ,  0.000570272652689 , -0.007891290206410 ,  0.001033137038436,
              0.004416835029136 , -0.001432564151476 , -0.002172166791423 ,  0.001200382969930 ,  0.000895026486984 , -0.000786948814885 , -0.000271674005415 ,  0.000434522468455,
              0.000034445887229  ,-0.000199822290230  , 0.000029335057291  , 0.000077141719276 , -0.000028737815521 , -0.000023530518023 ,  0.000016054424291  , 0.000004890217790,
              -0.000006696231827  ,-0.000000119914325  , 0.000002184478703 , -0.000000462247314 , -0.000000543741477 ,  0.000000256443349 ,  0.000000089542406 , -0.000000087637965,
              -0.000000002188580 ,  0.000000021396752 , -0.000000004435457 , -0.000000003604482  , 0.000000001742010 ,  0.000000000300822 , -0.000000000390444 ,  0.000000000037464,
              0.000000000055419 , -0.000000000018496 , -0.000000000003755  , 0.000000000003351 , -0.000000000000296 , -0.000000000000321  , 0.000000000000105 ,  0.000000000000008,
              -0.000000000000011,   0.000000000000002  , 0.000000000000000 , -0.000000000000000 ,  0.000000000000000  , 0.000000000000000 , -0.000000000000000 ,  0.000000000000000,
              -0.000000000000000,0.000000000000000,]
        Db = torch.tensor(np.array(Db),dtype=torch.float32)[None,None]*np.sqrt(2)
        m1,m2 = 0,Db.shape[-1]-1

    else:
      print('m is not implemented yet')    
     
    
    
    #Define the Wavelet

    W=Wavelet(Db,m1,m2,mode = mode,device =device )
    return(W)

def Sym_wavelets(m=4,device = 'cuda',mode = 'Periodic'):
    """Debauchies Wavelets

    Parameters:
    m (int) : Which Wavelet we would like
    device (string): 'cpu' or 'cuda'
    mode (string):  padding, "Symmetric" or "Periodic" border conditions
    
    Returns:
        Dbm Wavelet

    """
    
    if m == 2: 
      Sym = torch.tensor(np.array([0.341506350946220,   0.591506350945870,   0.158493649053780,  -0.091506350945870]),dtype=torch.float32)[None,None]
      m1,m2 =0,3
    elif m == 3 :
      Sym = torch.tensor(np.array([0.235233603892700,   0.570558457917310,   0.325182500263710,  -0.095467207784260,  -0.060416104155350,   0.024908749865890]),dtype=torch.float32)[None,None]
      m1,m2 =0,5
    elif m ==4 :
      Sym = torch.tensor(np.array([ 0.022785172948000,  -0.008912350720850,  -0.070158812089500,   0.210617267102000,   0.568329121705000,   0.351869534328000,  -0.020955482562550,  -0.053574450709000]),dtype=torch.float32)[None,None]
      m1,m2 =0,7
    elif m ==10:
        Sym = torch.tensor(np.array([ -0.000324794948391,   0.000040330601499,   0.003247864189341,  -0.000568767655337  -0.014393115971929,   0.004076408391889 ,  0.035351783781144 , -0.022620386152107,  -0.025128270170302 ,  0.271406505551398,   0.544125765368726,   0.333535669214576,  -0.050120107506458,  -0.112779486159978 ,  0.008209434708171,   0.032475462301481, -0.001036181960273,  -0.006110321317045  , 0.000067622509971 ,  0.000544585223622]),dtype=torch.float32)[None,None]
        m1,m2 =0,19
    else:
      print('m is not implemented yet')    
     
    #Normalize
    Sym = Sym/(Sym**2).sum()**0.5
    
    #Define the Wavelet

    W=Wavelet(Sym,m1,m2,mode = mode,device =device)
    return(W)

def BM_wavelets(m,device = 'cuda',mode='Symmetric'):
    """Symmetric BattleLemarié Wavelets (padding is necessarly symmetric)

    Parameters:
    m (int) : Which Wavelet we would like
    device (string): 'cpu' or 'cuda'
    mode (string):  padding, "Symmetric" or "Periodic" border conditions
    
    Returns:
        Bm Wavelet

    """
    
    if m==1:
        BMT=torch.tensor(np.array([-0.000122686, -0.000224296, 0.000511636,
                        0.000923371, -0.002201945, -0.003883261, 0.009990599,
                        0.016974805, -0.051945337, -0.06910102, 0.39729643,
                        0.817645956, 0.39729643, -0.06910102, -0.051945337,
                        0.016974805, 0.009990599, -0.003883261, -0.002201945,
                        0.000923371, 0.000511636, -0.000224296, -0.000122686]
        ),dtype=torch.float32)[None,None]
        m1,m2 =11,11
    elif m==3:
        BMT=torch.tensor(np.array([ 0.000146098,-0.000232304, -0.000285414,
                        0.000462093, 0.000559952, -0.000927187, -0.001103748,
                        0.00188212, 0.002186714, -0.003882426, -0.00435384,
                        0.008201477, 0.008685294, -0.017982291, -0.017176331,
                        0.042068328, 0.032080869, -0.110036987, -0.050201753,
                        0.433923147, 0.766130398, 0.433923147, -0.050201753,
                        -0.110036987, 0.032080869, 0.042068328, -0.017176331,
                        -0.017982291, 0.008685294, 0.008201477, -0.00435384,
                        -0.003882426, 0.002186714, 0.00188212, -0.001103748,
                        -0.000927187, 0.000559952, 0.000462093, -0.000285414,
                        -0.000232304, 0.000146098]),dtype=torch.float32)[None,None]
        
        m1,m2 =20,20
    else:
        print('m = '+str(m)+' is not implemented yet')
    #Define the Wavelet
    
   
    W=Wavelet(BMT,m1,m2,mode = mode,device =device)
    return(W)
