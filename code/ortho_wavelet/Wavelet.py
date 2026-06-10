import numpy as np
import torch
import torch.nn as nn



class Wavelet(nn.Module):
    def __init__(self, BMT, m1, m2,mode = 'Periodic',device = 'cuda'):
        
        """Create the Wavelet Object from a low pass filter
        
        Parameters:
        BMT (tensor): low pass filter h ~(1,1,m1+m2+1) (h(-m1),...,h(-1),h(0),h(1),....,h(m2))
        m1 (int): number of coefficients before h(0) in BMT
        m2 (int): number of coefficients after h(0) in BMT
        mode (string) : padding, "Symmetric" or "Periodic" border conditions
    
        """
        super().__init__()
        self.m1 = m1
        self.m2 = m2
        self.register_buffer('dec_lo', self.bmt_h(BMT).to(device))
        self.register_buffer('dec_hi', self.bmt_g(BMT).to(device))
        self.register_buffer('rec_lo', self.bmt_h_tilde(BMT).to(device))
        self.register_buffer('rec_hi', self.bmt_g_tilde(BMT).to(device))
        #self.dec_lo = self.bmt_h(BMT)
        #self.dec_hi = self.bmt_g(BMT)
        #self.rec_lo = self.bmt_h_tilde(BMT)
        #self.rec_hi = self.bmt_g_tilde(BMT)
        
        if mode == 'Symmetric':
            self.pad_reconstruct = self.pad_reconstruct_symmetric
            self.pad = self.pad_symmetric
        elif mode == 'Periodic':
            self.pad_reconstruct = self.pad_reconstruct_periodic
            self.pad = self.pad_periodic
        
    #Construction of the high pass and reconstruction orthogonal filters from the low pass filter
    def bmt_h(self,BMT):
        return (torch.flip(BMT, (-1,)))

    def bmt_h_tilde(self,BMT):
        return (BMT)

    def bmt_g_tilde(self,BMT):
        minus = torch.tensor([(-1) ** (j) for j in range(self.m1 + self.m2 + 1)],device=BMT.device)
        return (self.bmt_h(BMT) * minus)

    def bmt_g(self,BMT):
        return (torch.flip(self.bmt_g_tilde(BMT), (-1,)))
        
   
    #1d padding
    def pad_symmetric(self, x, n1, n2):
        # Reflective padding
        # x  ~(batch,k,N)
        # x_pad ~(batch,k,n1+N+n2)
        # ...x_2 x_1｜x_0 x_1 ... x_N-1 ｜x_N-2 x_N-3 .... x_1 ｜ x_0 x_1
        #  n1                   N              n2
        N = x.shape[2]
        k = (n1 + N + n2) // (2 * N - 2)
        r = ((2 * N - 2) - n1) % (2 * N - 2)

        x_rep = torch.concat([x, torch.flip(x[:,:, 1:-1], (2,))], axis=2)  # x_0 x_1 ... x_N-1 ｜x_N-2 x_N-3 ....x_1
        x_pad = torch.tile(x_rep, (k + 2,))  # ~(...,2*(N-2)*(k+2))
        x_pad = x_pad[:,:, r:]
        x_pad = x_pad[:,:, :n1 + N + n2]

        return x_pad  # ~(...,m1+N+m2)
        
    def pad_periodic(self, x, n1, n2):
        # Periodic padding
        # x  ~(batch,k,N)
        # x_pad ~(batch,k,n1+N+n2)
        # ...x_N-2 x_N-1｜x_0 x_1 ... x_N-1 ｜x_0 x_1 ... x_N-1 ｜ x_0 x_1 ...
        #  n1                   N              n2
        N = x.shape[2]
        k = (n1 + N + n2) // N 
        r = (N  - n1) % N

        x_pad = torch.tile(x, (k + 2,))  # ~(batch,k,N*(k+2))
        x_pad = x_pad[:,:, r:] 
        x_pad = x_pad[:,:, :n1 + N + n2]

        return x_pad  # ~(...,m1+N+m2)
        
    def pad_reconstruct_periodic(self, x, n1, n2, mode):
        # x  ~(batch,k,N/2)
        # x_pad ~(batch,k,n1+N+n2)
        
        #mode is not used here

        # ...x_0｜x_0 x_2 ... x_N-2 ｜ x_N-4 .... x_2 x_0 ｜ x_0 x_2
        #  Insert 0 then ...x_2 0｜x_0 0 x_2 ... x_N-2 0 ｜x_N-2 0 x_N-4 .... x_2 0 ｜ x_0 0 x_2
        #                     n1            N                       n2

        N = x.shape[2] * 2
        k = (n1 + N + n2) // N 
        r = (N  - n1) % N 
        
        #we want to do the padding on the last axis
        dim_last=len(x.shape)-1
        x_rep = torch.repeat_interleave(x, 2, dim=dim_last) #x_0 x_0 x_2 x_2 ..........x_N-2 x_N-2
        x_rep[:,:, 1::2] = torch.zeros_like(x_rep[:,:, 1::2])  # x_0 0 x_2 ... x_N-2 0

        x_pad = torch.tile(x_rep, (k + 2,))  # ~(batch,k,N*(k+2))
        x_pad = x_pad[:,:, r:]
        x_pad = x_pad[:,:, :n1 + N + n2]

        return x_pad  # ~(batch,k,m1+N+m2)

    def pad_reconstruct_symmetric(self, x, n1, n2, mode):
        # x  ~(batch,k,N/2)
        # x_pad ~(batch,k,n1+N+n2)

        """Central 0, Miror N/2-1  padding"""  # mode=0
        # ...x_2｜x_0 x_2 ... x_N-2 ｜x_N-2 x_N-4 .... x_2 ｜ x_0 x_2

        """Miror 0, Central N/2-1  padding"""  # mode=1
        # x_pad ~(batch,k,n1+N+n2)

        # ...x_0｜x_0 x_2 ... x_N-2 ｜ x_N-4 .... x_2 x_0 ｜ x_0 x_2
        #  Insert 0 then ...x_2 0｜x_0 0 x_2 ... x_N-2 0 ｜x_N-2 0 x_N-4 .... x_2 0 ｜ x_0 0 x_2
        #                     n1            N                       n2

        N = x.shape[2] * 2
        k = (n1 + N + n2) // (2 * N - 2)
        r = ((2 * N - 2) - n1) % (2 * N - 2)

        #we want to do the padding on the last axis
        dim_last=len(x.shape)-1
        if mode == 0:
            x_rep = torch.concat([x, torch.flip(x[:,:, 1:], (2,))], axis=dim_last)  # x_0 x_2 ... x_N-2 ｜x_N-2 x_N-3 ....x_2
            x_rep = torch.repeat_interleave(x_rep, 2, dim=dim_last)  # x_0 x_0 x_2 ... x_N-2 x_N-2｜x_N-2 x_N-2  ....x_2 x_2
        elif mode == 1:
            x_rep = torch.concat([x, torch.flip(x[:,:, :-1], (2,))], axis=dim_last)  # x_0 x_2 ... x_N-2 ｜x_N-4 x_N-3 ....x_0
            x_rep = torch.repeat_interleave(x_rep, 2, dim=dim_last)  # x_0 x_0 x_2 ... x_N-2 x_N-2｜x_N-2 x_N-2  ....x_0 x_0
        else:
            print('mode=0 or mode=1')
        x_rep[:,:, 1::2] = torch.zeros_like(x_rep[:,:, 1::2])  # x_0 0 x_2 ... x_N-2 0｜x_N-2 0  ....x_2 0

        x_pad = torch.tile(x_rep, (k + 2,))  # ~(...,2*(N-2)*(k+2))
        x_pad = x_pad[:,:, r:]
        x_pad = x_pad[:,:, :n1 + N + n2]

        return x_pad  # ~(...,m1+N+m2)
    
   
    def decompose(self,x):
        shape = x.shape #(...,T)
        x = x.reshape((-1,1,x.shape[-1])) #(-1,1,T)
        x_condi,x  = self.dwt(x)
        x = x.reshape(shape[:-1]+(x.shape[-1],)) #(...,T//2)
        x_condi = x_condi.reshape(shape[:-1]+(x_condi.shape[-1],)) #(...,T//2)
        return x,x_condi

    def forward(self,x,x_condi):
        shape = x.shape #(...,T//2)
        x = x.reshape((-1,1,x.shape[-1])) #(-1,1,T//2)
        x_condi = x_condi.reshape((-1,1,x.shape[-1])) #(-1,1,T//2)
        x = self.idwt(x_condi,x)
        x = x.reshape(shape[:-1]+(x.shape[-1],)) #(...,T)
        return x
       
    def dwt(self, x):
        #Direct 1d fast wavelet transform 
        # x  ~(batch,k,N) -> has to be of dim 3 for convolution to work
        # Please give N even

        device = x.device
        k = x.shape[1]
        N = x.shape[2]
        n_batch = x.shape[0]
        m1, m2 = self.m1, self.m2
        # Low Freq
        x_pad = self.pad(x.reshape(n_batch*k,1,N), m1, m2)   #~(batch*k,1,N+m1+m2)
        a = torch.nn.functional.conv1d(x_pad, self.dec_lo.to(device), padding='valid', groups=1,stride=2)  # ~(...,1,N/2)
        # High Freq
        x_pad = self.pad(x.reshape(n_batch*k,1,N), m2 - 1, m1 + 1)
        d = torch.nn.functional.conv1d(x_pad, self.dec_hi.to(device), padding='valid', groups=1,stride=2)  # ~(...1,N/2)
     
        
        return (a.reshape(n_batch,k,N//2), d.reshape(n_batch,k,N//2))

    def idwt(self, a, d):
        #Direct 1d fast inverse wavelet transform
        # Takes (a,d) with a low freq and d high freq filters
        # a,d ~(batch,k,N/2)
        device = a.device
        
        m1, m2 = self.m1, self.m2
        k = a.shape[1]
        M = a.shape[2]
        n_batch = a.shape[0]
        
        a, d = self.pad_reconstruct(a.reshape(n_batch*k,1,M), m2, m1, mode=0), self.pad_reconstruct(d.reshape(n_batch*k,1,M), m1 + 1, m2 - 1, mode=1) #(n_batch*k,1,2*M+m1+m2)
        
        x = torch.nn.functional.conv1d(a, self.rec_lo.to(device), padding='valid',
                                       groups=1) + torch.nn.functional.conv1d(d, self.rec_hi.to(device),
                                                                              padding='valid', groups=1)                                                    
        return (x.reshape(n_batch,k,M*2))
