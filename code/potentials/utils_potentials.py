import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy

import torch.nn as nn

def indices_third_order(J, L):
    num_filters = J*L+1

    indices = torch.zeros((2,int(J*L*(L*(1+J)/2+1))), dtype=torch.int32)

    rows = []
    columns = []
    for row in range(num_filters-1):
        j = int(row/L)
        l = row%L

        rows.extend([row for i in range(num_filters-L*j)])
        columns.extend([i for i in range(L*j, num_filters)])

    indices[0] = torch.Tensor(rows)
    indices[1] = torch.Tensor(columns)

    return indices

def indices_fourth_order_Q(J, Q, offset=0):
    num_filters_Q = J*Q
    num_filters = J+1

    triu_indices = torch.triu_indices(row=num_filters_Q, col=num_filters_Q, offset=offset)
    
    

    axis_a = []
    axis_b = []
    axis_c = []

    for i in range(triu_indices.shape[1]):
        row = triu_indices[1,i]
        j = row//Q

        axis_a.extend([triu_indices[0,i] for j in range(num_filters-j)])
        axis_b.extend([row for i in range(num_filters-j)])
        axis_c.extend([i for i in range(j, num_filters)])

    axis_a = torch.Tensor(axis_a).to(dtype=torch.int32)
    axis_b = torch.Tensor(axis_b).to(dtype=torch.int32)
    axis_c = torch.Tensor(axis_c).to(dtype=torch.int32)

  

    indices = torch.stack([axis_a,axis_b,axis_c])
    return indices

def indices_fourth_order(J,L):
    num_filters = J*L+1

    triu_indices = torch.triu_indices(row=num_filters-1, col=num_filters-1, offset=0)

    indices = torch.zeros((3,int(L*J/12*((2*J**2+1)*L**2+(3*J*(L+3)+3)*L+6))), dtype=torch.int32)

    axis_a = []
    axis_b = []
    axis_c = []

    for i in range(triu_indices.shape[1]):
        row = triu_indices[1,i]

        j = int(row/L)
        l = row%L

        axis_a.extend([triu_indices[0,i] for j in range(num_filters-L*j)])
        axis_b.extend([row for i in range(num_filters-L*j)])
        axis_c.extend([i for i in range(L*j, num_filters)])

    indices[0] = torch.Tensor(axis_a)
    indices[1] = torch.Tensor(axis_b)
    indices[2] = torch.Tensor(axis_c)

    return indices

def abs_eps(x, epsilon=0):
    return torch.sqrt(x.real**2+x.imag**2+epsilon)