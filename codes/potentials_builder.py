"""
MGD utilities - potential builders.

Assemble the dict of potentials (moments / sufficient statistics) passed to the SDE
samplers, selected by string keys in ``terms``: scalar, 1D, conditional 1D, and 2D.
"""

from potentials.potentials_scalar import *
from potentials.potentials_1d import *
from potentials.potentials_1d_condi import *
from potentials.potentials_2d import *


def get_scalar_potentials(terms):
    """Build a dict of scalar (pointwise) potentials selected by ``terms``.

    Recognizes ``'x1'..'x9'`` (monomials ``x**i``), ``'x_abs'`` (``|x|``) and
    ``'bimodal'``.

    Parameters
    ----------
    terms : iterable of str
        Potential keys to include.

    Returns
    -------
    dict[str, object]
        Name -> potential object.
    """

    potentials = {}

    for i in range(1,10):
        if 'x'+str(i) in terms:
            potentials['x'+str(i)] = Monomial(i)

    if 'x_abs' in terms:
        potentials['x_abs'] = Abs()

    if 'bimodal' in terms:
        potentials['bimodal'] = Bimodal()
    
    return potentials


def get_1d_potentials(terms, J, filters, Q=1, filters_Q=None, filters_Phi=None,scalar_param=None, parallel=False):
    """Build a dict of 1D potentials selected by ``terms``.

    Covers wavelet Lp-norm moments (``'L_2'..'L_10'`` and their ``'_phi'``
    low-pass variants, via ``L2p_norm`` / ``L2p1_norm``), quantile-confined scalar
    potentials, and scattering moments up to fourth order. Third/fourth-order
    imaginary terms are stubbed out (``pass``).

    NOTE: as written, ``'L_9'`` and ``'L_10'`` both assign to key ``'L_8'`` (so they
    overwrite ``L_8`` rather than adding new entries). The ``parallel`` branch also
    references an undefined ``order_terms_potentials`` and would raise if reached.

    Parameters
    ----------
    terms : iterable of str
        Potential keys to include.
    J : int
        Number of scales.
    filters : torch.Tensor
        Main filter bank (band-pass + low-pass).
    Q : int, optional
        Wavelets per octave, by default 1.
    filters_Q : torch.Tensor, optional
        Filter bank for the scattering / scalar terms; defaults to ``filters``.
    filters_Phi : torch.Tensor, optional
        Low-pass bank for the ``'_phi'`` variants.
    scalar_param : optional
        Parameters passed to the ``Scalar`` potentials.
    parallel : bool, optional
        Wrap potentials for parallel evaluation, by default False.

    Returns
    -------
    dict[str, object]
        Name -> potential object.
    """


    if filters_Q is None:
        filters_Q = filters
        Q = 1 
    
    potentials = {}

    if 'L_2' in terms:
       potentials['L_2'] = L2p_norm(1,filters)

    if 'L_1' in terms: 
        potentials['L_1'] = L2p1_norm(0,filters)
    if 'L_3' in terms:
        potentials['L_3'] = L2p1_norm(1,filters)

    if 'L_4' in terms:
        potentials['L_4'] = L2p_norm(2,filters)
    if 'L_5' in terms:
        potentials['L_5'] = L2p1_norm(2,filters)
    if 'L_6' in terms:
        potentials['L_6'] = L2p_norm(3,filters)

    if 'L_6_psi' in terms:
        potentials['L_6_psi'] = L2p_norm(3,filters_Q)

    if 'L_7' in terms:
        potentials['L_7'] = L2p1_norm(3,filters)
    if 'L_8' in terms:
        potentials['L_8'] = L2p_norm(4,filters)
    if 'L_9' in terms:
        potentials['L_8'] = L2p_norm(4,filters)
    if 'L_10' in terms:
        potentials['L_8'] = L2p_norm(5,filters)

    if 'L_1_phi' in terms:
       potentials['L_1_phi'] = L2p1_norm(0,filters_Phi)
    if 'L_2_phi' in terms:
       potentials['L_2_phi'] = L2p_norm(1,filters_Phi)
    if 'L_3_phi' in terms:
        potentials['L_3_phi'] = L2p1_norm(1,filters_Phi)
    if 'L_4_phi' in terms:
        potentials['L_4_phi'] = L2p_norm(2,filters_Phi)
    if 'L_5_phi' in terms:
        potentials['L_5_phi'] = L2p1_norm(2,filters_Phi)
    if 'L_6_phi' in terms:
        potentials['L_6_phi'] = L2p_norm(3,filters_Phi)
    if 'L_7_phi' in terms:
        potentials['L_7_phi'] = L2p1_norm(3,filters_Phi)
    if 'L_8_phi' in terms:
        potentials['L_8_phi'] = L2p_norm(4,filters_Phi)

    # capture low pass 
    if 'L_2_lowpass' in terms:
        potentials['L_2_lowpass'] = L2p_norm(1,filters[:,-1:,:])

    if 'L_4_lowpass' in terms: 
        potentials['L_4_lowpass'] = L2p_norm(2,filters[:,-1:,:])

    if 'Scalar_phi_quantile_confine' in terms:
        potentials['Scalar_phi_quantile_confine'] =Scalar(filters_Phi,scalar_param=scalar_param,quantiles = True,confine=True)

    if 'Scalar_morlet_quantile_confine' in terms:
        potentials['Scalar_morlet_quantile_confine'] =Scalar(filters,scalar_param=scalar_param,quantiles = True,confine=True)

    if 'Scalar_psi_quantile_confine' in terms:
        potentials['Scalar_psi_quantile_confine'] =Scalar(filters_Q,scalar_param=scalar_param,quantiles = True,confine=True)

    # potential to fit wavelet coefficients histogram 
    # generalized gaussian k regions 
    if 'Scalar_psi_gaussianK' in terms:
        potentials['Scalar_psi_gaussianK'] = Scalar_GGD_KRegion(filters_Q)

    if 'Scalar_morlet_gaussianK' in terms:
        potentials['Scalar_morlet_gaussianK'] = Scalar_GGD_KRegion(filters)



    """
    # Gaussian and Power Law tails 
    if 'Scalar_psi_windows' in terms: 
        potentials['Scalar_psi_windows'] = Scalar_GGD_GGD_Pow(filters_Q)

    if 'Scalar_morlet_windows' in terms: 
        potentials['Scalar_morlet_windows'] = Scalar_GGD_GGD_Pow(filters)

    # GenGamma and Gaussian:
    if 'Scalar_psi_GenGamma' in terms: 
        potentials['Scalar_psi_GenGamma'] = Scalar_GGD_GenGamma(filters_Q)

    if 'Scalar_morlet_GenGamma' in terms: 
        potentials['Scalar_morlet_GenGamma'] = Scalar_GGD_GenGamma(filters)

    # three gen gaussians 
    if 'Scalar_psi_GGG' in terms: 
        potentials['Scalar_psi_GGG'] = Scalar_GGD_GGD_GGD(filters_Q)

    if 'Scalar_morlet_GGG' in terms: 
        potentials['Scalar_morlet_GGG'] = Scalar_GGD_GGD_GGD(filters)

    """ 





    # scattering 
    if 'Scattering_First_Order' in terms:
        potentials['Scattering_First_Order'] = Scattering_First_Order_1d(filters_Q[:,:-1])
    
    if 'Scattering_Second_Order' in terms:
        potentials['Scattering_Second_Order'] = Scattering_Second_Order_1d(filters_Q)

    # only on the distribution bulk 
    if 'Scattering_Second_Order_bulk' in terms:
        potentials['Scattering_Second_Order_bulk'] = Scattering_Second_Order_Bulk_1d(filters_Q)

    if 'Scattering_Third_Order_Real' in terms:
        potentials['Scattering_Third_Order_Real'] = Scattering_Third_Order_Real_1d(J, filters_Q)

    if 'Scattering_Third_Order_Imag' in terms:
        pass
        #potentials['Scattering_Third_Order_Imag'] = Scattering_Third_Order_Imag_1d(J, filters_Q)
    
    if 'Scattering_Fourth_Order_Real' in terms:
        potentials['Scattering_Fourth_Order_Real'] = Scattering_Fourth_Order_Real_1d(J, Q, filters, filters_Q[:,:-1])

    # Q = 1 SCATTERING 2ND ORDER 
    if 'Scattering_Second_Order_Q1' in terms:
        potentials['Scattering_Second_Order_Q1'] = Scattering_Second_Order_1d(filters)


    # Q = 1 SCATTERING 4TH ORDER 
    if 'Scattering_Fourth_Order_Real_Q1' in terms:
        potentials['Scattering_Fourth_Order_Real_Q1'] = Scattering_Fourth_Order_Real_1d(J, 1, filters, filters[:,:-1])

    if 'Scattering_Fourth_Order_Imag' in terms:
        potentials['Scattering_Fourth_Order_Imag'] = Scattering_Fourth_Order_Imag_1d(J, Q, filters, filters_Q[:,:-1])

    if 'Scattering_Fourth_Order_Mod2_Real_Q1' in terms:
        potentials['Scattering_Fourth_Order_Mod2_Real_Q1'] = Scattering_Fourth_Order_Mod2_Real_1d(J, 1, filters, filters[:,:-1])


    if 'Scattering_Fourth_Order_Imag_Q1' in terms:
        potentials['Scattering_Fourth_Order_Imag_Q1'] = Scattering_Fourth_Order_Imag_1d(J, 1, filters, filters[:,:-1])
    if 'Scattering_Fourth_Order_Mod2_Imag_Q1' in terms:
        potentials['Scattering_Fourth_Order_Mod2_Imag_Q1'] = Scattering_Fourth_Order_Mod2_Imag_1d(J, 1, filters, filters[:,:-1])


    if parallel:
        for i in range(len(order_terms_potentials)):
            order_terms_potentials[i] = Potential_Parallel(order_terms_potentials[i])#nn.DataParallel(order_terms_potentials[i])

    return potentials

def get_1d_potentials_condi(W,terms, J, filters, Q=1, filters_Q=None, filters_Phi=None,scalar_param=None, parallel=False):
    """Conditional counterpart of :func:`get_1d_potentials`.

    Same selection of 1D moments, but each potential is wrapped in
    ``Potential_Condi(..., W)`` so it is evaluated on the ``(direct, condi)`` split
    produced by the decomposition operator ``W``. Uses the direct-part slices
    ``filters[:, :2]`` / ``filters_Q[:, :Q]``.

    NOTE: the ``parallel`` branch references an undefined ``order_terms_potentials``
    and would raise if reached.

    Parameters
    ----------
    W : object
        Decomposition operator passed to each ``Potential_Condi``.
    terms, J, filters, Q, filters_Q, filters_Phi, scalar_param, parallel
        As in :func:`get_1d_potentials`.

    Returns
    -------
    dict[str, object]
        Name -> conditional potential object.
    """


    if filters_Q is None:
        filters_Q = filters
        Q = 1 
    
    filters_Q_direct = filters_Q[:,:Q]
    filters_direct = filters[:,:2]
    filters_Phi = filters_Phi
    
    potentials = {}

    if 'L_1' in terms:
       potentials['L_1'] = Potential_Condi(L2p1_norm(0,filters_direct),W,parallel =parallel )
    if 'L_2' in terms:
       potentials['L_2'] = Potential_Condi(L2p_norm(1,filters_direct),W,parallel =parallel )
    if 'L_3' in terms:
        potentials['L_3'] = Potential_Condi(L2p1_norm(1,filters_direct),W,parallel =parallel )
    if 'L_4' in terms:
        potentials['L_4'] = Potential_Condi(L2p_norm(2,filters_direct),W,parallel =parallel )
    if 'L_5' in terms:
        potentials['L_5'] = Potential_Condi(L2p1_norm(2,filters_direct),W,parallel =parallel )
    if 'L_6' in terms:
        potentials['L_6'] = Potential_Condi(L2p_norm(3,filters_direct),W,parallel =parallel )
    if 'L_7' in terms:
        potentials['L_7'] = Potential_Condi(L2p1_norm(3,filters_direct),W,parallel =parallel )
    if 'L_8' in terms:
        potentials['L_8'] = Potential_Condi(L2p_norm(4,filters_direct),W,parallel =parallel )

    if 'L_1_phi' in terms:
        potentials['L_1_phi'] = Potential_Condi(L2p1_norm(0,filters_Phi),W,parallel =parallel )
    if 'L_2_phi' in terms:
       potentials['L_2_phi'] = Potential_Condi(L2p_norm(1,filters_Phi),W,parallel =parallel )
    if 'L_3_phi' in terms:
        potentials['L_3_phi'] = Potential_Condi(L2p1_norm(1,filters_Phi),W,parallel =parallel )
    if 'L_4_phi' in terms:
        potentials['L_4_phi'] = Potential_Condi(L2p_norm(2,filters_Phi),W,parallel =parallel )
    if 'L_5_phi' in terms:
        potentials['L_5_phi'] = Potential_Condi(L2p1_norm(2,filters_Phi),W,parallel =parallel )
    if 'L_6_phi' in terms:
        potentials['L_6_phi'] = Potential_Condi(L2p_norm(3,filters_Phi),W,parallel =parallel )
    if 'L_7_phi' in terms:
        potentials['L_7_phi'] = Potential_Condi(L2p1_norm(3,filters_Phi),W,parallel =parallel )
    if 'L_8_phi' in terms:
        potentials['L_8_phi'] = Potential_Condi(L2p_norm(4,filters_Phi),W,parallel =parallel )
        
    if 'Scalar_psi_quantile_confine' in terms:
        potentials['Scalar_psi_quantile_confine'] =Potential_Condi(Scalar(filters_Q_direct,scalar_param=scalar_param,quantiles = True,confine=True),W,parallel =parallel )
    if 'Scalar_morlet_quantile_confine' in terms:
        potentials['Scalar_morlet_quantile_confine'] =Potential_Condi(Scalar(filters[:,:1],scalar_param=scalar_param,quantiles = True,confine=True),W,parallel =parallel )

    if 'Scalar_psi_generalized_gaussian' in terms:
        potentials['Scalar_psi_generalized_gaussian'] = Potential_Condi(Scalar_psi_generalized_gaussian(filters_Q_direct), W, parallel=parallel)

    if 'Scalar_morlet_generalized_gaussian' in terms:
        potentials['Scalar_morlet_generalized_gaussian'] = Potential_Condi(Scalar_Morlet_Generalized_Gaussian(filters[:,:1]), W, parallel=parallel)
    
    if 'Scattering_First_Order' in terms:
        potentials['Scattering_First_Order'] = Potential_Condi(Scattering_First_Order_1d(filters_Q_direct),W,parallel =parallel )
    
    if 'Scattering_Second_Order' in terms:
        potentials['Scattering_Second_Order'] = Potential_Condi(Scattering_Second_Order_1d(filters_Q_direct),W,parallel =parallel )

    if 'Scattering_Fourth_Order_Real' in terms:
        potentials['Scattering_Fourth_Order_Real'] = Potential_Condi(Scattering_Fourth_Order_Real_1d_Condi(J,Q,filters,filters_Q[:,:-1],include_diag =False,lite=True),W,parallel =parallel )
        
    if 'Scattering_Fourth_Order_Imag' in terms:
        pass
        #potentials['Scattering_Fourth_Order_Imag'] = Scattering_Fourth_Order_Imag_1d(J, Q, filters, filters_Q[:,:-1])

    if parallel:
        for i in range(len(order_terms_potentials)):
            order_terms_potentials[i] = Potential_Parallel(order_terms_potentials[i])#nn.DataParallel(order_terms_potentials[i])

    return potentials

def get_2d_potentials(terms, J, L, filters, parallel=False):
    """Build a dict of 2D scattering potentials selected by ``terms``.

    Recognizes first / second / third (real) / fourth (real) order 2D scattering
    moments.

    NOTE: the ``parallel`` branch references an undefined ``order_terms_potentials``
    and would raise if reached.

    Parameters
    ----------
    terms : iterable of str
        Potential keys to include.
    J, L : int
        Number of scales and orientations.
    filters : torch.Tensor
        2D filter bank.
    parallel : bool, optional
        Wrap potentials for parallel evaluation, by default False.

    Returns
    -------
    dict[str, object]
        Name -> potential object.
    """

    potentials = {}

    if 'Scattering_First_Order' in terms:
        potentials['Scattering_First_Order'] = Scattering_First_Order_2d(filters)
    
    if 'Scattering_Second_Order' in terms:
        potentials['Scattering_Second_Order'] = Scattering_Second_Order_2d(filters)

    if 'Scattering_Third_Order_Real' in terms:
        potentials['Scattering_Third_Order_Real'] = Scattering_Third_Order_Real_2d(J, L, filters)

    if 'Scattering_Fourth_Order_Real' in terms:
        potentials['Scattering_Fourth_Order_Real'] = Scattering_Fourth_Order_Real_2d(J, L, filters)

    if parallel:
        for i in range(len(order_terms_potentials)):
            potentials[i] = Potential_Parallel(potentials[i])

    return potentials