"""Registry of the nested models: {display name: --label of the run}.

Adding a model = adding one line (the label of its launcher). The order fixes the colours and the
order of the tables. Runs are found by label + seed in turbulence/experiments/ (see
logp_entropy.find_run), so nothing else depends on the config name.

    M1  Gaussian: Scattering_Second_Order_Q1 only (turbulence_final_gauss_scat2.sh, ridge 1e-6)
    M2  L_6 + scalar potentials on the Q=1 Morlet bank        (launch/turbulence_nested_M2.sh)
    M2-Db4  same, scalar potentials on the decimated Db4 DWT   (launch/turbulence_nested_M2_db4.sh)
    M3  M2 + scalar potentials on the Q=3 bank                 (launch/turbulence_nested_M3.sh)
    M4  (parked) M3 + Scat3 + Scat4
"""

MODELS = {
    'M1 Gaussian': 'turbulence_final_gauss_scat2',
    'M2 Morlet': 'turbulence_nested_M2',
    'M2 Db4 (decimated)': 'turbulence_nested_M2_db4',
    # 'M3': 'turbulence_nested_M3',
}

SEEDS = (900, 901, 902, 903, 904)
