from scipy.stats import gmean


def denormalise(gp, ft=0, ftstd=0, lt=0, st=0, ot=0):
    """
    Denormalises the given parameters using the stored normalization factors in 'pars'.
    
    Parameters
    ----------
    gp : object
        GP model instance containing normalisation parameters
    ft : 
        Function values to denormalise (default 0)
    ftstd : 
        Function standard deviation to denormalise (default 0)
    lt : 
        Lengthscale posterior mean (default 0)
    st : 
        Signal variance posterior mean (default 0)
    ot : 
        Noise variance posterior mean (default 0)

    Returns
    -------
    xtr : 
        Denormalised training inputs
    ytr : 
        Denormalised training outputs
    ft : 
        Denormalised function values
    ftstd : 
        Denormalised function standard deviations
    lt : 
        Denormalised lengthscale
    st : 
        Denormalised signal variance
    ot : 
        Denormalised noise variance
    """
    # Denormalisation
    xtr = gp.normalized_inputs * gp.input_range + gp.input_min
    ytr = gp.normalized_outputs * gp.output_scale + gp.output_mean
    ft = ft * gp.output_scale[:, None]  + gp.output_mean[:, None]
    ftstd = ftstd * gp.output_scale[:, None]
    lt = lt * gmean(gp.input_range)
    ot = ot * gp.output_scale[:, None]
    st = st * gp.output_scale[:, None]

    return xtr, ytr, ft, ftstd, lt, st, ot