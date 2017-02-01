from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
import batman
from copy import deepcopy
import astropy.units as u

__all__ = ['transit_model_b', 'transit_model_b_depth_t0', 'params_b']

# Initialize transit parameter objects with the properties of WASP-85
# from https://arxiv.org/abs/1508.07281
# Planet b:
params_b = batman.TransitParams()
params_b.per = 2.6556777
params_b.t0 = 2456847.472856
params_b.inc = 89.69
params_b.rp = np.sqrt(0.01870)

# a/rs = b/cosi
b = 0.047

params_b.a = b / np.cos(np.radians(params_b.inc))
params_b.ecc = 0
params_b.w = 90
params_b.u = [0.4636, 0.1846] # sdss r filter, claret 2013
params_b.limb_dark = 'quadratic'

params_b.depth_error = 0.00002
params_b.duration = 0.10816 * u.day


def transit_model_b(times, params=params_b):
    """
    Get a transit model for TRAPPIST-1b at ``times``.

    Parameters
    ----------
    times : `~numpy.ndarray`
        Times in Julian date
    params : `~batman.TransitParams`
        Transiting planet parameters

    Returns
    -------
    flux : `numpy.ndarray`
        Fluxes at each time
    """
    m = batman.TransitModel(params, times)
    model = m.light_curve(params)
    return model


def transit_model_b_depth_t0(times, depth, t0, f0=1.0):
    """
    Get a transit model for TRAPPIST-1b at ``times``.

    Parameters
    ----------
    times : `~numpy.ndarray`
        Times in Julian date
    depth : float
        Depth of transit
    t0 : float
        Mid-transit time [JD]
    Returns
    -------
    flux : `numpy.ndarray`
        Fluxes at each time
    """
    params = deepcopy(params_b)
    params.t0 = t0
    params.rp = np.sqrt(depth)
    m = batman.TransitModel(params, times)
    model = f0*m.light_curve(params)
    return model
