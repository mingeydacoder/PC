# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 10:54:33 2024

@author: TDRC
"""
# Computes time derivatives for use with script coriolis.m.

import numpy as np

# ODEs of computing trajectories and momentum with curvature term.

# n = [∂u/∂t,∂v/∂t,∂x/∂t,∂y/∂t]
# a:      Radius of the Earth
# omega:  Angular velocity of the Earth

def xprim1(t,n,a,omega):
    prim = np.zeros(4)
    prim[0] = (2 * omega + n[0] / (a * np.cos(n[3]))) * np.sin(n[3]) * n[1]       # zonal direction momentum equation
    prim[1] = - (2 * omega + n[0] / (a * np.cos(n[3]))) * np.sin(n[3]) * n[0]     # meridional direction momentum equation
    prim[2] = n[0] / (a * np.cos(n[3]))                                           # zonal direction movement
    prim[3] = n[1] / a                                                            # meridional direction movement
    return prim

# ODEs of computing trajectories and momentum without curvature term.

# n = [∂u/∂t,∂v/∂t,∂x/∂t,∂y/∂t]
# a:      Radius of the Earth
# omega:  Angular velocity of the Earth
    
def xprim2(t,n,a,omega):
    prim = np.zeros(4)
    prim[0] = 2 * omega * np.sin(n[3]) * n[1]       # zonal direction momentum equation
    prim[1] = - 2 * omega * np.sin(n[3]) * n[0]     # meridional direction momentum equation
    prim[2] = n[0] / (a * np.cos(n[3]))             # zonal direction movement
    prim[3] = n[1] / a                              # meridional direction movement
    return prim