import numpy as np
# DEPTH Pressure to depth conversion.
#
# Computes depth in meters from pressure in decibars using Saunders 
# and Fofonoff's method. For positive pressure values, d is negative;
# this corresponds to a coordinate system in which the z axis is 
# positive upwards.
#
# Usage: d = depth(p);
#        d = depth(p,lat);
# 
# Input parameters:
#       pressure        p    [decibars]
#       latitude        lat  [degrees]
#
# Output parameters:
#       depth           d    [meters]

# References:
# Fofonoff, N.P, and R.C. Millard Jr., 1983, Algorithms
# for computation of fundamental properties of seawater, UNESCO
# Technical Papers in Marine Science, Vol. 44, 53 pp.

# (C) 2002 Rockland Oceanographic Services Inc.
# Author: Fabian Wolk
# Revision: 2002/07/05

def pres2depth(p,lat):
    x = np.sin(abs(lat)/57.29578)
    x = x**2

    # Divisor = gravity variation with latitude. Anon (1970) bulletin geodesique.
    depth = (((-1.82e-15*p+2.279e-10)*p-2.2512e-5)*p+9.72659)*p\
            /(9.780318*(1.0+(5.2788e-3+2.36e-5*x)*x) + 1.092e-6*p)

    # Define the z-coordinate positive upwards:
    depth = -depth
    return depth

   
