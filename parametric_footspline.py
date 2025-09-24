import numpy as np
from scipy.interpolate import CubicSpline

# [ 0.2, 0.2, 0.5, 0.4, -0.4, -0.1, -0.5, -0.5 ] right forward planted
# [ -0.6, 0.6, 0.1, 0.0, 0.0, 0.05, 0.25, 0.25 ] left forward in air
# [ -0.4, 0.4, 0.1, -0.2, -0.2, -0.5, -0.5, -0.5 ] left forward planted
# [ 0.0, 0.0, -0.05, 0.6, -0.6, -0.1, -0.25, -0.25 ] right forward in air
# [ 0.2, 0.2, 0.5, 0.4, -0.4, -0.1, -0.5, -0.5 ] right forward planted
S1 = [ 0.2, 0.2, 0.5, 0.4, -0.4, -0.1, -0.5, -0.5 ]
S2 = [ -0.6, 0.6, 0.1, 0.0, 0.0, 0.05, 0.25, 0.25 ]
S3 = [ -0.4, 0.4, 0.1, -0.2, -0.2, -0.5, -0.5, -0.5 ]
S4 = [ 0.0, 0.0, -0.05, 0.6, -0.6, -0.1, -0.25, -0.25 ]
S5 = [ 0.2, 0.2, 0.5, 0.4, -0.4, -0.1, -0.5, -0.5 ]
waypoints = np.vstack([S1, S2, S3, S4, S5])
t_nodes = np.array([0.0, 0.2, 0.5, 0.8, 1.0])

spline = CubicSpline(t_nodes, waypoints, bc_type='periodic', axis=0)
