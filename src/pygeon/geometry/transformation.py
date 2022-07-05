import numpy as np

def rotation(vect):
    # Rotation matrix for a vector
    d = np.linalg.norm(vect)
    dx, dy, dz = vect

    dxy = dx*dx+dy*dy
    r0 = (dx*dx*dz/d+dy*dy)/dxy
    r1 = dx*dy*(dz/d-1)/dxy
    r2 = (dy*dy*dz/d+dx*dx)/dxy

    return np.array([[r0, r1, -dx/d], [r1, r2, -dy/d], [dx/d, dy/d, dz/d]])

def scaling(vect):
    # Scaling matrix
    return np.diag(vect)
