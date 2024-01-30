import numpy as np

import porepy as pp
import pygeon as pg
from input_file import SlopeAngle, SlopeHeight, char_length, domain_extent_left, domain_extent_right


xx = np.array([domain_extent_left,0,SlopeHeight/np.tan(SlopeAngle),domain_extent_right])

def topography_func(x):
    y = np.maximum(0,np.minimum(np.tan(SlopeAngle)*x, SlopeHeight))
    return y

def bedrock_func(x):
    y = -5
    return y

def generate_mesh():
    irregular_pentagon =     [np.array([[xx[ 0], xx[ 0]], [bedrock_func   (xx[ 0]), topography_func(xx[ 0])]])]
    irregular_pentagon.append(np.array([[xx[ 0], xx[ 1]], [topography_func(xx[ 0]), topography_func(xx[ 1])]]))
    irregular_pentagon.append(np.array([[xx[ 1], xx[ 2]], [topography_func(xx[ 1]), topography_func(xx[ 2])]]))
    irregular_pentagon.append(np.array([[xx[ 2], xx[ 3]], [topography_func(xx[ 2]), topography_func(xx[ 3])]]))
    irregular_pentagon.append(np.array([[xx[ 3], xx[ 3]], [topography_func(xx[ 3]), bedrock_func   (xx[ 3])]]))
    irregular_pentagon.append(np.array([[xx[ 3], xx[-1]], [bedrock_func   (xx[ 3]), bedrock_func   (xx[-1])]]))

    domain_from_polytope = pp.Domain(polytope=irregular_pentagon)
    subdomain = pg.grid_from_domain(domain_from_polytope, char_length, as_mdg=False)
    mdg = pp.meshing.subdomains_to_mdg([subdomain])
    return mdg, subdomain

 

#iterator_vect = np.arange(np.size(xx)-1)
#irregular_pentagon = [np.array([[xx[0], xx[0]], [bedrock_func(xx[0]), topography_func(xx[0])]])]
#for i in iterator_vect:
#    line = np.array([[xx[i], xx[i+1]], [topography_func(xx[i]), topography_func(xx[i+1])]]) 
#    irregular_pentagon.append(line)
#line = np.array([[xx[-1], xx[-1]], [topography_func(xx[-1]), bedrock_func(xx[-1])]])
#irregular_pentagon.append(line)

#line = np.array([[xx[-1], xx[0]], [bedrock_func(xx[-1]), bedrock_func(xx[0])]])
#irregular_pentagon.append(line)



# Prepare the domain and its mesh
#subdomain = pp.StructuredTriangleGrid([2*N, 3*N], [2,3])
#subdomain.compute_geometry()
# Convert it to a mixed-dimensional grid
#mdg = pp.meshing.subdomains_to_mdg([subdomain])
#pp.plot_grid(subdomain, info="all", alpha=0)