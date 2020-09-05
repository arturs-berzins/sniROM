"""

"""
import config
import utils
from tvtk.api import tvtk, write_data
import numpy as np
import os
join = os.path.join

nn = 10000  # number of nodes
ne = 9801   # number of elements
nsd = 2     # number of space dimensions
ndf = 3     # number of degrees of freedom
nel = 4     # number of nodes forming an element

L = 10      # number of first L bases to visualize

def main():
    ## MXYZ: node coordinates
    path_mxyz = join('visualization','mxyz')
    mxyz = np.fromfile(path_mxyz, np.float64()).byteswap()
    mxyz = mxyz.reshape(nn, nsd)
    # pad missing space dimensions with zeros, because vtk expects 3D
    mxyz = np.pad(mxyz, [(0,0),(0,3-nsd)])
    
    ## MIEN: node connectivity
    path_mien = join('visualization','mien')
    mien = np.fromfile(path_mien, np.intc()).byteswap()
    mien = mien.reshape(ne, nel)
    # fortran indexing starts from 1, shift to 0 for python
    mien = mien - 1

    V = {}    
    for component in config.components:
        V[component] = utils.load_POD_V(component)
    
    for idx in range(L):
        
        ## POLYDATA
        polydata = tvtk.PolyData(points=mxyz, polys=mien)
        
        # Velocity
        velocity = V['u'][:,idx]
        velocity = velocity.reshape(nn, 2)
        #import pdb; pdb.set_trace()
        velocity = np.pad(velocity, [(0,0),(0,3-nsd)])
        polydata.point_data.add_array(velocity)
        polydata.point_data.get_array(0).name = 'velocity'
        
        # Pressure
        pressure = V['p'][:,idx]
        polydata.point_data.add_array(pressure)
        polydata.point_data.get_array(1).name = 'pressure'

        # Store in XML format
        path_root = join('visualization','bases')
        os.makedirs(path_root, exist_ok=True)
        path = join(path_root, F'basis_{idx}.vtp')
        write_data(polydata, path)

if __name__ == '__main__':
    main()
