"""
See predict.py
Functionalities: 
Based on https://stackoverflow.com/a/30658723
"""
import config
import utils
from tvtk.api import tvtk, write_data
import numpy as np
from os.path import join

nn = 10000  # number of nodes
ne = 9801   # number of elements
nsd = 2     # number of space dimensions
ndf = 3     # number of degrees of freedom
nel = 4     # number of nodes forming an element

dataset = 'test_Hesthaven_Ubbiali' 
types = ['truth', 'pred', 'proj', 'delta']
idxs = [0, 1, 2]  # index of sample

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
    
    ## PARAMETRS
    parameters = utils.load_parameters(dataset)

    
    for idx in idxs:
        for t in types:
            ## DATA
            path_data = join('visualization',dataset,F'{t}_{idx}') # node data
            data = np.fromfile(path_data, np.float64()).byteswap()
            data = data.reshape(nn, ndf)
            # deform mesh based on parameters
            mxyz_deformed = deform_mxyz(parameters[idx], mxyz)
            
            ## POLYDATA
            polydata = tvtk.PolyData(points=mxyz_deformed, polys=mien)
            
            # Velocity
            velocity = data[:,[0,1]]
            velocity = np.pad(velocity, [(0,0),(0,3-nsd)])
            polydata.point_data.add_array(velocity)
            polydata.point_data.get_array(0).name = 'velocity'
            
            # Pressure
            pressure = data[:,2]
            polydata.point_data.add_array(pressure)
            polydata.point_data.get_array(1).name = 'pressure'

            # Store in XML format
            path = join('visualization',dataset,F'{t}_{idx}.vtp')
            write_data(polydata, path)
    
    
def deform_mxyz(mu, mxyz_undeformed):
    mxyz_deformed = mxyz_undeformed.copy()
    
    # scale x
    mxyz_deformed[:,0] *= mu[0]
    # scale y
    mxyz_deformed[:,1] *= mu[1]
    # skew 
    mxyz_deformed[:,0] += mxyz_deformed[:,1] * np.cos(mu[2])
    mxyz_deformed[:,1] *= np.sin(mu[2])
    
    return mxyz_deformed

if __name__ == '__main__':
    main()
