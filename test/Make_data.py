"""
    Make test data for input function
"""
import numpy as np

# Modify
#-----------------------------------------------------
version = 1 # number just to distinguish different files

frame = 7 # number of frames (graph per file)
npart = 12 # number of particles (node per graph)
Bound_min = 0 # box buond minimum 
Bound_max = 12  # box buond maximum

output = f'/Test/data/raw/Dati{version}.txt'
# output = f'/home/Lorenzo/Test/data/raw/Dati{version}.txt' <-- example
#-----------------------------------------------------

# Leave it as it is
#-----------------------------------------------------
with open(output,'w') as fdata:
    for nframe in range(1,frame):
        fdata.write('ITEM: TIMESTEP\n')
        fdata.write(f'{nframe}\n')
        fdata.write('ITEM: NUMBER OF ATOMS\n')
        fdata.write(f'{npart}\n')
        fdata.write('ITEM: BOX BOUNDS pp pp pp\n')
        fdata.write(f'{Bound_min}   {Bound_max}\n')
        fdata.write(f'{Bound_min}   {Bound_max}\n')
        fdata.write(f'{Bound_min}   {Bound_max}\n')
        fdata.write('ITEM: ATOMS type x y z fx fy fz\n')
    
        data = np.ones(shape=(npart, 7))*nframe
        for i in data:
            fdata.write(f'{i[0]} {i[1]} {i[2]} {i[3]} {i[4]} {i[5]} {i[6]}\n')
fdata.close()
