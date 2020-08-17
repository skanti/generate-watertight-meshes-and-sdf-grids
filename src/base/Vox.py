import os
import struct
import numpy as np

class Vox:
    def __init__(self, dims=None, res=None, grid2world=None, sdf=None, pdf=None):
        self.dims = dims
        self.res = res
        self.grid2world = grid2world
        self.sdf = sdf

def load_vox(filename):
    assert os.path.isfile(filename), "file not found: %s" % filename

    fin = open(filename, 'rb')

    s = Vox()
    s.dims = [0,0,0]
    s.dims[0] = struct.unpack('I', fin.read(4))[0]
    s.dims[1] = struct.unpack('I', fin.read(4))[0]
    s.dims[2] = struct.unpack('I', fin.read(4))[0]
    s.res = struct.unpack('f', fin.read(4))[0]
    n_elems = s.dims[0]*s.dims[1]*s.dims[2]

    s.grid2world = struct.unpack('f'*16, fin.read(16*4))
    s.grid2world = np.asarray(s.grid2world, dtype=np.float32).reshape([4, 4], order="F")
    fin.close()

    # -> sdf 1-channel
    offset = 4*(3 + 1 + 16)
    s.sdf = np.fromfile(filename, count=n_elems, dtype=np.float32, offset=offset).reshape([1, s.dims[2], s.dims[1], s.dims[0]])
    # <-


    return s

def write_vox(filename, s):
    fout = open(filename, 'wb')
    fout.write(struct.pack('I', s.dims[0]))
    fout.write(struct.pack('I', s.dims[1]))
    fout.write(struct.pack('I', s.dims[2]))
    fout.write(struct.pack('f', s.res))
    n_elems = np.prod(s.dims)
    fout.write(struct.pack('f'*16, *s.grid2world.flatten('F')))
    fout.write(struct.pack('f'*n_elems, *s.sdf.flatten('C')))
    fout.close()

