import numpy as np
import Vox
from skimage import measure
from plyfile import (PlyData, PlyElement, make2d, PlyParseError, PlyProperty)
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--in_vox', type=str, required=True, help="filename of input vox file")
parser.add_argument('--out_ply', type=str, required=True, help="filename of output ply file")
parser.add_argument('--iso', type=float, default=0.0, help="iso surface value")


def save_as_ply(filename, verts, faces, color=None):
    if not color:
        color = list(np.random.randint(0, 255, size=3))
    verts = [tuple(v) + tuple(color) for v in verts]
    faces = [(tuple(f),) for f in faces]
    verts = np.asarray(verts, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    faces = np.asarray(faces, dtype=[('vertex_indices', 'i4', (3,))])
    objdata = PlyData([PlyElement.describe(verts, 'vertex', comments=['vertices']),  PlyElement.describe(faces, 'face')], comments=['faces'])
    with open(filename, mode='wb') as f:
        PlyData(objdata).write(f)

if __name__ == '__main__':
    opt = parser.parse_args()

    grid = Vox.load_vox(opt.in_vox)
    verts, faces, normals, _ = measure.marching_cubes(grid.sdf, opt.iso)
    verts = np.fliplr(verts) # <-- go from zyx to xyz

    rot = grid.grid2world[0:3,0:3]
    trans = grid.grid2world[0:3,3]
    verts = np.matmul(verts, rot.transpose()) + trans


    color = [200, 50, 50]

    save_as_ply(opt.out_ply, verts, faces, color)



