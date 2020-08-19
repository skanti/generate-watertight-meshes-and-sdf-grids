./build/watertight --in ./data/sofa0.obj --out ./data/sofa0.vox2
python ./python_scripts/marching_cubes.py --in_vox ./data/sofa0.vox2 --out_ply ./data/sofa0.ply
echo "sofa0 (sdf generation + marching cubes) done..."
./build/watertight --in ./data/sofa1.obj --out ./data/sofa1.vox2
python ./python_scripts/marching_cubes.py --in_vox ./data/sofa1.vox2 --out_ply ./data/sofa1.ply
echo "sofa1 (sdf generation + marching cubes) done..."
./build/watertight --in ./data/chair0.obj --out ./data/chair0.vox2
python ./python_scripts/marching_cubes.py --in_vox ./data/chair0.vox2 --out_ply ./data/chair0.ply
echo "chair0 (sdf generation + marching cubes) done..."
./build/watertight --in ./data/chair1.obj --out ./data/chair1.vox2
python ./python_scripts/marching_cubes.py --in_vox ./data/chair1.vox2 --out_ply ./data/chair1.ply
echo "chair1 (sdf generation + marching cubes) done..."
