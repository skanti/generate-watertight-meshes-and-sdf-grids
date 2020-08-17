#include <iostream>
#include <cuda_runtime_api.h>

#include <vector>
#include "watershed.cuh"

#include <eigen3/Eigen/Dense>

#define INDEX3(data, x, y, z, dims) (data[(z)*dims(1)*dims(0) + (y)*dims(0) + (x)])
#define IS_IN_RANGE3(x, y, z) ((x) >= 0 && (x) < dims(0) && (y) >= 0 && (y) < dims(1) && (z) >= 0 && (z) < dims(2))
#define OFFSET3(dx, dy, dz) (IS_IN_RANGE3((x + dx), (y + dy), (z + dz)) ? \
		data[(z + dz)*dims(1)*dims(0) + (y + dy)*dims(0) + (x + dx)] : -1)

#define NEIGHTBOR3(v) \
		  (OFFSET3(-1, -1, -1) == v || OFFSET3(-1, -1, 0) == v || OFFSET3(-1, -1, 1) == v \
		|| OFFSET3(-1, 0, -1)  == v || OFFSET3(-1, 0, 0)  == v || OFFSET3(-1, 0, 1)  == v \
		|| OFFSET3(-1, 1, -1)  == v || OFFSET3(-1, 1, 0)  == v || OFFSET3(-1, 1, 1)  == v \
		|| OFFSET3(0, -1, -1)  == v || OFFSET3(0, -1, 0)  == v || OFFSET3(0, -1, 1)  == v \
		|| OFFSET3(0, 0, -1)   == v             ||                OFFSET3(0, 0, 1)   == v \
		|| OFFSET3(0, 1, -1)   == v || OFFSET3(0, 1, 0)   == v || OFFSET3(0, 1, 1)   == v \
		|| OFFSET3(1, -1, -1)  == v || OFFSET3(1, -1, 0)  == v || OFFSET3(1, -1, 1)  == v \
		|| OFFSET3(1, 0, -1)   == v || OFFSET3(1, 0, 0)   == v || OFFSET3(1, 0, 1)   == v \
		|| OFFSET3(1, 1, -1)   == v || OFFSET3(1, 1, 0)   == v || OFFSET3(1, 1, 1)   == v) \


template<typename vtype>
void init_water(std::vector<vtype>& walls, Eigen::Array3i dims) {
	INDEX3(walls.data(), 0, 0, 0, dims) = ID_WATER;
	INDEX3(walls.data(), dims(0) - 1, 0, 0, dims) = ID_WATER;
	INDEX3(walls.data(), 0, dims(1) - 1, 0, dims) = ID_WATER;
	INDEX3(walls.data(), dims(0) - 1, dims(1) - 1, 0, dims) = ID_WATER;
	INDEX3(walls.data(), dims(0) - 1, 0, dims(2) - 1, dims) = ID_WATER;
	INDEX3(walls.data(), 0, dims(1) - 1, dims(2) - 1, dims) = ID_WATER;
	INDEX3(walls.data(), 0, 0, dims(2) - 1, dims) = ID_WATER;
	INDEX3(walls.data(), dims(0) - 1, dims(1) - 1, dims(2) - 1, dims) = ID_WATER;
}

template<typename vtype>
void watershed(std::vector<vtype>& walls, Eigen::Array3i dims) {

	//// -> memory
	//int size_verts = verts.size();
	//float *verts_;
	//cudaMalloc(&verts_, size_verts*sizeof(float)); 
	//cudaMemcpy(verts_, verts.data(), size_verts*sizeof(int32_t), cudaMemcpyHostToDevice);
	//// <-

	size_t n_elems = dims(0)*dims(1)*dims(2);
	assert(walls.size() == n_elems);

	// -> thread blocks
	Eigen::Array3i dims_padded = dims.unaryExpr([](const int x) { 
			int pad = 8 - x%8;
			int f = x%8 > 0 ? 1 : 0;
			return x + f*pad;
			});

	assert(dims_padded(0)%8 == 0);
	assert(dims_padded(1)%8 == 0);
	assert(dims_padded(2)%8 == 0);
	//size_t n_triangles = faces.size();
	dim3 blocks(dims_padded(0)/8, dims_padded(1)/8, dims_padded(2)/8);
	dim3 threads(8, 8, 8);
	// <-



	// -> init
	init_water(walls, dims);
	vtype* water;
	vtype *tmp;
	cudaMalloc(&water, n_elems*sizeof(vtype)); 
	cudaMalloc(&tmp, n_elems*sizeof(vtype)); 
	cudaMemcpy(water, walls.data(), n_elems*sizeof(vtype), cudaMemcpyHostToDevice);
	cudaMemcpy(tmp, walls.data(), n_elems*sizeof(vtype), cudaMemcpyHostToDevice);
	// <-

	int n_iteration = dims.maxCoeff();
	// -> fill
	for (int i = 0; i < n_iteration; i++) {
		watershed_kernel<vtype><<<blocks, threads>>>(water, tmp, dims);
		cudaDeviceSynchronize();
		cudaMemcpy(water, tmp, n_elems*sizeof(vtype), cudaMemcpyDeviceToDevice);
	}
	{
		replace_wall_kernel<vtype><<<blocks, threads>>>(water, tmp, dims);
		cudaDeviceSynchronize();
		cudaMemcpy(water, tmp, n_elems*sizeof(vtype), cudaMemcpyDeviceToDevice);
	}
	// <-

	// copy data
	cudaMemcpy(walls.data(), water, n_elems*sizeof(vtype), cudaMemcpyDeviceToHost);
	cudaFree(water);
	// <-

};


template<typename vtype>
__global__ void watershed_kernel(vtype* water, vtype* tmp, Eigen::Array3i dims) {

	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	int z = blockDim.z * blockIdx.z + threadIdx.z;
	if (!IS_IN_RANGE3(x, y, z))
		return;

	vtype v = INDEX3(water, x, y, z, dims);
	if (v == ID_SOLID) {
		vtype* data = water; // <-- needed for macro
		if (NEIGHTBOR3(ID_WATER)) 
			INDEX3(tmp, x, y, z, dims) = ID_WATER;
	}
}

template<typename vtype>
__global__ void replace_wall_kernel(vtype* water, vtype* tmp, Eigen::Array3i dims) {

	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	int z = blockDim.z * blockIdx.z + threadIdx.z;
	if (!IS_IN_RANGE3(x, y, z))
		return;

	vtype v = INDEX3(water, x, y, z, dims);
	vtype* data = water; // <-- needed for macro
	if (v == ID_WALL) {
		INDEX3(tmp, x, y, z, dims) = ID_SOLID;
	} else if (v == ID_WATER) {
		if (NEIGHTBOR3(ID_WALL)) 
			INDEX3(tmp, x, y, z, dims) = ID_WALL;
	}
}


template void watershed<uint8_t>(std::vector<uint8_t>& , Eigen::Array3i);
template __global__ void watershed_kernel<uint8_t>(uint8_t*, uint8_t*, Eigen::Array3i);
