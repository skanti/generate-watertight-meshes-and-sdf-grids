#include <cuda_runtime_api.h>

#include <vector>
#include <cmath>
#include "sdf.cuh"

#include <eigen3/Eigen/Dense>

#define INDEX3(data, x, y, z, dims) (data[(z)*dims(1)*dims(0) + (y)*dims(0) + (x)])
#define IS_IN_RANGE3(x, y, z) ((x) >= 0 && (x) < dims(0) && (y) >= 0 && (y) < dims(1) && (z) >= 0 && (z) < dims(2))
#define OFFSET3(dx, dy, dz) (IS_IN_RANGE3((x + dx), (y + dy), (z + dz)) ? \
		data[(z + dz)*dims(1)*dims(0) + (y + dy)*dims(0) + (x + dx)] : 1e3)




void sdf(std::vector<uint8_t>& walls, std::vector<float>& out, Eigen::Array3i dims, int trunc) {

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
	uint8_t* water;
	cudaMalloc(&water, n_elems*sizeof(uint8_t)); 
	cudaMemcpy(water, walls.data(), n_elems*sizeof(uint8_t), cudaMemcpyHostToDevice);
	
	float *sdf;
	cudaMalloc(&sdf, n_elems*sizeof(float)); 
	
	float *tmp;
	cudaMalloc(&tmp, n_elems*sizeof(float)); 
	// <-

	// -> init
	init_kernel<<<blocks, threads>>>(water, sdf, dims, trunc);
	cudaDeviceSynchronize();
	// <-

	// -> thicken
	int n_iteration = dims.maxCoeff();
	for (int i = 0; i < n_iteration; i++) {
		thicken_kernel<<<blocks, threads>>>(sdf, tmp, dims, trunc);
		cudaDeviceSynchronize();
		cudaMemcpy(sdf, tmp, n_elems*sizeof(float), cudaMemcpyDeviceToDevice);
	}
	// <-

	// copy data
	cudaMemcpy(out.data(), sdf, n_elems*sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(water);
	// <-

};

__global__ void init_kernel(uint8_t* water, float* tmp, Eigen::Array3i dims, int trunc) {

	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	int z = blockDim.z * blockIdx.z + threadIdx.z;
	if (!IS_IN_RANGE3(x, y, z))
		return;

	uint8_t v = INDEX3(water, x, y, z, dims);
	if (v == ID_SOLID) {
		INDEX3(tmp, x, y, z, dims) = -1e3;
	} else if (v == ID_WALL) {
		INDEX3(tmp, x, y, z, dims) = 0;
	} else if (v == ID_WATER) {
		INDEX3(tmp, x, y, z, dims) = 1e3;
	} else {
		assert(0);
	}
}

__device__ int min_neightbor(float* data, int x, int y, int z, Eigen::Array3i dims) {
	float v[26] = {};
	v[0] = OFFSET3(-1, -1, -1);
	v[1] = OFFSET3(-1, -1, 0);
	v[2] = OFFSET3(-1, -1, 1);
	v[3] = OFFSET3(-1, 0, -1);
	v[4] = OFFSET3(-1, 0, 0);
	v[5] = OFFSET3(-1, 0, 1);
	v[6] = OFFSET3(-1, 1, -1);
	v[7] = OFFSET3(-1, 1, 0);
	v[8] = OFFSET3(-1, 1, 1);
	v[9] = OFFSET3(0, -1, -1);
	v[10] = OFFSET3(0, -1, 0);
	v[11] = OFFSET3(0, -1, 1);
	v[12] = OFFSET3(0, 0, -1);
	v[13] = OFFSET3(0, 0, 1);
	v[14] = OFFSET3(0, 1, -1);
	v[15] = OFFSET3(0, 1, 0);
	v[16] = OFFSET3(0, 1, 1);
	v[17] = OFFSET3(1, -1, -1);
	v[18] = OFFSET3(1, -1, 0);
	v[19] = OFFSET3(1, -1, 1);
	v[20] = OFFSET3(1, 0, -1);
	v[21] = OFFSET3(1, 0, 0);
	v[22] = OFFSET3(1, 0, 1);
	v[23] = OFFSET3(1, 1, -1);
	v[24] = OFFSET3(1, 1, 0);
	v[25] = OFFSET3(1, 1, 1);
	float vmin = 1e3;
	for (int i =0; i < 26; i++) {
		float vi = std::abs(v[i]); 
		vmin = vi < vmin ? vi : vmin;
	}
	return vmin;
}

template <typename T> __device__ int sign(T val) {
	return (T(0) < val) - (val < T(0));
}

template __device__ int sign<float>(float);
template __device__ int sign<int>(int);

__global__ void thicken_kernel(float* sdf, float* tmp, Eigen::Array3i dims, int trunc) {

	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	int z = blockDim.z * blockIdx.z + threadIdx.z;
	if (!IS_IN_RANGE3(x, y, z))
		return;

	auto v = INDEX3(sdf, x, y, z, dims);
	//INDEX3(tmp, x, y, z, dims) = trunc;

	auto vmin = min_neightbor(sdf, x, y, z, dims);
	auto s = sign(v);

	INDEX3(tmp, x, y, z, dims) = s*(vmin+ 1.0);

}
