#include <cuda_runtime_api.h>

#include <vector>
#include <eigen3/Eigen/Dense>

#define ID_SOLID 0
#define ID_WALL 2
#define ID_WATER 4

void sdf(std::vector<uint8_t>& grid, std::vector<float>& out, Eigen::Array3i dims, int trunc);

__global__ void init_kernel(uint8_t* water, float* tmp, Eigen::Array3i dims, int trunc);

__global__ void thicken_kernel(float* sdf, float* tmp, Eigen::Array3i dims, int trunc);

template <typename T>
__device__ int sign(T val);

