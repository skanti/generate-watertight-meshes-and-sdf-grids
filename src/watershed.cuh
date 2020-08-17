#include <cuda_runtime_api.h>

#include <vector>
#include <eigen3/Eigen/Dense>

#define ID_SOLID 0
#define ID_WALL 2
#define ID_WATER 4

template<typename vtype>
void watershed(std::vector<vtype>& grid, Eigen::Array3i dims);

template<typename vtype>
__global__ void watershed_kernel(vtype* water, vtype* tmp, Eigen::Array3i dims);

template<typename vtype>
__global__ void replace_wall_kernel(vtype* water, vtype* tmp, Eigen::Array3i dims);


