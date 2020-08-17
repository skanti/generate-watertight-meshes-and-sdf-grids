#include <cuda_runtime_api.h>

#include <vector>
#include <eigen3/Eigen/Dense>

void watershed(std::vector<float>& grid, const Eigen::MatrixXf verts, const Eigen::MatrixXi& faces, Eigen::Array3i dims, Eigen::Matrix4f grid2world);

__global__ void watershed_kernel(float* __restrict__ grid, const float* __restrict__ verts, 
		const int32_t* __restrict__ faces, int n_triangles, Eigen::Vector3i dims, Eigen::Matrix4f grid2world);

__device__ float calc_distance_point_to_triangle(
		Eigen::Vector3f v0, Eigen::Vector3f v1, Eigen::Vector3f v2, Eigen::Vector3f p);

