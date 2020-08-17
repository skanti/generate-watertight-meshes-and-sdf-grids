#include <cuda_runtime_api.h>

#include <vector>
#include "watershed.cuh"

#include <eigen3/Eigen/Dense>

void watershed(std::vector<float>& grid, const Eigen::MatrixXf verts, const Eigen::MatrixXi& faces, Eigen::Array3i dims) {

	//// -> memory
	//int size_verts = verts.size();
	//float *verts_;
	//cudaMalloc(&verts_, size_verts*sizeof(float)); 
	//cudaMemcpy(verts_, verts.data(), size_verts*sizeof(int32_t), cudaMemcpyHostToDevice);
	//// <-
	
	// -> thread blocks
	assert(dims(0)%8 == 0);
	assert(dims(1)%8 == 0);
	assert(dims(2)%8 == 0);
	size_t n_elems = dims(0)*dims(1)*dims(2);
	size_t n_triangles = faces.size();
	dim3 blocks(dims(0)/8, dims(1)/8, dims(2)/8);
	dim3 threads(8, 8, 8);
	// <-

	// -> launch
	float* grid_;
	cudaMalloc(&grid_, n_elems*sizeof(float)); 
	for (int i = 0; i < 100; i++) {
		watershed_kernel<<<blocks, threads>>>(grid_, verts.data(), faces.data(), n_triangles, dims, grid2world);
	}
	cudaDeviceSynchronize();
	// <-

	// copy data
	assert(grid.size() == n_elems);
	cudaMemcpy(grid.data(), grid_, n_elems*sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(grid_);
	// <-

};


__global__ void watershed_kernel(float* __restrict__ grid, const float* __restrict__ verts, const int32_t* __restrict__ faces, 
		int n_triangles, Eigen::Vector3i dims, float bbox_min, float bbox_max) {

	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	int z = blockDim.z * blockIdx.z + threadIdx.z;
	int idx = z*dims(1)*dims(0) + y*dims(0) + x;

	Eigen::Vector4p p_ = grid2world*Eigen::Vector4f(x, y, z, 1);
	Eigen::Vector3f p = p_.topRows(3);


	float distance_min = 1e6;

	//for (int i = 0; i < n_triangles; i++) {
	//	const int32_t t0 = faces[0*n_triangles + i];
	//	Eigen::Vector3f v0(verts[3*t0 + 0], verts[3*t0 + 1], verts[3*t0 + 2]);

	//	const int32_t t1 = faces[1*n_triangles + i];
	//	Eigen::Vector3f v1(verts[3*t1 + 0], verts[3*t1 + 1], verts[3*t1 + 2]);

	//	const int32_t t2 = faces[2*n_triangles + i];
	//	Eigen::Vector3f v2(verts[3*t2 + 0], verts[3*t2 + 1], verts[3*t2 + 2]);
	//	
	//	float d = calc_distance_point_to_triangle(v0, v1, v2, p);
	//	distance_min = d < distance_min ? d : distance_min;
	//}
	grid[idx] = 1;
}

__device__ float calc_distance_point_to_triangle(Eigen::Vector3f v0, Eigen::Vector3f v1, Eigen::Vector3f v2, Eigen::Vector3f p) {
	float s,t; //  <-- temporary variables

	Eigen::Matrix<float, 3, 1> base = v0;
    Eigen::Matrix<float, 3, 1> E0 = v1 - v0;
    Eigen::Matrix<float, 3, 1> E1 = v2 - v0;

	float a = E0.dot(E0);
	float b = E0.dot(E1);
	float c = E1.dot(E1);

    // distance vector
    const Eigen::Matrix<float, 3, 1> D = base - p;

    // Precalculate distance factors.
    const float d = E0.dot(D);
    const float e = E1.dot(D);
    const float f = D.dot(D);

    // Do classification
    const float det = a*c - b*b;

    s = b*e - c*d;
    t = b*d - a*e;

    if (s+t < det)
    {
        if (s < 0)
        {
            if (t < 0)
            {
                //region 4
                if (e > 0)
                {
                    //min on edge t = 0
                    t = 0;
                    s = (d >= 0 ? 0 : (-d >= a ? 1 : -d/a));
                }
                else
                {
                    //min on edge s=0
                    s = 0;
                    t = (e >= 0 ? 0 : (-e >= c ? 1 : -e/c));
                }
            }
            else
            {
                //region 3. Min on edge s = 0
                s = 0;
                t = (e >= 0 ? 0 : (-e >= c ? 1 : -e/c));
            }
        }
        else if (t < 0)
        {
            //region 5
            t = 0;
            s = (d >= 0 ? 0 : (-d >= a ? 1 : -d/a));
        }
        else
        {
            //region 0
            const float invDet = 1/det;
            s *= invDet;
            t *= invDet;
        }
    }
    else
    {
        if (s < 0)
        {
            //region 2
            const float tmp0 = b + d;
            const float tmp1 = c + e;
            if (tmp1 > tmp0)
            {
                //min on edge s+t=1
                const float numer = tmp1 - tmp0;
                const float denom = a-2*b+c;
                s = (numer >= denom ? 1 : numer/denom);
                t = 1 - s;
            }
            else
            {
                //min on edge s=0
                s = 0;
                t = (tmp1 <= 0 ? 1 : (e >= 0 ? 0 : - e/c));
            }
        }
        else if (t < 0)
        {
            //region 6
            const float tmp0 = b + d;
            const float tmp1 = c + e;
            if (tmp1 > tmp0)
            {
                //min on edge s+t=1
                const float numer = tmp1 - tmp0;
                const float denom = a-2*b+c;
                s = (numer >= denom ? 1 : numer/denom);
                t = 1 - s;
            }
            else
            {
                //min on edge t=0
                t = 0;
                s = (tmp1 <= 0 ? 1 : (d >= 0 ? 0 : - d/a));
            }
        }
        else
        {
            //region 1
            const float numer = c+e-(b+d);
            if (numer <= 0)
            {
                s = 0;
            }
            else
            {
                const float denom = a-2*b+c;
                s = (numer >= denom ? 1 : numer/denom);
            }
        }
        t = 1 - s;
    }
	return std::sqrt(a*s*s + 2*b*s*t + c*t*t + 2*d*s + 2*e*t + f);

}
