#include "voxelize_project.h"

#include "glm/glm.hpp"

// An Axis Aligned box
template <typename T>
struct AABox {
	T min;
	T max;
	AABox() : min(T()), max(T()) {}
	AABox(T min, T max) : min(min), max(max) {}
};

template<typename T, int m>
inline glm::vec<m, float, glm::precision::highp> eigen2glm(const Eigen::Matrix<T, m, 1>& em) {
	glm::vec<m, float, glm::precision::highp> v;
	for (int i = 0; i < m; ++i)
	{
		v[i] = em(i);
	}
	return v;
}

template<typename vtype>
Vox<vtype> voxelize(Eigen::MatrixXf& V, Eigen::MatrixXi& F, float res, vtype w0, vtype w1, int pad, bool verbose){

	//// Common variables used in the voxelization process
	//glm::vec3 delta_p(res, res, res);
	//glm::vec3 c(0.0f, 0.0f, 0.0f); // critical point
	//glm::vec3 grid_max(info.gridsize.x - 1, info.gridsize.y - 1, info.gridsize.z - 1); // grid max (grid runs from 0 to gridsize-1)

	Vox<vtype> vox;

	assert(V.rows() == 3);
	assert(V.cols() >= 3);


	Eigen::Vector3f vmin = V.rowwise().minCoeff() - Eigen::Vector3f::Constant(pad*res);
	Eigen::Vector3f vmax = V.rowwise().maxCoeff() + Eigen::Vector3f::Constant(pad*res);

	Eigen::Array3i dims;
	Eigen::Matrix4f grid2world;

	dims = ((vmax - vmin)/res).array().floor().cast<int>() + 1;
	int n_elems = dims(0)*dims(1)*dims(2);


	grid2world = Eigen::Matrix4f::Identity();
	grid2world.block(0,0,3,3) *= res;
	grid2world.block(0,3, 3, 1) = vmin;

	vox.dims = dims;
	vox.res = res;
	vox.grid2world = grid2world;
	vox.sdf = std::vector<vtype>(n_elems, w0);

	size_t debug_n_triangles = 0;
	size_t debug_n_voxels_tested = 0;
	size_t debug_n_voxels_marked = 0;

	for (int f = 0; f < (int)F.cols(); f++) {
		Eigen::Vector3i idx = F.col(f);

		Eigen::MatrixXf v = Eigen::MatrixXf(3, 3);
		v.col(0) = V.col(idx(0)) - vmin;
		v.col(1) = V.col(idx(1)) - vmin;
		v.col(2) = V.col(idx(2)) - vmin;

		// Common variables used in the voxelization process
		glm::vec3 delta_p(res, res, res);
		glm::vec3 c(0.0f, 0.0f, 0.0f); // critical point
		glm::vec3 grid_max(dims(0) - 1, dims(1) - 1, dims(2) - 1);

		debug_n_triangles++;

		// COMPUTE COMMON TRIANGLE PROPERTIES
		// Move vertices to origin using bbox
		glm::vec3 v0 = eigen2glm<float, 3>(v.col(0));
		glm::vec3 v1 = eigen2glm<float, 3>(v.col(1));
		glm::vec3 v2 = eigen2glm<float, 3>(v.col(2));

		// Edge vectors
		glm::vec3 e0 = v1 - v0;
		glm::vec3 e1 = v2 - v1;
		glm::vec3 e2 = v0 - v2;
		// Normal vector pointing up from the triangle
		glm::vec3 n = glm::normalize(glm::cross(e0, e1));

		// COMPUTE TRIANGLE BBOX IN GRID
		// Triangle bounding box in world coordinates is min(v0,v1,v2) and max(v0,v1,v2)
		AABox<glm::vec3> t_bbox_world(glm::min(v0, glm::min(v1, v2)), glm::max(v0, glm::max(v1, v2)));
		// Triangle bounding box in voxel grid coordinates is the world bounding box divided by the grid unit vector
		AABox<glm::ivec3> t_bbox_grid;
		t_bbox_grid.min = glm::clamp(glm::floor(t_bbox_world.min / delta_p), glm::vec3(0.0f, 0.0f, 0.0f), grid_max);
		t_bbox_grid.max = glm::clamp(glm::floor(t_bbox_world.max / delta_p), glm::vec3(0.0f, 0.0f, 0.0f), grid_max);

		// PREPARE PLANE TEST PROPERTIES
		if (n.x > 0.0f) { c.x = res; }
		if (n.y > 0.0f) { c.y = res; }
		if (n.z > 0.0f) { c.z = res; }
		float d1 = glm::dot(n, (c - v0));
		float d2 = glm::dot(n, ((delta_p - c) - v0));

		// PREPARE PROJECTION TEST PROPERTIES
		// XY plane
		glm::vec2 n_xy_e0(-1.0f * e0.y, e0.x);
		glm::vec2 n_xy_e1(-1.0f * e1.y, e1.x);
		glm::vec2 n_xy_e2(-1.0f * e2.y, e2.x);
		if (n.z < 0.0f) {
			n_xy_e0 = -n_xy_e0;
			n_xy_e1 = -n_xy_e1;
			n_xy_e2 = -n_xy_e2;
		}
		float d_xy_e0 = (-1.0f * glm::dot(n_xy_e0, glm::vec2(v0.x, v0.y))) + glm::max(0.0f, res * n_xy_e0[0]) + glm::max(0.0f, res * n_xy_e0[1]);
		float d_xy_e1 = (-1.0f * glm::dot(n_xy_e1, glm::vec2(v1.x, v1.y))) + glm::max(0.0f, res * n_xy_e1[0]) + glm::max(0.0f, res * n_xy_e1[1]);
		float d_xy_e2 = (-1.0f * glm::dot(n_xy_e2, glm::vec2(v2.x, v2.y))) + glm::max(0.0f, res * n_xy_e2[0]) + glm::max(0.0f, res * n_xy_e2[1]);
		// YZ plane
		glm::vec2 n_yz_e0(-1.0f * e0.z, e0.y);
		glm::vec2 n_yz_e1(-1.0f * e1.z, e1.y);
		glm::vec2 n_yz_e2(-1.0f * e2.z, e2.y);
		if (n.x < 0.0f) {
			n_yz_e0 = -n_yz_e0;
			n_yz_e1 = -n_yz_e1;
			n_yz_e2 = -n_yz_e2;
		}
		float d_yz_e0 = (-1.0f * glm::dot(n_yz_e0, glm::vec2(v0.y, v0.z))) + glm::max(0.0f, res * n_yz_e0[0]) + glm::max(0.0f, res * n_yz_e0[1]);
		float d_yz_e1 = (-1.0f * glm::dot(n_yz_e1, glm::vec2(v1.y, v1.z))) + glm::max(0.0f, res * n_yz_e1[0]) + glm::max(0.0f, res * n_yz_e1[1]);
		float d_yz_e2 = (-1.0f * glm::dot(n_yz_e2, glm::vec2(v2.y, v2.z))) + glm::max(0.0f, res * n_yz_e2[0]) + glm::max(0.0f, res * n_yz_e2[1]);
		// ZX plane
		glm::vec2 n_zx_e0(-1.0f * e0.x, e0.z);
		glm::vec2 n_zx_e1(-1.0f * e1.x, e1.z);
		glm::vec2 n_zx_e2(-1.0f * e2.x, e2.z);
		if (n.y < 0.0f) {
			n_zx_e0 = -n_zx_e0;
			n_zx_e1 = -n_zx_e1;
			n_zx_e2 = -n_zx_e2;
		}
		float d_xz_e0 = (-1.0f * glm::dot(n_zx_e0, glm::vec2(v0.z, v0.x))) + glm::max(0.0f, res * n_zx_e0[0]) + glm::max(0.0f, res * n_zx_e0[1]);
		float d_xz_e1 = (-1.0f * glm::dot(n_zx_e1, glm::vec2(v1.z, v1.x))) + glm::max(0.0f, res * n_zx_e1[0]) + glm::max(0.0f, res * n_zx_e1[1]);
		float d_xz_e2 = (-1.0f * glm::dot(n_zx_e2, glm::vec2(v2.z, v2.x))) + glm::max(0.0f, res * n_zx_e2[0]) + glm::max(0.0f, res * n_zx_e2[1]);

		// test possible grid boxes for overlap
		for (int z = t_bbox_grid.min.z; z <= t_bbox_grid.max.z; z++) {
			for (int y = t_bbox_grid.min.y; y <= t_bbox_grid.max.y; y++) {
				for (int x = t_bbox_grid.min.x; x <= t_bbox_grid.max.x; x++) {
					// if (checkBit(voxel_table, location)){ continue; }
					debug_n_voxels_tested++;

					// TRIANGLE PLANE THROUGH BOX TEST
					glm::vec3 p(x * res, y * res, z * res);
					float nDOTp = glm::dot(n, p);
					if (((nDOTp + d1) * (nDOTp + d2)) > 0.0f) { continue; }

					// PROJECTION TESTS
					// XY
					glm::vec2 p_xy(p.x, p.y);
					if ((glm::dot(n_xy_e0, p_xy) + d_xy_e0) < 0.0f) { continue; }
					if ((glm::dot(n_xy_e1, p_xy) + d_xy_e1) < 0.0f) { continue; }
					if ((glm::dot(n_xy_e2, p_xy) + d_xy_e2) < 0.0f) { continue; }

					// YZ
					glm::vec2 p_yz(p.y, p.z);
					if ((glm::dot(n_yz_e0, p_yz) + d_yz_e0) < 0.0f) { continue; }
					if ((glm::dot(n_yz_e1, p_yz) + d_yz_e1) < 0.0f) { continue; }
					if ((glm::dot(n_yz_e2, p_yz) + d_yz_e2) < 0.0f) { continue; }

					// XZ	
					glm::vec2 p_zx(p.z, p.x);
					if ((glm::dot(n_zx_e0, p_zx) + d_xz_e0) < 0.0f) { continue; }
					if ((glm::dot(n_zx_e1, p_zx) + d_xz_e1) < 0.0f) { continue; }
					if ((glm::dot(n_zx_e2, p_zx) + d_xz_e2) < 0.0f) { continue; }
					debug_n_voxels_marked += 1;
					size_t idx = z*dims(1)*dims(0) + y*dims(0) + x;
					vox.sdf[idx] = w1;
					continue;
				}
			}
		}
	}
	if (verbose) {
		printf("[Debug] Processed %lu triangles on the CPU \n", debug_n_triangles);
		printf("[Debug] Tested %lu voxels for overlap on CPU \n", debug_n_voxels_tested);
		printf("[Debug] Marked %lu voxels as filled (includes duplicates!) on CPU \n", debug_n_voxels_marked);
	}


	return vox;
}


template Vox<uint8_t> voxelize(Eigen::MatrixXf&, Eigen::MatrixXi&, float , uint8_t , uint8_t ,int, bool);
