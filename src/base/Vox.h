#pragma once

#include <string>
#include <vector>
#include <cmath>
#include <cassert>
#include <iostream>
#include <fstream>
#include <eigen3/Eigen/Dense>


template<typename vtype = float>
struct Vox {
	Eigen::Vector3i dims;
	float res;
	Eigen::Matrix4f grid2world;
	std::vector<vtype> sdf;
	std::vector<vtype> pdf;

	Eigen::Vector3f voxel2World(int i, int j, int k) {
		return (grid2world*Eigen::Vector4f(i, j, k, 1.0f)).topRows(3);
	}
	
	vtype& operator()(int i, int j, int k) {
		bool is_in_bounds = (Eigen::Array3i(i, j, k) < dims.array()).all();
		assert(is_in_bounds);
		return sdf[k*dims(1)*dims(0) + j*dims(0) + i];
	}
	const vtype& operator()(int i, int j, int k) const {
		bool is_in_bounds = (Eigen::Array3i(i, j, k) < dims.array()).all();
		assert(is_in_bounds);
		return sdf[k*dims(1)*dims(0) + j*dims(0) + i];
	}


	void load(std::string filename) {

		std::ifstream f(filename, std::ios::binary);
		assert(f.is_open());

		f.read((char*)dims.data(), 3*sizeof(int32_t));
		f.read((char*)&res, sizeof(float));
		f.read((char*)grid2world.data(), 16*sizeof(float));

		int n_elems = dims(0)*dims(1)*dims(2);	

		sdf.resize(n_elems);
		f.read((char*)sdf.data(), n_elems*sizeof(vtype));

		if(f && f.peek() != EOF) {
			pdf.resize(n_elems);
			f.read((char*)pdf.data(), n_elems*sizeof(vtype));
		}

		f.close();	
	}

	void load_header(std::string filename) {

		std::ifstream f(filename, std::ios::binary);
		assert(f.is_open());


		f.read((char*)dims.data(), 3*sizeof(int32_t));
		f.read((char*)&res, sizeof(float));
		f.read((char*)grid2world.data(), 16*sizeof(float));

		f.close();	

	}

	void save(std::string filename) {
		std::ofstream f;
		f.open(filename, std::ofstream::out | std::ios::binary);
		assert(f.is_open());
		f.write((char*)dims.data(), 3*sizeof(int32_t));
		f.write((char*)&res, sizeof(float));
		f.write((char*)grid2world.data(), 16*sizeof(float));

		std::size_t n_elems = dims(0)*dims(1)*dims(2);
		assert(sdf.size() == n_elems);
		f.write((char*)sdf.data(), n_elems*sizeof(vtype));

		if (pdf.size() > 0) {
			assert(pdf.size() == n_elems);
			f.write((char*)pdf.data(), n_elems*sizeof(vtype));
		}
		f.close();
	}
};
