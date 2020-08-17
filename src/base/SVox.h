#pragma once

#include <string>
#include <vector>
#include <cmath>
#include <cassert>
#include <iostream>
#include <fstream>
#include <eigen3/Eigen/Dense>


class SVox {
	public:
		int n_elems;
		float res;
		Eigen::Matrix<int32_t, 3, 2> bbox;
		Eigen::Matrix4f grid2world;
		Eigen::MatrixXi coords;
		Eigen::MatrixXf pdf;
		Eigen::MatrixXf rgb;


		void load_svox(std::string filename) {

			std::ifstream f(filename, std::ios::binary);
			assert(f.is_open());


			f.read((char*)&n_elems, sizeof(int32_t));
			f.read((char*)&res, sizeof(float));
			f.read((char*)bbox.data(), 6*sizeof(int32_t));
			f.read((char*)grid2world.data(), 16*sizeof(float));

			f.close();	

		}

		void load(std::string filename) {

			std::ifstream f(filename, std::ios::binary);
			assert(f.is_open());


			f.read((char*)&n_elems, sizeof(int32_t));
			f.read((char*)&res, sizeof(float));
			f.read((char*)bbox.data(), 6*sizeof(int32_t));
			f.read((char*)grid2world.data(), 16*sizeof(float));

			coords.resize(3, n_elems);
			f.read((char*)coords.data(), 3*n_elems*sizeof(int32_t));

			if(f && f.peek() != EOF) {
				pdf.resize(1, n_elems);
				f.read((char*)pdf.data(), n_elems*sizeof(float));
			}
			if(f && f.peek() != EOF) {
				rgb.resize(3, n_elems);
				f.read((char*)rgb.data(), 3*n_elems*sizeof(float));
			}

			f.close();	

		}

		void save(std::string filename) {
			std::ofstream f;
			f.open(filename, std::ofstream::out | std::ios::binary);
			assert(f.is_open());

			f.write((char*)&n_elems, sizeof(int32_t));
			f.write((char*)&res, sizeof(float));
			f.write((char*)bbox.data(), 6*sizeof(int32_t));
			f.write((char*)grid2world.data(), 16*sizeof(float));

			if (coords.size() > 0) {
				assert(coords.size() == 3*n_elems);
				f.write((char*)coords.data(), 3*n_elems*sizeof(int32_t));
			}
			if (pdf.size() > 0) {
				assert(pdf.size() == n_elems);
				f.write((char*)pdf.data(), n_elems*sizeof(float));
			}
			if (rgb.size() > 0) {
				assert(rgb.size() == 3*n_elems);
				f.write((char*)rgb.data(), 3*n_elems*sizeof(float));
			}
			f.close();

		}
};
