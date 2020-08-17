#pragma once

#include <eigen3/Eigen/Dense>
#include <vector>
#include "Vox.h"
#include "Triangle.h"

template<typename vtype>
Vox<vtype> voxelize(Eigen::MatrixXf& V, Eigen::MatrixXi& F, float res, vtype w0, vtype w1, int pad, bool verbose);
