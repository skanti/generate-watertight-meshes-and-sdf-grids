#pragma once

#include <cmath>
#include <eigen3/Eigen/Dense>

template <typename real>
real calc_distance_point_to_triangle(const Eigen::Matrix<real, 3, 1> v0, const Eigen::Matrix<real, 3, 1>& v1, const Eigen::Matrix<real, 3, 1>& v2, const Eigen::Matrix<real, 3, 1>& P) {
	real t = 0;
	real s = 0;

	Eigen::Matrix<real, 3, 1> base = v0;
    Eigen::Matrix<real, 3, 1> E0 = v1 - v0;
    Eigen::Matrix<real, 3, 1> E1 = v2 - v0;

	real a = E0.dot(E0);
	real b = E0.dot(E1);
	real c = E1.dot(E1);

    // distance vector
    const Eigen::Matrix<real, 3, 1> D = base - P;

    // Precalculate distance factors.
    const real d = E0.dot(D);
    const real e = E1.dot(D);
    const real f = D.dot(D);

    // Do classification
    const real det = a*c - b*b;

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
            const real invDet = 1/det;
            s *= invDet;
            t *= invDet;
        }
    }
    else
    {
        if (s < 0)
        {
            //region 2
            const real tmp0 = b + d;
            const real tmp1 = c + e;
            if (tmp1 > tmp0)
            {
                //min on edge s+t=1
                const real numer = tmp1 - tmp0;
                const real denom = a-2*b+c;
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
            const real tmp0 = b + d;
            const real tmp1 = c + e;
            if (tmp1 > tmp0)
            {
                //min on edge s+t=1
                const real numer = tmp1 - tmp0;
                const real denom = a-2*b+c;
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
            const real numer = c+e-(b+d);
            if (numer <= 0)
            {
                s = 0;
            }
            else
            {
                const real denom = a-2*b+c;
                s = (numer >= denom ? 1 : numer/denom);
            }
        }
        t = 1 - s;
    }
	return std::sqrt(a*s*s + 2*b*s*t + c*t*t + 2*d*s + 2*e*t + f);

}

