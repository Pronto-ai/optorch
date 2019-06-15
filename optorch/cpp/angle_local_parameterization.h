#pragma once

#include <cmath>
#include <ceres/local_parameterization.h>

template <typename T>
inline T normalize_angle(const T& angle_radians)
{
    T two_pi(2.0 * M_PI);
    return angle_radians -
        two_pi * ceres::floor((angle_radians + T(M_PI)) / two_pi);
}

class AngleLocalParameterization {
public:
    template <typename T>
    bool operator()(const T* theta_radians, const T* delta_theta_radians,
        T* theta_radians_plus_delta) const {
        *theta_radians_plus_delta = normalize_angle(*theta_radians + *delta_theta_radians);
        return true;
    }

    static ceres::LocalParameterization *Create() {
        return new ceres::AutoDiffLocalParameterization<AngleLocalParameterization, 1, 1>;
    }
};
