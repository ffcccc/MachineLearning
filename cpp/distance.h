#include "eigenDist.h"

// double distEuclidean(const Eigen::Map<Eigen::ArrayXd> &x, const Eigen::Map<Eigen::ArrayXd> &y);
#include <iostream>

inline double distEuclidean(const Eigen::Map<Eigen::ArrayXd> &x, const Eigen::Map<Eigen::ArrayXd> &y) {
    // double res(0.0);
    double res = ComputeEuclidean<double>::compute(x, y);
    // std::cout << res << "\n";
    return res;
}