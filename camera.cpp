#include "camera.h"
#include <assert.h>

#include <iostream>
#include <opencv2/core.hpp>
#include <Eigen/Dense>



namespace camera_space {

template <typename DataType>  // Eigen::Vector3d
double getDegAngle3d(const DataType v1, const DataType v2) {
    double radian = atan2(v1.cross(v2).norm(), v1.transpose() * v2);  //弧度
    /*if (v1.cross(v2).z() < 0) {
        radian = 2 * M_PI - radian;
    }*/

    return radian; // * 180 / M_PI;  //角度
}


CameraBase::CameraBase() {
    ;
}
CameraBase::~CameraBase() {
    ;
}
EquidCamera::EquidCamera() {
    fov_threshold_ = parameters_.fov() / 360 * M_PI;
}

bool EquidCamera::readParam(const std::string &path) {
    bool flag = parameters_.readParameter(path);
    if (flag) {
        fov_threshold_ = parameters_.fov() / 360 * M_PI;
    }
    return flag;
}

EquidCamera::Parameters::Parameters() {
    cx_ = 1.9089052734375000e+03;
    cy_ = 1.0820064697265625e+03;
    f_ = 1.9303140869140625e+03;
    k1_ = 1.;
    k2_ = -8.4859871998897735e-03;
    k3_ = 2.0359604318164626e-02;
    k4_ = -1.5424185860609096e-02;
    width_ = 3840;
    height_ = 2160;
    fov_ = 156;
}

bool EquidCamera::Parameters::readParameter(const std::string &path) {
    try {
        //读入标定参数
        cv::FileStorage fs(path, cv::FileStorage::READ);
        cx_ = (double)fs["cx"];
        cy_ = (double)fs["cy"];
        f_ = (double)fs["f"];
        k1_ = (double)fs["k1"];
        k2_ = (double)fs["k2"];
        k3_ = (double)fs["k3"];
        k4_ = (double)fs["k4"];
        width_ = (double)fs["width_"];
        height_ = (double)fs["height"];
        fov_ = (double)fs["fov"];
        fs.release();
    }
    catch (const char *&e) {
        std::cout << e << std::endl;
        return false;
    }
    return true;
}

double EquidCamera::distort(const double theta) {
    return (((parameters_.k4() * theta + parameters_.k3()) * theta + parameters_.k2()) * theta + parameters_.k1()) * theta;
}

//int EquidCamera::eclude2fisheye(const double x, const double y, const double z, double &u, double &v) {
//    //homework1 补全代码
//    /*double distort_u = x * parameters_.f() + parameters_.cx();
//    double distort_v = y * parameters_.f() + parameters_.cy();*/
//    std::cout << "x: " << x << "  y: " << y << "  z: " << z << " x*x+y*y+z*z = " << x * x + y * y + z * z << std::endl;
//    Eigen::Vector3d vector_z(0, 0, 1);
//    Eigen::Vector3d vector_p(x, y, z);
//    //求出光线入射角度
//    double theta = getDegAngle3d(vector_p, vector_z);
//    double arc_theta = acos(z);
//    std::cout << "theta: " << theta << std::endl;
//    std::cout << "arctheta: " << arc_theta << std::endl;
//    double thetad = distort(theta);
//    
//    std::cout << "thetad: " << thetad << std::endl;
//
//    /*double R = parameters_.f() * thetad;
//    double */
//
//    //求出矫正入射角度后入射光线的球面坐标
//    double new_x = x * (sin(thetad) / sin(theta));
//    double new_y = y * (sin(thetad) / sin(theta));
//    double new_z = cos(thetad);
//    //std::cout << "new_x: " << new_x << "  new_y: " << new_y << "  new_z: " << new_z
//              //<< " new_x*new_x+new_y*new_y+new_z*new_z = " << new_x * new_x + new_y * new_y + new_z * new_z << std::endl;
//
//    //////求解像素平面坐标
//    ////double edge_u = new_x * (parameters_.f() / new_z);
//    ////double edge_v = new_y * (parameters_.f() / new_z);
//    //////std::cout << "edge_u: " << edge_u << std::endl;
//    //////std::cout << "edge_v: " << edge_v << std::endl;
//    double uu = edge_u + parameters_.cx();
//    double vv = edge_v + parameters_.cy();
//    //std::cout << "uu: " << uu << std::endl;
//    //std::cout << "vv: " << vv << std::endl;
//    if (uu > parameters_.width() || vv > parameters_.height() || uu < 0 || vv < 0)
//    {
//        u = 0;
//        v = 0;
//        return -1;
//    }
//    u = (int32_t)uu;
//    v = (int32_t)vv;
//    return 0;
//}

int EquidCamera::eclude2fisheye(const double x, const double y, const double z, double &u, double &v) {
    double theta = acos(z);
    double phi = atan2(y, x);
    if (phi < 0) {
        phi += 2 * M_PI;
    }
    double thetad = distort(theta);
    double R = parameters_.f() * thetad;
    u = R * cos(phi) + parameters_.cx(), v = R * sin(phi) + parameters_.cy();

    return 0;
}

int EquidCamera::fisheye2eclude(double u, double v, double &x, double &y, double &z) {
    if (u < 0 || u > parameters_.width() || v < 0 || v > parameters_.height()) {
        x = 0.0;
        y = 0.0;
        z = 0.0;
        return -1;
    }

    double edgex = (u - parameters_.cx()) / parameters_.f();
    double edgey = (v - parameters_.cy()) / parameters_.f();
    double phy = atan2(edgey, edgex);
    if (phy < 0) {
        phy += 2 * M_PI;
    }

    double thetad = sqrt(edgex * edgex + edgey * edgey);

    double theta = undistort(thetad);
    if (theta > fov_threshold_) {
        x = 0.0;
        y = 0.0;
        z = 0.0;
        return -1;
    }
    z = cos(theta);
    double sin_theta = sin(theta);
    x = sin_theta * cos(phy);
    y = sin_theta * sin(phy);
    return 0;
}

double EquidCamera::undistort(const double thetad) {
    double thetadn = thetad;
    for (int i = 0; i < 20; i++) {
        double thetadn2 = thetadn * thetadn;
        double thetadn3 = thetadn2 * thetadn;
        double thetadn4 = thetadn3 * thetadn;
        double fthetadn =
            parameters_.k1() * thetadn + parameters_.k2() * thetadn2 + parameters_.k3() * thetadn3 + parameters_.k4() * thetadn4 - thetad;
        double fthetadndev =
            parameters_.k1() + 2 * parameters_.k2() * thetadn + 3 * parameters_.k3() * thetadn2 + 4 * parameters_.k4() * thetadn3;
        double thetadn_1 = thetadn - fthetadn / fthetadndev;
        if (abs(thetadn_1 - thetadn) < 1e-4) {
            break;
        }
        thetadn = thetadn_1;
    }
    return thetadn;
}
}  // namespace camera_space
