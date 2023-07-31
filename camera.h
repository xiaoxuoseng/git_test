//-------------------
//writ by luotianjiao insa360
// 2023.05.10
//-------------------
#ifndef _CAMERA_H_
#define _CAMERA_H_
// #include <Eigen/Core>
// #include <Eigen/Dense>
#include <string>
namespace camera_space {
class CameraBase {
public:
    CameraBase();
    virtual ~CameraBase();
    class ParameterBase {
    public:
        virtual bool readParameter(const std::string &path) = 0;
    };
    virtual bool readParam(const std::string &path) = 0;
    virtual int fisheye2eclude(const double u, const double v, double &x, double &y, double &z) = 0;
    virtual int eclude2fisheye(const double x, const double y, const double z, double &u, double &v) = 0;
};

class EquidCamera : public CameraBase {
public:
    EquidCamera();

    class Parameters : public ParameterBase {
    public:
        Parameters();
        bool readParameter(const std::string &path);
        double cx() const { return cx_; }
        double cy() const { return cy_; }
        double f() const { return f_; }
        double k1() const { return k1_; }
        double k2() const { return k2_; }
        double k3() const { return k3_; }
        double k4() const { return k4_; }
        double width() const { return width_; }
        double height() const { return height_; }
        double fov() const { return fov_; }

    private:
        double cx_;
        double cy_;
        double f_;
        double k1_;
        double k2_;
        double k3_;
        double k4_;
        double width_;
        double height_;
        double fov_;
    };

    virtual bool readParam(const std::string &path);
    virtual int fisheye2eclude(double u, double v, double &x, double &y, double &z);
    virtual int eclude2fisheye(const double x, const double y, const double z, double &u, double &v);

private:
    double distort(const double theta);
    double undistort(const double thetad);
private:
    Parameters parameters_;
    double fov_threshold_;
};
}
#endif