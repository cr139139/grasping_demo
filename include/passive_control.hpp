#ifndef __PASSIVE_CONTROL__
#define __PASSIVE_CONTROL__
#include "ros/ros.h"
#include <ros/package.h>

#include <mutex>
#include <fstream>
#include <pthread.h>
#include <vector>
#include <memory>

#include <Eigen/Dense>
#include <iiwa_tools/iiwa_tools.h>
#include <tf2/LinearMath/Quaternion.h>
// #include "thirdparty/Utils.hpp"



struct Robot
{
    unsigned int no_joints = 7;
    Eigen::VectorXd jnt_position = Eigen::VectorXd(no_joints);
    Eigen::VectorXd jnt_velocity = Eigen::VectorXd(no_joints);
    Eigen::VectorXd jnt_torque = Eigen::VectorXd(no_joints);
    
    Eigen::VectorXd nulljnt_position = Eigen::VectorXd(no_joints);
    std::string name = "robot_";

    Eigen::Vector3d ee_pos, ee_vel, ee_acc, ee_angVel, ee_angAcc;
    Eigen::Vector4d ee_quat;

    Eigen::Vector3d ee_des_pos, ee_des_vel, ee_des_acc, ee_des_angVel, ee_des_angAcc;
    Eigen::Vector4d ee_des_quat;


    Eigen::MatrixXd jacob       = Eigen::MatrixXd(6, 7);
    Eigen::MatrixXd jacob_drv   = Eigen::MatrixXd(6, 7);
    Eigen::MatrixXd jacob_t_pinv= Eigen::MatrixXd(7, 6);
    Eigen::MatrixXd jacobPos    = Eigen::MatrixXd(3, 7);
    Eigen::MatrixXd jacobAng    = Eigen::MatrixXd(3, 7);

    Eigen::MatrixXd pseudo_inv_jacob       = Eigen::MatrixXd(6,6);
    Eigen::MatrixXd pseudo_inv_jacobJnt       = Eigen::MatrixXd(7,7);
    Eigen::MatrixXd pseudo_inv_jacobPos    = Eigen::MatrixXd(3,3);
    Eigen::MatrixXd pseudo_inv_jacobPJnt    = Eigen::MatrixXd(7,7);
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
template <class MatT>
Eigen::Matrix<typename MatT::Scalar, MatT::ColsAtCompileTime, MatT::RowsAtCompileTime> pseudo_inverse(const MatT& mat, typename MatT::Scalar tolerance = typename MatT::Scalar{1e-4}) // choose appropriately
{
    typedef typename MatT::Scalar Scalar;
    auto svd = mat.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
    const auto& singularValues = svd.singularValues();
    Eigen::Matrix<Scalar, MatT::ColsAtCompileTime, MatT::RowsAtCompileTime> singularValuesInv(mat.cols(), mat.rows());
    singularValuesInv.setZero();
    for (unsigned int i = 0; i < singularValues.size(); ++i) {
        if (singularValues(i) > tolerance) {
            singularValuesInv(i, i) = Scalar{1} / singularValues(i);
        }
        else {
            singularValuesInv(i, i) = Scalar{0};
        }
    }
    return svd.matrixV() * singularValuesInv * svd.matrixU().adjoint();
}
class PassiveDS
{
private:
    double eigVal0;
    double eigVal1;
    Eigen::Matrix3d damping_eigval = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d baseMat = Eigen::Matrix3d::Identity();

    Eigen::Matrix3d Dmat = Eigen::Matrix3d::Identity();
    Eigen::Vector3d control_output = Eigen::Vector3d::Zero();
    void updateDampingMatrix(const Eigen::Vector3d& ref_vel);
public:
    PassiveDS();
    ~PassiveDS();
    void set_damping_eigval(const double& lam0, const double& lam1);
    void update(const Eigen::Vector3d& vel, const Eigen::Vector3d& des_vel);
    Eigen::Vector3d get_output();
};


class PassiveDSImpedanceController
{
private:
    Robot _robot;
    iiwa_tools::IiwaTools _tools;

    bool first = true;
    bool enable_control = false;
    double load_added = 0.;

    Eigen::VectorXd tau_d = Eigen::VectorXd::Zero(7);
    void computeTorqueCmd();

    std::unique_ptr<PassiveDS> passive_ds_controller;
    std::unique_ptr<PassiveDS> ang_passive_ds_controller;


    Eigen::Vector3d     F_linear_des_;     // desired linear force
    Eigen::Vector3d     F_angular_des_;    // desired angular force
    Eigen::VectorXd     F_ee_des_;         // desired end-effector force

    Eigen::VectorXd nullgains = Eigen::VectorXd::Zero(7);
    
public:
    PassiveDSImpedanceController();
    PassiveDSImpedanceController(const std::string& urdf_string,const std::string& end_effector);
    ~PassiveDSImpedanceController();

    void updateRobot(const Eigen::VectorXd& jnt_p,const Eigen::VectorXd& jnt_v,const Eigen::VectorXd& jnt_t);
    void set_desired_twist(const Eigen::Vector3d& lin_vel, const Eigen::Vector3d& ang_pos);
    void set_desired_twist_quarternion(const Eigen::Vector3d& lin_vel, const Eigen::Vector4d& ang_pos);


    void set_pos_gains(const double& lambda0,const double& lambda1);
    void set_ori_gains(const double& lambda0,const double& lambda1);
    void set_null_gains(const Eigen::VectorXd& nullGain);
    void set_null_pos(const Eigen::VectorXd& nullPosition);
    void set_load(const double& mass);


    Eigen::VectorXd getCmd(){
        computeTorqueCmd();
        return tau_d;}
    Eigen::Vector3d getEEpos();
    Eigen::Vector4d getEEquat();

};

#endif