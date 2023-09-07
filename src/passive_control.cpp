//*************************************************************************************
// PassiveDS and PassiveDSImpedanceController Class taken and modified from https://github.com/epfl-lasa/dual_iiwa_toolkit.git
//|
//|    Copyright (C) 2020 Learning Algorithms and Systems Laboratory, EPFL, Switzerland
//|    Authors:  Farshad Khadivr (maintainer)
//|    email:   farshad.khadivar@epfl.ch
//|    website: lasa.epfl.ch
//|
//|    This file is part of iiwa_toolkit.
//|
//|    iiwa_toolkit is free software: you can redistribute it and/or modify
//|    it under the terms of the GNU General Public License as published by
//|    the Free Software Foundation, either version 3 of the License, or
//|    (at your option) any later version.
//|
//|    iiwa_toolkit is distributed in the hope that it will be useful,
//|    but WITHOUT ANY WARRANTY; without even the implied warranty of
//|    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//|    GNU General Public License for more details.
//|

#include "ros/ros.h"
#include "passive_control.hpp"

#include <kinematics_utils.hpp>

PassiveDS::PassiveDS(){
}

PassiveDS::~PassiveDS(){}
void PassiveDS::set_damping_eigval(const double& lam0, const double& lam1){
    if((lam0 > 0)&&(lam1 > 0)){
        eigVal0 = lam0;
        eigVal1 = lam1;
        damping_eigval(0,0) = eigVal0;
        damping_eigval(1,1) = eigVal1;
        damping_eigval(2,2) = eigVal1;
    }else{
        std::cerr << "wrong values for the eigenvalues"<<"\n";
    }
}
void PassiveDS::updateDampingMatrix(const Eigen::Vector3d& ref_vel){ 

    if(ref_vel.norm() > 1e-6){
        baseMat.setRandom();
        baseMat.col(0) = ref_vel.normalized();
        for(uint i=1;i<3;i++){
            for(uint j=0;j<i;j++)
                baseMat.col(i) -= baseMat.col(j).dot(baseMat.col(i))*baseMat.col(j);
            baseMat.col(i).normalize();
        }
        Dmat = baseMat*damping_eigval*baseMat.transpose();
    }else{
        Dmat = Eigen::Matrix3d::Identity();
    }
    // otherwise just use the last computed basis
}

void PassiveDS::update(const Eigen::Vector3d& vel, const Eigen::Vector3d& des_vel){
    // compute damping
    updateDampingMatrix(des_vel);
    // dissipate
    control_output = - Dmat * vel;
    // compute control
    control_output += eigVal0*des_vel;
}
Eigen::Vector3d PassiveDS::get_output(){ return control_output;}


//**********************************************
PassiveDSImpedanceController::PassiveDSImpedanceController(const std::string& urdf_string,const std::string& end_effector)
{
    _tools.init_rbdyn(urdf_string, end_effector);

    passive_ds_controller = std::make_unique<PassiveDS>();
    ang_passive_ds_controller = std::make_unique<PassiveDS>();
    
  
    _robot.name +=std::to_string(0);
    _robot.jnt_position.setZero();
    _robot.jnt_velocity.setZero();
    _robot.jnt_torque.setZero();
    _robot.nulljnt_position.setZero();
    _robot.ee_pos.setZero(); 
    _robot.ee_vel.setZero();   
    _robot.ee_acc.setZero();

    
    double angle0 = 0.25*M_PI;
    _robot.ee_quat[0] = (std::cos(angle0/2));
    _robot.ee_quat.segment(1,3) = (std::sin(angle0/2))* Eigen::Vector3d::UnitZ();
    
    _robot.ee_angVel.setZero();
    _robot.ee_angAcc.setZero();
   
   //* desired things
    _robot.ee_des_pos = {0.45, 0.0, 0.4}; 
    double angled = 1.0*M_PI;
    // _robot.ee_des_quat[0] = (std::cos(angled/2));
    // _robot.ee_des_quat.segment(1,3) = (std::sin(angled/2))* Eigen::Vector3d::UnitX();
    _robot.ee_des_quat << 0.70710678118, 0.0, 0.70710678118, 0.0;
    
    //** do we need these parts in here?
    _robot.ee_des_vel.setZero();   
    _robot.ee_des_acc.setZero();
    _robot.ee_des_angVel.setZero();
    _robot.ee_des_angAcc.setZero();


    _robot.jacob.setZero();
    _robot.jacob.setZero();       
    _robot.jacob_drv.setZero();   
    _robot.jacob_t_pinv.setZero();
    _robot.jacobPos.setZero();   
    _robot.jacobAng.setZero();
    _robot.pseudo_inv_jacob.setZero();   
    _robot.pseudo_inv_jacobPos.setZero();


    //_robot.nulljnt_position << 0.044752691045324394, 0.6951627023357917, -0.01416978801753847, -1.0922311725109015, -0.0050429618456282, 1.1717338014778385, -0.01502630060305613;

    F_ee_des_.resize(6);

}

PassiveDSImpedanceController::~PassiveDSImpedanceController(){}


void PassiveDSImpedanceController::updateRobot(const Eigen::VectorXd& jnt_p,const Eigen::VectorXd& jnt_v,const Eigen::VectorXd& jnt_t){
    
    _robot.jnt_position = jnt_p;
    _robot.jnt_velocity = jnt_v;
    _robot.jnt_torque   = jnt_t;

    iiwa_tools::RobotState robot_state;
    robot_state.position.resize(jnt_p.size());
    robot_state.velocity.resize(jnt_p.size());
    for (size_t i = 0; i < jnt_p.size(); i++) {
        robot_state.position[i] = _robot.jnt_position[i];
        robot_state.velocity[i] = _robot.jnt_velocity[i];
    }

    std::tie(_robot.jacob, _robot.jacob_drv) = _tools.jacobians(robot_state);
    _robot.jacobPos =  _robot.jacob.bottomRows(3);
    _robot.jacobAng =  _robot.jacob.topRows(3);

    _robot.pseudo_inv_jacob    = pseudo_inverse(Eigen::MatrixXd(_robot.jacob * _robot.jacob.transpose()) );
    _robot.pseudo_inv_jacobPos = pseudo_inverse(Eigen::MatrixXd(_robot.jacobPos * _robot.jacobPos.transpose()) );
    _robot.pseudo_inv_jacobJnt = pseudo_inverse(Eigen::MatrixXd(_robot.jacob.transpose() * _robot.jacob ) );

    auto ee_state = _tools.perform_fk(robot_state);
    _robot.ee_pos = ee_state.translation;
    _robot.ee_quat[0] = ee_state.orientation.w();
    _robot.ee_quat.segment(1,3) = ee_state.orientation.vec();
    
    
    Eigen::VectorXd vel = _robot.jacob * _robot.jnt_velocity;
    _robot.ee_vel    = vel.tail(3); // check whether this is better or filtering position derivitive

    _robot.ee_angVel = vel.head(3); // compare it with your quaternion derivitive equation

}

Eigen::Vector3d PassiveDSImpedanceController::getEEpos(){
    return _robot.ee_pos;
}

Eigen::Vector4d PassiveDSImpedanceController::getEEquat(){
    return _robot.ee_quat;
}


void PassiveDSImpedanceController::set_pos_gains(const double& lambda0,const double& lambda1){
    passive_ds_controller->set_damping_eigval(lambda0,lambda1);

}
void PassiveDSImpedanceController::set_ori_gains(const double& lambda0,const double& lambda1){
    ang_passive_ds_controller->set_damping_eigval(lambda0,lambda1);
}

void PassiveDSImpedanceController::set_null_gains(const Eigen::VectorXd& nullGain){
    nullgains = nullGain;
}

void PassiveDSImpedanceController::set_null_pos(const Eigen::VectorXd& nullPosition){
    if (nullPosition.size() == _robot.nulljnt_position.size() )
    {
        _robot.nulljnt_position = nullPosition;
    }else{
        ROS_ERROR("wrong size for the null joint position");
    }
}


void PassiveDSImpedanceController::set_desired_twist(const Eigen::Vector3d& lin_vel, const Eigen::Vector3d& ang_pos){
    //tricky here, linear part is the velocity, angular part is euler angle
     _robot.ee_des_vel    = lin_vel;

    //  tf2::Quaternion ee_des_qua;
    //  ee_des_qua.setRPY(ang_pos(0), ang_pos(1), ang_pos(2));
    //  ee_des_qua = ee_des_qua.normalize();
    //  _robot.ee_des_quat << ee_des_qua.getX(), ee_des_qua.getY(), ee_des_qua.getZ(), ee_des_qua.getW();

     enable_control = true;
}

void PassiveDSImpedanceController::set_desired_twist_quarternion(const Eigen::Vector3d& lin_vel, const Eigen::Vector4d& ang_pos){
     _robot.ee_des_vel    = lin_vel;
     _robot.ee_des_quat << ang_pos(0), ang_pos(1), ang_pos(2), ang_pos(3);

     enable_control = true;
}


void PassiveDSImpedanceController::set_load(const double& mass ){
    load_added = mass;
}



void PassiveDSImpedanceController::computeTorqueCmd(){

    //////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////              COMPUTING TASK CONTROL TORQUE           //////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////

    // desired linear values
    passive_ds_controller->update(_robot.ee_vel,_robot.ee_des_vel);
    Eigen::Vector3d wrenchPos = passive_ds_controller->get_output() + load_added * 9.8*Eigen::Vector3d::UnitZ();   
    Eigen::VectorXd tmp_jnt_trq_pos = _robot.jacobPos.transpose() * wrenchPos;



    // desired angular values
    Eigen::Vector4d dqd = KinematicsUtils<double>::slerpQuaternion(_robot.ee_quat, _robot.ee_des_quat, 0.5);    
    Eigen::Vector4d deltaQ = dqd -  _robot.ee_quat;

    Eigen::Vector4d qconj = _robot.ee_quat;
    qconj.segment(1,3) = -1 * qconj.segment(1,3);
    Eigen::Vector4d temp_angVel = KinematicsUtils<double>::quaternionProduct(deltaQ, qconj);

    Eigen::Vector3d tmp_angular_vel = temp_angVel.segment(1,3);
    double maxDq = 0.2;
    if (tmp_angular_vel.norm() > maxDq)
        tmp_angular_vel = maxDq * tmp_angular_vel.normalized();

    double theta_gq = (-.5/(4*maxDq*maxDq)) * tmp_angular_vel.transpose() * tmp_angular_vel;

    _robot.ee_des_angVel  = 2 * 2.50*(1+std::exp(theta_gq)) * tmp_angular_vel;
    
    ang_passive_ds_controller->update(_robot.ee_angVel,_robot.ee_des_angVel);
    Eigen::Vector3d wrenchAng   = ang_passive_ds_controller->get_output();
    Eigen::VectorXd tmp_jnt_trq_ang = _robot.jacobAng.transpose() * wrenchAng;


    //sum up:
    Eigen::VectorXd tmp_jnt_trq = tmp_jnt_trq_pos + tmp_jnt_trq_ang;

    // null pos control
    Eigen::MatrixXd tempMat2 =  Eigen::MatrixXd::Identity(7,7) - _robot.jacob.transpose()* _robot.pseudo_inv_jacob* _robot.jacob;
    Eigen::VectorXd er_null = _robot.jnt_position -_robot.nulljnt_position;

    if(er_null.norm()<1.){
        first = false;
    }

    if(er_null.norm()>4e-1){
        er_null = 0.4*er_null.normalized();
    }
    Eigen::VectorXd tmp_null_trq = Eigen::VectorXd::Zero(7);
    for (int i =0; i<7; i++){ 
        tmp_null_trq[i] = -nullgains[i] * er_null[i];
        tmp_null_trq[i] +=-2. * _robot.jnt_velocity[i];
    }
    if (!enable_control){
        tau_d = tmp_null_trq;
        ROS_INFO_ONCE("Go to the home configuration ");                 
    }else{
        ROS_INFO_ONCE("Tracking in process");
        tau_d = tmp_jnt_trq + tempMat2 * tmp_null_trq;
    }

}