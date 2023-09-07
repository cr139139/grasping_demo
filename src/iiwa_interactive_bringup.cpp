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
#include "iiwa_interactive_bringup.hpp"


IiwaInteractiveBringup::IiwaInteractiveBringup(ros::NodeHandle &n,double frequency) : _n(n)
                                                                        , _loopRate(frequency)
                                                                        , _dt(1.0f/frequency)
{
    _stop =false;
}

IiwaInteractiveBringup::~IiwaInteractiveBringup(){}

bool IiwaInteractiveBringup::init()
{
    _feedback.jnt_position.setZero();
    _feedback.jnt_velocity.setZero();
    _feedback.jnt_torque.setZero();
    command_trq.setZero();

    robotStates = _n.subscribe<sensor_msgs::JointState> ("/iiwa/joint_states", 1, boost::bind(&IiwaInteractiveBringup::updateRobotStates,this,_1), ros::VoidPtr(),ros::TransportHints().reliable().tcpNoDelay());

    torqueCmdPublisher = _n.advertise<std_msgs::Float64MultiArray>("/iiwa/TorqueController/command",1);
    taskStatesPublisher = _n.advertise<geometry_msgs::Pose>("/iiwa/task_states",1);

    if (_n.getParam("control_interface", control_interface))
    {
        ROS_INFO("control_interface: %s", control_interface.c_str());
    }else
    {
        ROS_ERROR("Unable to get param 'control_interface'");
    }


    if (control_interface == "twist"){
        robotControl = _n.subscribe<geometry_msgs::Twist>("/iiwa/desired_twist", 1, boost::bind(&IiwaInteractiveBringup::updateControlTwist,this,_1), ros::VoidPtr(),ros::TransportHints().reliable().tcpNoDelay());
    }else if (control_interface == "position"){
        robotControl = _n.subscribe<geometry_msgs::Pose>("/iiwa/desired_pos", 1, boost::bind(&IiwaInteractiveBringup::updateControlPos,this,_1), ros::VoidPtr(),ros::TransportHints().reliable().tcpNoDelay());
    }else{
        ROS_ERROR("Control interface does not exist");
        return false;
    }

    std::string urdf_string;
    std::string full_param;
    std::string robot_description;
    std::string end_effector;

    robot_description  = "/robot_description";

    if (!_n.searchParam(robot_description, full_param)) {
        ROS_ERROR("Could not find parameter %s on parameter server", robot_description.c_str());
        return false;
    }

    while (urdf_string.empty()) {
        ROS_INFO_ONCE_NAMED("Controller", "Controller is waiting for model" " URDF in parameter [%s] on the ROS param server.", robot_description.c_str());
        _n.getParam(full_param, urdf_string);
        usleep(100000);
    }

    ROS_INFO_STREAM_NAMED("Controller", "Received urdf from param server, parsing...");

    _n.param<std::string>("params/end_effector", end_effector,  "iiwa_link_ee");


    // Initialize PassiveDS params

    //**** Initialize LINEAR PassiveDS params ****//
    Eigen::Vector3d damping_eigvals_yaml_;
    damping_eigvals_yaml_.setZero();
    std::string damping_params_name = "linear_damping_eigenvalues";
    loadParams(damping_params_name, 3, damping_eigvals_yaml_);


    //**** Initialize ANGULAR PassiveDS params ****//
    Eigen::Vector3d ang_damping_eigvals_yaml_;
    ang_damping_eigvals_yaml_.setZero();
    std::string ang_damping_params_name = "angular_damping_eigenvalues";
    loadParams(ang_damping_params_name, 3, ang_damping_eigvals_yaml_);


    // Initialize variables for nullspace control from yaml config file
    Eigen::VectorXd q_d_nullspace_gain = Eigen::VectorXd::Zero(7);
    std::string q_nullgains_params_name = "q_nullgains";
    loadParams(q_nullgains_params_name, 7, q_d_nullspace_gain);

    Eigen::VectorXd null_jt_pos = Eigen::VectorXd::Zero(7);
    std::string null_jt_pos_name = "q_nullpose";
    loadParams(null_jt_pos_name, 7, null_jt_pos);

    // Create Controller
    _controller  = std::make_unique<PassiveDSImpedanceController>(urdf_string,  end_effector);

    _controller->set_pos_gains(damping_eigvals_yaml_(0),damping_eigvals_yaml_(1));
    _controller->set_ori_gains(ang_damping_eigvals_yaml_(0),ang_damping_eigvals_yaml_(1));
    _controller->set_null_gains(q_d_nullspace_gain);
    _controller->set_null_pos(null_jt_pos);
    _controller->set_load(0.4);

    return true;

}

void IiwaInteractiveBringup::run()
{
    while(!_stop && ros::ok())
    { 
      _mutex.lock();
          _controller->updateRobot(_feedback.jnt_position,_feedback.jnt_velocity,_feedback.jnt_torque);
          publishCommandTorque(_controller->getCmd());
          publishTaskState();

      _mutex.unlock();
        
      ros::spinOnce();
      _loopRate.sleep();
    }
    //
    Eigen::Matrix<double, 7,1> cmdTrq_zero;
    cmdTrq_zero.setZero();
    publishCommandTorque(cmdTrq_zero);
    //
    ros::spinOnce();
    _loopRate.sleep();
    ros::shutdown();
}

void IiwaInteractiveBringup::updateRobotStates(const sensor_msgs::JointState::ConstPtr &msg)
{
  for (int i = 0; i < No_JOINTS; i++){
    _feedback.jnt_position[i] = (double)msg->position[i];
    _feedback.jnt_velocity[i] = (double)msg->velocity[i];
    _feedback.jnt_torque[i]   = (double)msg->effort[i];
  }
  // std::cout << "joint ps : " << _feedback.jnt_position.transpose() << "\n";   
}

void IiwaInteractiveBringup::updateTorqueCommand(const std_msgs::Float64MultiArray::ConstPtr &msg)
{
  for (int i = 0; i < No_JOINTS; i++){
      command_trq[i] = (double)msg->data[i];
  }  
}

void IiwaInteractiveBringup::publishCommandTorque(const Eigen::Matrix<double, 7,1>  &cmdTrq)
{
    std_msgs::Float64MultiArray _cmd_jnt_torque;
    _cmd_jnt_torque.data.resize(No_JOINTS);

    if (cmdTrq.size() == No_JOINTS){
        for(int i = 0; i < No_JOINTS; i++)
            _cmd_jnt_torque.data[i] = cmdTrq[i];
        torqueCmdPublisher.publish(_cmd_jnt_torque);
    }
}

void IiwaInteractiveBringup::publishTaskState()
{
    Eigen::Vector3d pos = _controller->getEEpos();
    Eigen::Vector4d ori = _controller->getEEquat();
    geometry_msgs::Pose task_space_pose;
    task_space_pose.position.x = pos(0);
    task_space_pose.position.y = pos(1);
    task_space_pose.position.z = pos(2);

    task_space_pose.orientation.x = ori(0);
    task_space_pose.orientation.y = ori(1);
    task_space_pose.orientation.z = ori(2);
    task_space_pose.orientation.w = ori(3);

    taskStatesPublisher.publish(task_space_pose);
}

void IiwaInteractiveBringup::updateControlPos(const geometry_msgs::Pose::ConstPtr& msg)
{
    Eigen::Matrix<double, 3, 3> matA;
    matA << -1.0,  0,  0, 0, -1.0,  0, 0,  0, -1.0; //should be able to modify
    Eigen::Matrix<double, 3, 1> matB;
    matB << (double)msg->position.x, (double)msg->position.y, (double)msg->position.z;
    Eigen::Vector3d lin_vel = matA * _controller->getEEpos() + matB;
    Eigen::Vector4d ang_pos;
    ang_pos << (double)msg->orientation.x, (double)msg->orientation.y, (double)msg->orientation.z, (double)msg->orientation.w;
    if(lin_vel.norm()<3.){
        _controller->set_desired_twist_quarternion(lin_vel, ang_pos);
    }else{
        ROS_WARN("VELOCITY OUT OF BOUND");
    }

}


void IiwaInteractiveBringup::updateControlTwist(const geometry_msgs::Twist::ConstPtr& msg)
{
    Eigen::Vector3d lin_vel;
    Eigen::Vector3d ang_pos;
    lin_vel << (double)msg->linear.x, (double)msg->linear.y, (double)msg->linear.z;
    ang_pos << (double)msg->angular.x, (double)msg->angular.y, (double)msg->angular.z;
    if(lin_vel.norm()<3.){
        _controller->set_desired_twist(lin_vel, ang_pos);
    }else{
        ROS_WARN("VELOCITY OUT OF BOUND");
    }

}

void IiwaInteractiveBringup::loadParams(std::string& param_name, int n_element, Eigen::Ref<Eigen::VectorXd> res)
{
    std::vector<double> param_vec;
    if (_n.getParam(param_name, param_vec)) {

      if (param_vec.size() != n_element) {
        ROS_ERROR(
          "PassiveDSImpedanceController: Invalid or no %s parameters provided, aborting controller init!", param_name.c_str());
      }
      for (size_t i = 0; i < n_element; ++i) 
        res[i] = param_vec.at(i);
      ROS_INFO_STREAM("Loaded params (from YAML)");
    }
}

int main (int argc, char **argv)
{
    float frequency = 200.0f;
    ros::init(argc,argv, "iiwa_interactive_bringup");
    ros::NodeHandle n;

    // Options options;

    // while(!n.getParam("options/filter_gain", options.filter_gain)){ROS_INFO("Wating for the option setting");}


    std::unique_ptr<IiwaInteractiveBringup> IiwaTrack = std::make_unique<IiwaInteractiveBringup>(n,frequency);

    if (!IiwaTrack->init()){
        return -1;
    }else{
        IiwaTrack->run();
    }
    return 0;
}
