#include <mutex>
#include <fstream>
#include <pthread.h>
#include <memory>
#include "std_msgs/Float64MultiArray.h"
#include "sensor_msgs/JointState.h"
#include "geometry_msgs/PoseStamped.h"
#include "geometry_msgs/Twist.h"
#include "std_srvs/Empty.h"
#include "ros/ros.h"
#include <ros/package.h>
#include <Eigen/Dense>


#include "passive_control.hpp"

#define No_JOINTS 7


struct feedback
{
    Eigen::VectorXd jnt_position = Eigen::VectorXd(No_JOINTS);
    Eigen::VectorXd jnt_velocity = Eigen::VectorXd(No_JOINTS);
    Eigen::VectorXd jnt_torque   = Eigen::VectorXd(No_JOINTS);
};

class IiwaInteractiveBringup{
    protected:

        double _dt;


        ros::NodeHandle _n;
        ros::Rate _loopRate;

        ros::Subscriber robotStates;
        ros::Subscriber robotControl;
        

        ros::Publisher torqueCmdPublisher;
        ros::Publisher taskStatesPublisher;

        feedback _feedback;

        // std::shared_ptr<iiwa_tools::IiwaTools> _tools;
        Eigen::Matrix<double, 7,1> command_trq;

        std::unique_ptr<PassiveDSImpedanceController> _controller;

        bool _stop;                        // Check for CTRL+C
        std::mutex _mutex;

        Eigen::VectorXd q_d_nullspace_gain;

        std::string control_interface;

    public:

        IiwaInteractiveBringup(ros::NodeHandle &n,double frequency);
        ~IiwaInteractiveBringup();

        bool init();
        // run node
        void run();

    private:

        void updateRobotStates(const sensor_msgs::JointState::ConstPtr &msg);
        void updateTorqueCommand(const std_msgs::Float64MultiArray::ConstPtr &msg);

        void publishCommandTorque(const Eigen::Matrix<double, 7,1>  &cmdTrq);
        void publishTaskState();

        void updateControlPos(const geometry_msgs::Pose::ConstPtr& msg);
        void updateControlTwist(const geometry_msgs::Twist::ConstPtr& msg);


        void loadParams(std::string& param_name, int n_element, Eigen::Ref<Eigen::VectorXd> res);

};