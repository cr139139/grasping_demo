#include "ros/ros.h"
#include "geometry_msgs/Twist.h"
#include "geometry_msgs/PoseStamped.h"
#include "tf2_ros/transform_listener.h"
#include "tf2_ros/buffer.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.h"
#include "ros/ros.h"
#include <Eigen/Dense>

class PositionInterface
{
    private:
        ros::NodeHandle n;
        ros::Publisher des_twist_pub;
        ros::Subscriber task_state_sub;
        ros::Publisher des_pos_pub;
        ros::Timer timer;

        std::string control_interface;

        Eigen::Matrix<double, 3, 3> matA;
        Eigen::Matrix<double, 3, 1> matB;

        Eigen::Vector3d curr_pos;

        std::shared_ptr<tf2_ros::TransformListener> transform_listener_{nullptr};
        std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
        
    
    public:
        PositionInterface(ros::NodeHandle nh, std::string control_interface):
        timer(nh.createTimer(ros::Duration(0.01), &PositionInterface::main_loop, this)),
        control_interface(control_interface)
        {
            des_twist_pub = n.advertise<geometry_msgs::Twist>("/iiwa/desired_twist", 1);
            task_state_sub = n.subscribe<geometry_msgs::Pose>("/iiwa/task_states", 1, boost::bind(&PositionInterface::updateTaskState,this,_1), ros::VoidPtr(),ros::TransportHints().reliable().tcpNoDelay());

            des_pos_pub = n.advertise<geometry_msgs::Pose>("/iiwa/desired_pos", 1);


            tf_buffer_ = std::make_unique<tf2_ros::Buffer>();
            transform_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

            matA << -1.0,  0,  0,
                    0, -1.0,  0,
                    0,  0, -1.0;

            matB << 0.45,
                    0.0,
                    0.4;
        }

        void publishDesiredOptitrackPose(){
            geometry_msgs::TransformStamped transformStamped;
            try{
                transformStamped = tf_buffer_->lookupTransform("robot", "object",
                                        ros::Time(0));
            }
            catch (tf2::TransformException &ex) {
                ROS_WARN("%s",ex.what());
                ros::Duration(1.0).sleep();
            }
            // std::cout << " " << std::endl;
            // std::cout << "x: " << transformStamped.transform.translation.x << std::endl;
            // std::cout << "y: " << transformStamped.transform.translation.y << std::endl;
            // std::cout << "z: " << transformStamped.transform.translation.z << std::endl;

            geometry_msgs::Pose pose;
            pose.position.x = -transformStamped.transform.translation.y-0.3;
            pose.position.y = transformStamped.transform.translation.x;
            pose.position.z = transformStamped.transform.translation.z;

            pose.orientation.x = 0.70710678118;
            pose.orientation.y = 0.0;
            pose.orientation.z = 0.70710678118;
            pose.orientation.w = 0.0;
            des_pos_pub.publish(pose);

        }

        void updateTaskState(const geometry_msgs::Pose::ConstPtr& msg){
            curr_pos << (double)msg->position.x, (double)msg->position.y, (double)msg->position.z;
        }

        void updateDesiredState(const geometry_msgs::Pose::ConstPtr& msg){
            matB << (double)msg->position.x,
                    (double)msg->position.y,
                    (double)msg->position.z;
        }

        void publishDesiredTwist(){
            Eigen::Vector3d desired_twist = matA * curr_pos + matB;


            geometry_msgs::Twist twist;

            twist.linear.x = desired_twist(0);
            twist.linear.y = desired_twist(1);
            twist.linear.z = desired_twist(2);

            twist.angular.x = 0.0;
            twist.angular.y = 3.1415926;
            twist.angular.z = 0.0;

            des_twist_pub.publish(twist);
        }

        void publishDesiredFixedPose(){
            geometry_msgs::Pose pose;
            pose.position.x = matB(0);
            pose.position.y = matB(1);
            pose.position.z = matB(2);

            pose.orientation.x = 0.70710678118;
            pose.orientation.y = 0.0;
            pose.orientation.z = 0.70710678118;
            pose.orientation.w = 0.0;
            des_pos_pub.publish(pose);
        }

        void main_loop(const ros::TimerEvent &){
            if (control_interface == "position"){
                publishDesiredFixedPose();
            }else{
                publishDesiredTwist();
            }

            //publishDesiredOptitrackPose();
            
        }
};


int main(int argc, char **argv)
{
    ros::init(argc, argv, "PositionInterface");
    ros::NodeHandle n;

    std::string control_interface = "position";
    n.getParam("control_interface", control_interface);

    PositionInterface ds = PositionInterface(n, control_interface);
    ros::spin();
    return 0;
}
