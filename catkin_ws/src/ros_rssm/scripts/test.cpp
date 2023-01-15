#include "ros/ros.h"
#include "ros_rssm/SendRssmPredictPosition.h"
#include <cstdlib>

int main(int argc, char **argv)
{
  ros::init(argc, argv, "test_client");
  if (argc != 3)
  {
    ROS_INFO("usage: add_two_ints_client X Y");
    return 1;
  }

  ros::NodeHandle n;
  ros::ServiceClient client = n.serviceClient<ros_rssm::SendRssmPredictPosition>("PredictPosition_RSSM");
  ros_rssm::SendRssmPredictPosition srv;
  srv.request.success = true;

  if (client.call(srv))
  {
    ROS_INFO("Sum(cpp): ");
  }
  else
  {
    ROS_ERROR("Failed to call service add_two_ints");
    return 1;
  }

  return 0;
}