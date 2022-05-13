#!/bin/bash

################################################################################

# Link the default shell 'sh' to Bash.
alias sh='/bin/bash'

################################################################################

# Configure the terminal.

# Disable flow control. If enabled, inputting 'ctrl+s' locks the terminal until inputting 'ctrl+q'.
stty -ixon

################################################################################

# Configure 'umask' for giving read/write/execute permission to group members.
umask 0002

################################################################################

# Display the Docker container version.

# TODO: If switched from Dockerfile-sensitive generation to folder-sensitive generation,
# then show latest commit of the 'turtlebot3-devel' folder with 'git log -n 1 --format="%h %aN %s %ad" -- $directory'.
DOCKERFILE_LATEST_HASH=$(git -C /root/TurtleBot3/ log -n 1 --no-merges --pretty=format:%h ./docker/turtlebot3-devel/Dockerfile)
DOCKERFILE_LATEST_DATE=$(git -C /root/TurtleBot3/ log -n 1 --no-merges --pretty=format:%cd ./docker/turtlebot3-devel/Dockerfile)
DOCKERFILE_CREATION_DATE=$(git -C /root/TurtleBot3/ show --no-patch --no-notes --pretty='%cd' ${DOCKER_IMAGE_VERSION})

echo -e "Container version: turtlebot3-devel:${DOCKER_IMAGE_VERSION:0:7} from ${DOCKERFILE_CREATION_DATE}"
if [[ "${DOCKERFILE_CREATION_DATE}" != "${DOCKERFILE_LATEST_DATE}" ]]; then
  echo -e "Newer image available: turtlebot3-devel:${DOCKERFILE_LATEST_HASH} from ${DOCKERFILE_LATEST_DATE}"
fi

################################################################################

# Source the ROS environment.
echo "Sourcing the ROS environment from '/opt/ros/melodic/setup.bash'."
source /opt/ros/melodic/setup.bash

# Source the Catkin workspace.
echo "Sourcing the Catkin workspace from '/root/TurtleBot3/catkin_ws/devel/setup.bash'."
source /root/TurtleBot3/catkin_ws/devel/setup.bash

################################################################################

# Add the Catkin workspace to the 'ROS_PACKAGE_PATH'.
export ROS_PACKAGE_PATH=${ROS_PACKAGE_PATH}:/root/TurtleBot3/catkin_ws/src/

################################################################################

# Define Bash functions to conveniently execute the helper scripts in the current shell process.

function sde-build-catkin-workspace () {
  # Store the current directory and execute scripts in the current shell process.
  pushd .
  source /root/TurtleBot3/docker/turtlebot3-devel/scripts/fix-permission-issues.bash
  source /root/TurtleBot3/docker/turtlebot3-devel/scripts/build-catkin-workspace.bash
  popd
}

function sde-reset-catkin-workspace () {
  # Store the current directory and execute scripts in the current shell process.
  pushd .
  source /root/TurtleBot3/docker/turtlebot3-devel/scripts/fix-permission-issues.bash
  source /root/TurtleBot3/docker/turtlebot3-devel/scripts/reset-catkin-workspace.bash
  popd
}

function sde-fix-permission-issues () {
  # Store the current directory and execute scripts in the current shell process.
  pushd .
  source /root/TurtleBot3/docker/turtlebot3-devel/scripts/fix-permission-issues.bash
  popd
}

function sde-download-model-data () {
  # Store the current directory and execute scripts in the current shell process.
  pushd .
  source /root/TurtleBot3/docker/turtlebot3-devel/scripts/fix-permission-issues.bash
  source /root/TurtleBot3/docker/turtlebot3-devel/scripts/download-model-data.bash
  popd
}

function sde-get-fully-started () {
  # Store the current directory and execute scripts in the current shell process.
  pushd .
  source /root/TurtleBot3/docker/turtlebot3-devel/scripts/fix-permission-issues.bash
  source /root/TurtleBot3/docker/turtlebot3-devel/scripts/download-model-data.bash
  source /root/TurtleBot3/docker/turtlebot3-devel/scripts/reset-catkin-workspace.bash
  popd
}

################################################################################

# Set the TurtleBot3/ROS network interface.

# Look for the robot host name inside the local network and resolve it to get its IP address.
# The value of 'TURTLEBOT3_HOSTNAME' should be initialized in '~/.bashrc' by './RUN-DOCKER-CONTAINER.bash' when entering the container.
TURTLEBOT3_IP=`getent hosts ${TURTLEBOT3_HOSTNAME} | cut -d ' ' -f 1`

if [ -z "${TURTLEBOT3_IP}" ]; then
  # If no robot host name is found, set the local 'ROS_IP' using the default Docker network interface ('docker0').
  export ROS_IP=$(LANG=C /sbin/ifconfig docker0 | grep -Eo 'inet (addr:)?([0-9]*\.){3}[0-9]*' | grep -Eo '([0-9]*\.){3}[0-9]*')
else
  # If the robot host name is found, set the local 'ROS_IP' using the network interface that connects to the robot.
  # TODO: Use Bash instead of Python.
  export ROS_IP=`python /root/TurtleBot3/docker/turtlebot3-devel/scripts/print-interface-ip.py ${TURTLEBOT3_IP}`
fi
if [ -z "${ROS_IP}" ]; then
  # If the local 'ROS_IP' is still empty, default to the Docker network interface ('docker0') for sanity.
  export ROS_IP=$(LANG=C /sbin/ifconfig docker0 | grep -Eo 'inet (addr:)?([0-9]*\.){3}[0-9]*' | grep -Eo '([0-9]*\.){3}[0-9]*')
fi
echo "ROS_IP is set to '${ROS_IP}'."

export ROS_HOME=~/.ros

alias simulation_mode='export ROS_MASTER_URI=http://localhost:11311; export PS1="\[[44;1;37m\]<local>\[[0m\]\w$ "'
alias robot_mode='export ROS_MASTER_URI=http://localhost:11311; export PS1="\[[41;1;37m\]<turtlebot3>\[[0m\]\w$ "'

################################################################################

# Select the TurtleBot3 model name.
# https://emanual.robotis.com/docs/en/platform/turtlebot3/quick-start/#set-turtlebot3-model-name-1

export TURTLEBOT3_MODEL=waffle_pi

################################################################################

# Move to the working directory.
cd /root/TurtleBot3/
