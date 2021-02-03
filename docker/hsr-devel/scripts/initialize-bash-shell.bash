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
# then show latest commit of the 'hsr-devel' folder with 'git log -n 1 --format="%h %aN %s %ad" -- $directory'.
DOCKERFILE_LATEST_HASH=$(git -C /root/HSR/ log -n 1 --no-merges --pretty=format:%h ./docker/hsr-devel/Dockerfile)
DOCKERFILE_LATEST_DATE=$(git -C /root/HSR/ log -n 1 --no-merges --pretty=format:%cd ./docker/hsr-devel/Dockerfile)
DOCKERFILE_CREATION_DATE=$(git -C /root/HSR/ show --no-patch --no-notes --pretty='%cd' ${DOCKER_IMAGE_VERSION})

echo -e "Container version: hsr-devel:${DOCKER_IMAGE_VERSION:0:7} from ${DOCKERFILE_CREATION_DATE}"
if [[ "${DOCKERFILE_CREATION_DATE}" != "${DOCKERFILE_LATEST_DATE}" ]]; then
  echo -e "Newer image available: hsr-devel:${DOCKERFILE_LATEST_HASH} from ${DOCKERFILE_LATEST_DATE}"
fi

################################################################################

# Source the ROS environment.
echo "Sourcing the ROS environment from '/opt/ros/melodic/setup.bash'."
source /opt/ros/melodic/setup.bash

# Source the Catkin workspace.
echo "Sourcing the Catkin workspace from '/root/HSR/catkin_ws/devel/setup.bash'."
source /root/HSR/catkin_ws/devel/setup.bash

################################################################################

# Add the Catkin workspace to the 'ROS_PACKAGE_PATH'.
export ROS_PACKAGE_PATH=${ROS_PACKAGE_PATH}:/root/HSR/catkin_ws/src/

################################################################################

# Add the Gazebo models of the 'sdewg_wrs_fcsc_gazebo_worlds' ROS package to the 'GAZEBO_MODEL_PATH'.
export GAZEBO_MODEL_PATH=${GAZEBO_MODEL_PATH}:/root/HSR/catkin_ws/src/sdewg_wrs_fcsc_gazebo_worlds/models/

# Add the Gazebo models of the 'sdewg_wrs_prc_gazebo_worlds' ROS package to the 'GAZEBO_MODEL_PATH'.
export GAZEBO_MODEL_PATH=${GAZEBO_MODEL_PATH}:/root/HSR/catkin_ws/src/sdewg_wrs_prc_gazebo_worlds/models/

################################################################################

# Define Bash functions to conveniently execute the helper scripts in the current shell process.

function hsr-fix-git-paths () {
  # Store the current directory and execute scripts in the current shell process.
  pushd .
  source /root/HSR/docker/hsr-devel/scripts/fix-git-paths.bash
  popd
}

function hsr-initialize-catkin-workspace () {
  # Store the current directory and execute scripts in the current shell process.
  pushd .
  source /root/HSR/docker/hsr-devel/scripts/fix-git-paths.bash
  source /root/HSR/docker/hsr-devel/scripts/fix-permission-issues.bash
  source /root/HSR/docker/hsr-devel/scripts/initialize-catkin-workspace.bash
  popd
}

function hsr-reset-catkin-workspace () {
  # Store the current directory and execute scripts in the current shell process.
  pushd .
  source /root/HSR/docker/hsr-devel/scripts/fix-git-paths.bash
  source /root/HSR/docker/hsr-devel/scripts/fix-permission-issues.bash
  source /root/HSR/docker/hsr-devel/scripts/reset-catkin-workspace.bash
  popd
}

function hsr-fix-permission-issues () {
  # Store the current directory and execute scripts in the current shell process.
  pushd .
  source /root/HSR/docker/hsr-devel/scripts/fix-git-paths.bash
  source /root/HSR/docker/hsr-devel/scripts/fix-permission-issues.bash
  popd
}

function hsr-download-model-data () {
  # Store the current directory and execute scripts in the current shell process.
  pushd .
  source /root/HSR/docker/hsr-devel/scripts/fix-permission-issues.bash
  source /root/HSR/docker/hsr-devel/scripts/download-model-data.bash
  popd
}

function hsr-get-fully-started () {
  # Store the current directory and execute scripts in the current shell process.
  pushd .
  source /root/HSR/docker/hsr-devel/scripts/fix-git-paths.bash
  source /root/HSR/docker/hsr-devel/scripts/fix-permission-issues.bash
  source /root/HSR/docker/hsr-devel/scripts/download-model-data.bash
  source /root/HSR/docker/hsr-devel/scripts/reset-catkin-workspace.bash
  popd
}

################################################################################

# Set HSR/ROS network interface.
# https://docs.hsr.io/manual_en/howto/pc_install.html
# https://docs.hsr.io/manual_en/howto/network_settings.html

# Look for the robot host name inside the local network and resolve it to get its IP address.
# The value of 'HSRB_HOSTNAME' should be initialized in '~/.bashrc' by './RUN-DOCKER-CONTAINER.bash' when entering the container.
HSRB_IP=`getent hosts ${HSRB_HOSTNAME} | cut -d ' ' -f 1`

if [ -z "${HSRB_IP}" ]; then
  # If no robot host name is found, set the local 'ROS_IP' using the default Docker network interface ('docker0').
  export ROS_IP=$(LANG=C /sbin/ifconfig docker0 | grep -Eo 'inet (addr:)?([0-9]*\.){3}[0-9]*' | grep -Eo '([0-9]*\.){3}[0-9]*')
else
  # If the robot host name is found, set the local 'ROS_IP' using the network interface that connects to the robot.
  # TODO: Use Bash instead of Python.
  export ROS_IP=`python /root/HSR/docker/hsr-devel/scripts/print-interface-ip.py ${HSRB_IP}`
fi
if [ -z "${ROS_IP}" ]; then
  # If the local 'ROS_IP' is still empty, default to the Docker network interface ('docker0') for sanity.
  export ROS_IP=$(LANG=C /sbin/ifconfig docker0 | grep -Eo 'inet (addr:)?([0-9]*\.){3}[0-9]*' | grep -Eo '([0-9]*\.){3}[0-9]*')
fi
echo "ROS_IP is set to '${ROS_IP}'."

export ROS_HOME=~/.ros

alias sim_mode='export ROS_MASTER_URI=http://localhost:11311; export PS1="\[[44;1;37m\]<local>\[[0m\]\w$ "'
alias hsrb_mode='export ROS_MASTER_URI=http://hsrb.local:11311; export PS1="\[[41;1;37m\]<hsrb>\[[0m\]\w$ "'

################################################################################

# Move to the working directory.
cd /root/HSR/
