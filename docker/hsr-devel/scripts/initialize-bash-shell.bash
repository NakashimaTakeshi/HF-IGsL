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
# then show latest commit of the 'rulo-devel' folder with 'git log -n 1 --format="%h %aN %s %ad" -- $directory'.
DOCKERFILE_LATEST_HASH=$(git -C /root/RULO/ log -n 1 --no-merges --pretty=format:%h ./docker/rulo-devel/Dockerfile)
DOCKERFILE_LATEST_DATE=$(git -C /root/RULO/ log -n 1 --no-merges --pretty=format:%cd ./docker/rulo-devel/Dockerfile)
DOCKERFILE_CREATION_DATE=$(git -C /root/RULO/ show --no-patch --no-notes --pretty='%cd' ${DOCKER_IMAGE_VERSION})

echo -e "Container version: rulo-devel:${DOCKER_IMAGE_VERSION:0:7} from ${DOCKERFILE_CREATION_DATE}"
if [[ "${DOCKERFILE_CREATION_DATE}" != "${DOCKERFILE_LATEST_DATE}" ]]; then
  echo -e "Newer image available: rulo-devel:${DOCKERFILE_LATEST_HASH} from ${DOCKERFILE_LATEST_DATE}"
fi

################################################################################

# Source the ROS environment.
echo "Sourcing the ROS environment from '/opt/ros/melodic/setup.bash'."
source /opt/ros/melodic/setup.bash

# Source the Catkin workspace.
echo "Sourcing the Catkin workspace from '/root/RULO/catkin_ws/devel/setup.bash'."
source /root/RULO/catkin_ws/devel/setup.bash

################################################################################

# Add the Catkin workspace to the 'ROS_PACKAGE_PATH'.
export ROS_PACKAGE_PATH=${ROS_PACKAGE_PATH}:/root/RULO/catkin_ws/src/

################################################################################

# Define Bash functions to conveniently execute the helper scripts in the current shell process.

function rulo-fix-git-paths () {
  # Store the current directory and execute scripts in the current shell process.
  pushd .
  source /root/RULO/docker/rulo-devel/scripts/fix-git-paths.bash
  popd
}

function rulo-initialize-catkin-workspace () {
  # Store the current directory and execute scripts in the current shell process.
  pushd .
  source /root/RULO/docker/rulo-devel/scripts/fix-git-paths.bash
  source /root/RULO/docker/rulo-devel/scripts/fix-permission-issues.bash
  source /root/RULO/docker/rulo-devel/scripts/initialize-catkin-workspace.bash
  popd
}

function rulo-reset-catkin-workspace () {
  # Store the current directory and execute scripts in the current shell process.
  pushd .
  source /root/RULO/docker/rulo-devel/scripts/fix-git-paths.bash
  source /root/RULO/docker/rulo-devel/scripts/fix-permission-issues.bash
  source /root/RULO/docker/rulo-devel/scripts/reset-catkin-workspace.bash
  popd
}

function rulo-fix-permission-issues () {
  # Store the current directory and execute scripts in the current shell process.
  pushd .
  source /root/RULO/docker/rulo-devel/scripts/fix-git-paths.bash
  source /root/RULO/docker/rulo-devel/scripts/fix-permission-issues.bash
  popd
}

function rulo-download-model-data () {
  # Store the current directory and execute scripts in the current shell process.
  pushd .
  source /root/RULO/docker/rulo-devel/scripts/fix-permission-issues.bash
  source /root/RULO/docker/rulo-devel/scripts/download-model-data.bash
  popd
}

function rulo-get-fully-started () {
  # Store the current directory and execute scripts in the current shell process.
  pushd .
  source /root/RULO/docker/rulo-devel/scripts/fix-git-paths.bash
  source /root/RULO/docker/rulo-devel/scripts/fix-permission-issues.bash
  source /root/RULO/docker/rulo-devel/scripts/download-model-data.bash
  source /root/RULO/docker/rulo-devel/scripts/reset-catkin-workspace.bash
  popd
}

################################################################################

# Set RULO/ROS network interface.

# Look for the robot host name inside the local network and resolve it to get its IP address.
# The value of 'RULO_HOSTNAME' should be initialized in '~/.bashrc' by './RUN-DOCKER-CONTAINER.bash' when entering the container.
RULO_IP=`getent hosts ${RULO_HOSTNAME} | cut -d ' ' -f 1`

if [ -z "${RULO_IP}" ]; then
  # If no robot host name is found, set the local 'ROS_IP' using the default Docker network interface ('docker0').
  export ROS_IP=$(LANG=C /sbin/ifconfig docker0 | grep -Eo 'inet (addr:)?([0-9]*\.){3}[0-9]*' | grep -Eo '([0-9]*\.){3}[0-9]*')
else
  # If the robot host name is found, set the local 'ROS_IP' using the network interface that connects to the robot.
  # TODO: Use Bash instead of Python.
  export ROS_IP=`python /root/RULO/docker/rulo-devel/scripts/print-interface-ip.py ${RULO_IP}`
fi
if [ -z "${ROS_IP}" ]; then
  # If the local 'ROS_IP' is still empty, default to the Docker network interface ('docker0') for sanity.
  export ROS_IP=$(LANG=C /sbin/ifconfig docker0 | grep -Eo 'inet (addr:)?([0-9]*\.){3}[0-9]*' | grep -Eo '([0-9]*\.){3}[0-9]*')
fi
echo "ROS_IP is set to '${ROS_IP}'."

export ROS_HOME=~/.ros

alias sim_mode='export ROS_MASTER_URI=http://localhost:11311; export PS1="\[[44;1;37m\]<local>\[[0m\]\w$ "'
alias rulo_mode='export ROS_MASTER_URI=http://rulo.local:11311; export PS1="\[[41;1;37m\]<rulo>\[[0m\]\w$ "'

################################################################################

# Move to the working directory.
cd /root/RULO/
