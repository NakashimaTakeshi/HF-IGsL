#!/bin/bash

################################################################################

# Pin the versions of the core tools and packages for improved stability.
CONTAINERD_VERSION="1.4.12-1"
DOCKER_CE_VERSION="5:20.10.11~3-0~ubuntu-focal"
DOCKER_COMPOSE_VERSION="1.29.2"
NVIDIA_DOCKER_VERSION="2.8.0-1"
NVIDIA_RUNTIME_VERSION="3.7.0-1"

################################################################################

# Pass the 'sudo' privileges if previously granted in parent scripts.
if [ ! -z "$SUDO_USER" ]; then
  export USER=$SUDO_USER
fi

################################################################################

# Install Docker Community Edition.
# https://docs.docker.com/engine/install/ubuntu/

# Remove the older versions of Docker if any.
apt-get remove \
  docker \
  docker-engine \
  docker.io \
  containerd \
  runc

# Gather the required packages for Docker installation.
apt-get update && apt-get install -y \
  apt-transport-https \
  ca-certificates \
  curl \
  gnupg-agent \
  software-properties-common

# Add the official Docker GPG key.
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | apt-key add -
apt-key fingerprint 0EBFCD88

# Add the Docker repository.
add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu focal stable"

# Install Docker version 'DOCKER_CE_VERSION'.
# Any existing installation will be replaced.
apt-get update && apt-get install -y \
  docker-ce=${DOCKER_CE_VERSION} --allow-downgrades \
  docker-ce-cli=${DOCKER_CE_VERSION} \
  containerd.io=${CONTAINERD_VERSION}

# Test the Docker installation after making sure that the service is running.
service docker stop
service docker start
while ! pgrep dockerd > /dev/null; do
  sleep 1
done
docker version
docker run --rm hello-world

################################################################################

# Add the current user to the 'docker' group to run Docker without 'sudo'.
# Logging out and back in is required for the group change to take effect.
usermod -a -G docker ${USER}
echo "Added the current user '${USER}' to the 'docker' group."

# Configure the host system so that 'adduser' command adds future new users to the 'docker' group automatically.
# This enables new users to set up their environment without 'sudo' by only executing 'INCL-USER-ENV.sh'.
ADDUSER_CONFIG=/etc/adduser.conf
if [ ! -f ${ADDUSER_CONFIG} ]; then
  echo "Failed to add future new users to the 'docker' group because the system configuration file '${ADDUSER_CONFIG}' was not found."
else
  if ! grep -q "#EXTRA_GROUPS=\"dialout cdrom floppy audio video plugdev users\"" ${ADDUSER_CONFIG}; then
    echo "Failed to add future new users to the 'docker' group because 'EXTRA_GROUPS' in '${ADDUSER_CONFIG}' has already been customized."
  else
    sed -i 's/#EXTRA_GROUPS="dialout cdrom floppy audio video plugdev users"/EXTRA_GROUPS="dialout cdrom floppy audio video plugdev users docker"/' ${ADDUSER_CONFIG}
    sed -i 's/#ADD_EXTRA_GROUPS=1/ADD_EXTRA_GROUPS=1/' ${ADDUSER_CONFIG}
    echo "Modified '${ADDUSER_CONFIG}' to add all future new users to the 'docker' group upon creation."
  fi
fi

################################################################################

# Install Docker Compose.
# https://docs.docker.com/compose/install/
# https://github.com/docker/compose/releases

# Install Docker Compose version 'DOCKER_COMPOSE_VERSION'.
curl -L "https://github.com/docker/compose/releases/download/${DOCKER_COMPOSE_VERSION}/docker-compose-Linux-x86_64" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Install Bash command completion for Docker Compose version 'DOCKER_COMPOSE_VERSION'.
curl -L https://raw.githubusercontent.com/docker/compose/${DOCKER_COMPOSE_VERSION}/contrib/completion/bash/docker-compose -o /etc/bash_completion.d/docker-compose

# Test the Docker Compose installation.
docker-compose --version

################################################################################

# Install Nvidia Docker 2.
# https://github.com/NVIDIA/nvidia-docker/wiki/Installation-(version-2.0)
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker

# Remove 'nvidia-docker' and all existing GPU containers.
docker volume ls -q -f driver=nvidia-docker | xargs -r -I{} -n1 docker ps -q -a -f volume={} | xargs -r docker rm -f
apt-get purge -y nvidia-docker

# Add the Nvidia Docker package repositories.
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/ubuntu20.04/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list

# Install 'nvidia-docker2' version 'NVIDIA_DOCKER_VERSION' and reload the Docker daemon configuration.
apt-get update && apt-get install -y \
  nvidia-docker2=${NVIDIA_DOCKER_VERSION} \
  nvidia-container-runtime=${NVIDIA_RUNTIME_VERSION}

# Test the Nvidia Docker installation after making sure that the service is running and that Nvidia drivers are found.
service docker stop
service docker start
while ! pgrep dockerd > /dev/null; do
  sleep 1
done
if [ -e /proc/driver/nvidia/version ]; then
  docker run --runtime=nvidia --rm nvidia/cudagl:11.4.2-devel-ubuntu20.04 nvidia-smi
fi

################################################################################

# Install Terminator terminal.
# https://gnometerminator.blogspot.com/

# Install the latest version of Terminator from the Ubuntu repositories.
apt-get update && apt-get install -y \
  terminator

# Prevent the Terminator installation to replace the default Ubuntu terminal.
update-alternatives --set x-terminal-emulator /usr/bin/gnome-terminal.wrapper
