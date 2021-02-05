# General guidelines and recommendations for writing 'Dockerfile':
# https://docs.docker.com/engine/userguide/eng-image/dockerfile_best-practices/

################################################################################

# Pin the versions of the core tools and packages using global arguments for improved stability.
ARG nvidia_cudagl_version=10.1-devel-ubuntu18.04
ARG nvidia_cudnn_version=7.6.0.64-1+cuda10.1
ARG ros_desktop_version=1.4.1-0bionic.20210120.181615

################################################################################

# Install Ubuntu Bionic Beaver (18.04) with CUDA and OpenGL support.
# https://hub.docker.com/r/nvidia/cudagl/
# https://gitlab.com/nvidia/container-images/cudagl
# https://github.com/NVIDIA/libglvnd

# Pull the official parent image from the Nvidia repository.
FROM nvidia/cudagl:${nvidia_cudagl_version}
ARG nvidia_cudagl_version
ENV NVIDIA_CUDAGL_VERSION=${nvidia_cudagl_version}

# Install the Nvidia cuDNN library missing in the parent image.
# https://gitlab.com/nvidia/container-images/cuda/blob/ubuntu18.04/10.1/devel/cudnn7/Dockerfile
ARG nvidia_cudnn_version
ENV NVIDIA_CUDNN_VERSION=${nvidia_cudnn_version}
RUN apt-get update && apt-get install -y --no-install-recommends \
  libcudnn7=${NVIDIA_CUDNN_VERSION} \
  libcudnn7-dev=${NVIDIA_CUDNN_VERSION} \
  && apt-mark hold libcudnn7 \
  && rm -rf /var/lib/apt/lists/*

# Support Intel 3D acceleration when no Nvidia drivers are found:
# If the argument 'nvidia_gl' is 'false', remove the configuraion files related to the GL library provided by Nvidia and re-generate '/etc/ld.so.cache' by executing 'ldconfig'.
# This is required to avoid problems occurring when the host system does not have the GL library provided by Nvidia.
# In addition, remove the value of the environment variable 'LD_LIBRARY_PATH' which will be inherited from the base image.
# The variable is not necessary as long as the same information is provided by '/etc/ld.so.cache'.
ARG nvidia_gl="true"
RUN if [ "${nvidia_gl}" = "false" ]; then \
    rm /etc/ld.so.conf.d/nvidia.conf /etc/ld.so.conf.d/glvnd.conf; \
    ldconfig; \
  fi
ENV LD_LIBRARY_PATH=

# Set the default Docker runtime.
# By default, the 'nvidia/cudagl' image comes with Nvidia GL support.
# If no Nvidia GPU drivers are installed on the host, the image needs to be started with the runtime argument '--env DOCKER_RUNTIME=runc'.
# This is done at the container start-up by './scripts/initialize-docker-container.bash'.
ENV DOCKER_RUNTIME="nvidia"

# Remove the value of the environment variable 'LD_LIBRARY_PATH' which will be inherited from the base image.
# The variable is not necessary as long as the same information is provided by '/etc/ld.so.cache' but is breaking 'glxgears'.
ENV LD_LIBRARY_PATH=

################################################################################

# Install ROS Melodic Morenia.
# http://wiki.ros.org/melodic/Installation/Ubuntu
# http://packages.ros.org/ros/ubuntu/dists/bionic/main/binary-amd64/Packages

# Update the package list.
RUN echo "deb http://packages.ros.org/ros/ubuntu bionic main" > /etc/apt/sources.list.d/ros-latest.list

# Add the package keys.
RUN apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654

# Bypass the interactive installation of 'tzdata' included in 'ros-melodic-desktop-full'.
ENV DEBIAN_FRONTEND=noninteractive

# Install the 'ros-melodic-desktop-full' packages (including ROS, Rqt, Rviz, and more).
ARG ros_desktop_version
ENV ROS_DESKTOP_VERSION=${ros_desktop_version}
RUN apt-get update && apt-get install -y --no-install-recommends \
  ros-melodic-desktop-full=${ROS_DESKTOP_VERSION} \
  python-catkin-tools=0.6.1-1 \
  python-rosdep=0.20.0-1 \
  && rm -rf /var/lib/apt/lists/*

# Initialize 'rosdep'.
# http://wiki.ros.org/rosdep
RUN rosdep init

################################################################################

# Install the TurtleBot3 ROS dependencies.
# https://emanual.robotis.com/docs/en/platform/turtlebot3/quick-start/#pc-setup

# Gather the dependencies from the ROS repositories.
RUN apt-get update && apt-get install -y --no-install-recommends \
  ros-melodic-joy=1.14.0-1bionic.20210112.180610 \
  ros-melodic-teleop-twist-joy=0.1.3-0bionic.20210112.180932 \
  ros-melodic-teleop-twist-keyboard=1.0.0-1bionic.20210112.175758 \
  ros-melodic-laser-proc=0.1.5-0bionic.20210112.181310 \
  ros-melodic-rgbd-launch=2.2.2-0bionic.20210112.201433 \
  ros-melodic-depthimage-to-laserscan=1.0.8-0bionic.20210112.181847 \
  ros-melodic-rosserial-arduino=0.8.0-0bionic.20210112.200455 \
  ros-melodic-rosserial-python=0.8.0-0bionic.20210112.173857 \
  ros-melodic-rosserial-server=0.8.0-0bionic.20210112.181141 \
  ros-melodic-rosserial-client=0.8.0-0bionic.20210112.195855 \
  ros-melodic-rosserial-msgs=0.8.0-0bionic.20201017.034136 \
  ros-melodic-amcl=1.16.7-1bionic.20210112.184159 \
  ros-melodic-map-server=1.16.7-1bionic.20210112.181127 \
  ros-melodic-move-base=1.16.7-1bionic.20210112.195703 \
  ros-melodic-urdf=1.13.2-1bionic.20201017.083057 \
  ros-melodic-xacro=1.13.9-1bionic.20201017.081248 \
  ros-melodic-compressed-image-transport=1.9.5-0bionic.20210112.183412 \
  ros-melodic-rqt-image-view=0.4.16-1bionic.20210112.182133 \
  ros-melodic-gmapping=1.4.1-1bionic.20210112.191214 \
  ros-melodic-navigation=1.16.7-1bionic.20210112.200640 \
  ros-melodic-interactive-markers=1.11.5-1bionic.20210112.190840 \
  && rm -rf /var/lib/apt/lists/*

################################################################################

# Install dependencies for the 'flexbe_app' ROS package.
# https://github.com/FlexBE/flexbe_app

# Gather the dependencies from the Ubuntu repositories.
RUN apt-get update && apt-get install -y --no-install-recommends \
  libcanberra-gtk3-module=0.30-5ubuntu1 \
  libgconf2-dev=3.2.6-4ubuntu1 \
  libxtst6=2:1.2.3-1 \
  && rm -rf /var/lib/apt/lists/*

################################################################################

# Install common optional packages/tools.
RUN apt-get update && apt-get install -y --no-install-recommends \
  git=1:2.17.1-1ubuntu0.7 \
  inetutils-ping=2:1.9.4-3 \
  iproute2=4.15.0-2ubuntu1.1 \
  mesa-utils=8.4.0-1 \
  nano=2.9.3-2 \
  net-tools=1.60+git20161116.90da8a0-1ubuntu1 \
  openssh-client=1:7.6p1-4ubuntu0.3 \
  unzip=6.0-21ubuntu1 \
  wget=1.19.4-1ubuntu2.2 \
  && rm -rf /var/lib/apt/lists/*

################################################################################

# Set up the Bash shell environment.

# Define the working directory.
WORKDIR /root/

# Copy the helper scripts to the working directory.
COPY ./docker/rulo-devel/scripts/. /root/RULO/docker/rulo-devel/scripts/
RUN chmod -R +x /root/RULO/docker/rulo-devel/scripts/*

# Setup the Bash shell environment with '/root/.bashrc'.
# The value of 'TMP_HOSTNAME' is substituted by '../../RUN-DOCKER-CONTAINER.bash' when entering the container.
RUN echo "RULO_HOSTNAME=\"TMP_HOSTNAME\"" >> /root/.bashrc
RUN echo "source /root/RULO/docker/rulo-devel/scripts/initialize-bash-shell.bash" >> /root/.bashrc

# Force color prompt in terminal.
RUN sed -i 's/#force_color_prompt=yes/force_color_prompt=yes/' /root/.bashrc

# Pass variables for version traceability.
ARG DOCKERFILE_COMMIT_SHORT_SHA
ENV DOCKER_IMAGE_VERSION=${DOCKERFILE_COMMIT_SHORT_SHA}

# Overwrite the entry point of the parent image.
ENTRYPOINT []

# Enter the container with a Bash shell.
CMD ["/bin/bash"]
