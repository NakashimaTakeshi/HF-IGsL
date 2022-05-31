# General guidelines and recommendations for writing 'Dockerfile':
# https://docs.docker.com/engine/userguide/eng-image/dockerfile_best-practices/

################################################################################

# Pin the versions of the core tools and packages using global arguments for improved stability.
ARG nvidia_cudagl_version=10.0-devel-ubuntu18.04
ARG nvidia_cudnn_version=7.4.2.24-1+cuda10.0
ARG ros_desktop_version=1.4.1-0bionic.20220413.183028
ARG python3_pip_version=21.3.1
ARG python2_pip_version=20.3.4

################################################################################

# Install Ubuntu Bionic Beaver (18.04) with CUDA and OpenGL support.
# https://hub.docker.com/r/nvidia/cudagl/
# https://gitlab.com/nvidia/container-images/cudagl
# https://github.com/NVIDIA/libglvnd

# Pull the official parent image from the Nvidia repository.
FROM nvidia/cudagl:${nvidia_cudagl_version}
ARG nvidia_cudagl_version
ENV NVIDIA_CUDAGL_VERSION=${nvidia_cudagl_version}

# Fix Nvidia GPG keys.
# https://developer.nvidia.com/blog/updating-the-cuda-linux-gpg-repository-key/
RUN apt-key del 7fa2af80 \
  && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
  && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

# Install the Nvidia cuDNN library missing in the parent image.
# https://gitlab.com/nvidia/container-images/cuda/blob/ubuntu18.04/10.0/devel/cudnn7/Dockerfile
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
  python-rosdep=0.21.0-1 \
  && rm -rf /var/lib/apt/lists/*

# Initialize 'rosdep'.
# http://wiki.ros.org/rosdep
RUN rosdep init

################################################################################

# Install the TurtleBot3 ROS dependencies.
# https://emanual.robotis.com/docs/en/platform/turtlebot3/quick-start/#pc-setup

# Gather the dependencies from the ROS repositories.
RUN apt-get update && apt-get install -y --no-install-recommends \
  ros-melodic-joy=1.14.0-1bionic.20220127.152644 \
  ros-melodic-teleop-twist-joy=0.1.3-0bionic.20220127.152746 \
  ros-melodic-teleop-twist-keyboard=1.0.0-1bionic.20220127.145634 \
  ros-melodic-laser-proc=0.1.5-0bionic.20220127.152559 \
  ros-melodic-rgbd-launch=2.2.2-0bionic.20220222.193830 \
  ros-melodic-depthimage-to-laserscan=1.0.8-0bionic.20220222.185906 \
  ros-melodic-rosserial-arduino=0.8.0-0bionic.20220127.160859 \
  ros-melodic-rosserial-python=0.8.0-0bionic.20220127.145542 \
  ros-melodic-rosserial-server=0.8.0-0bionic.20220127.150809 \
  ros-melodic-rosserial-client=0.8.0-0bionic.20220127.160614 \
  ros-melodic-rosserial-msgs=0.8.0-0bionic.20210505.005832 \
  ros-melodic-amcl=1.16.7-1bionic.20220222.183836 \
  ros-melodic-map-server=1.16.7-1bionic.20220127.150654 \
  ros-melodic-move-base=1.16.7-1bionic.20220222.193709 \
  ros-melodic-urdf=1.13.2-1bionic.20220127.150505 \
  ros-melodic-xacro=1.13.17-1bionic.20220214.135922 \
  ros-melodic-compressed-image-transport=1.9.5-0bionic.20220222.190527 \
  ros-melodic-rqt-image-view=0.4.16-1bionic.20220328.223754 \
  ros-melodic-gmapping=1.4.1-1bionic.20220127.154525 \
  ros-melodic-navigation=1.16.7-1bionic.20220222.194313 \
  ros-melodic-interactive-markers=1.11.5-1bionic.20220127.155017 \
  ros-melodic-multirobot-map-merge=2.1.4-1bionic.20220127.162049 \
  && rm -rf /var/lib/apt/lists/*

################################################################################

# Upgrade Gazebo 9 to fix errors when loading the AWS worlds.
# http://gazebosim.org/tutorials?cat=install&tut=install_ubuntu&ver=9.0

# Install prerequisite packages/tools.
RUN apt-get update && apt-get install -y --no-install-recommends \
  wget=1.19.4-1ubuntu2.2 \
  && rm -rf /var/lib/apt/lists/*

# Update the package lists.
RUN echo "deb http://packages.osrfoundation.org/gazebo/ubuntu-stable bionic main" > /etc/apt/sources.list.d/gazebo-stable.list

# Add the package keys.
RUN wget https://packages.osrfoundation.org/gazebo.key -P /tmp \
  && apt-key add /tmp/gazebo.key \
  && rm /tmp/gazebo.key

# Upgrade Gazebo 9.
RUN apt-get update && apt-get install -y --no-install-recommends \
  gazebo9=9.19.0-1~bionic \
  && rm -rf /var/lib/apt/lists/*

# Upgrade 'libignition' to avoid symbol errors after upgrading Gazebo 9.
# https://qiita.com/seigot/items/b7ba0f41656f07eeeea0
# TODO: Find and pin 'libignition-math2' version.
RUN apt-get update && apt-get upgrade -y \
  libignition-math2 \
  && rm -rf /var/lib/apt/lists/*

################################################################################

# Install common optional packages/tools.
RUN apt-get update && apt-get install -y --no-install-recommends \
  git=1:2.17.1-1ubuntu0.11 \
  inetutils-ping=2:1.9.4-3 \
  iproute2=4.15.0-2ubuntu1.1 \
  mesa-utils=8.4.0-1 \
  nano=2.9.3-2 \
  net-tools=1.60+git20161116.90da8a0-1ubuntu1 \
  openssh-client=1:7.6p1-4ubuntu0.5 \
  unzip=6.0-21ubuntu1 \
  wget=1.19.4-1ubuntu2.2 \
  && rm -rf /var/lib/apt/lists/*

################################################################################

# Install common Python 3 packages/tools.

# Gather the dependencies from the Ubuntu repositories.
RUN apt-get update && apt-get install -y --no-install-recommends \
  python3-pip=9.0.1-2.3~ubuntu1.18.04.5 \
  python3-setuptools=39.0.1-2 \
  python3-tk=3.6.9-1~18.04 \
  && rm -rf /var/lib/apt/lists/*

################################################################################

# Install common Python 2 packages/tools.

# Gather the dependencies from the Ubuntu repositories.
RUN apt-get update && apt-get install -y --no-install-recommends \
  ntpdate=1:4.2.8p10+dfsg-5ubuntu7.3 \
  python-pip=9.0.1-2.3~ubuntu1.18.04.5 \
  python-tk=2.7.17-1~18.04 \
  && rm -rf /var/lib/apt/lists/*

################################################################################

# Install the Python 3 dependencies of the Serket library.
# (Including TensorFlow with GPU support.)
# http://serket.naka-lab.org/getting_started.html
# https://github.com/naka-lab/Serket
# https://www.tensorflow.org/install/pip

# Gather the dependencies from the Pip repositories.
ARG python3_pip_version
ENV PYTHON3_PIP_VERSION=${python3_pip_version}
RUN pip3 install --upgrade pip==${PYTHON3_PIP_VERSION}
RUN pip3 install \
  matplotlib==3.3.4 \
  numba==0.53.1 \
  scikit-learn==0.24.2 \
  tensorflow-gpu==1.13.1

################################################################################

# Install the Python 2 dependencies of the Serket library.
# (For integration with ROS Melodic.)
# http://serket.naka-lab.org/getting_started.html
# https://github.com/naka-lab/Serket

# Gather the dependencies from the Pip repositories.
ARG python2_pip_version
ENV PYTHON2_PIP_VERSION=${python2_pip_version}
RUN pip2 install --upgrade pip==${PYTHON2_PIP_VERSION}
RUN pip2 install \
  Janome==0.3.10 \
  llvmlite==0.31.0 \
  numba==0.47.0 \
  opencv-python==4.2.0.32

################################################################################

# Set up the Bash shell environment.

# Define the working directory.
WORKDIR /root/

# Copy the helper scripts to the working directory.
COPY ./docker/turtlebot3-devel/scripts/. /root/TurtleBot3/docker/turtlebot3-devel/scripts/
RUN chmod -R +x /root/TurtleBot3/docker/turtlebot3-devel/scripts/*

# Setup the Bash shell environment with '/root/.bashrc'.
# The value of 'TMP_HOSTNAME' is substituted by '../../RUN-DOCKER-CONTAINER.bash' when entering the container.
RUN echo "TURTLEBOT3_HOSTNAME=\"TMP_HOSTNAME\"" >> /root/.bashrc
RUN echo "source /root/TurtleBot3/docker/turtlebot3-devel/scripts/initialize-bash-shell.bash" >> /root/.bashrc

# Force color prompt in terminal.
RUN sed -i 's/#force_color_prompt=yes/force_color_prompt=yes/' /root/.bashrc

# Pass variables for version traceability.
ARG DOCKERFILE_COMMIT_SHORT_SHA
ENV DOCKER_IMAGE_VERSION=${DOCKERFILE_COMMIT_SHORT_SHA}

# Overwrite the entry point of the parent image.
ENTRYPOINT []

# Enter the container with a Bash shell.
CMD ["/bin/bash"]
