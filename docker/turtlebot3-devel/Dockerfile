# General guidelines and recommendations for writing 'Dockerfile':
# https://docs.docker.com/engine/userguide/eng-image/dockerfile_best-practices/

################################################################################

# Pin the versions of the core tools and packages using global arguments for improved stability.
ARG nvidia_cudagl_version=11.2.2-devel-ubuntu20.04
ARG nvidia_cudnn_version=8.1.1.33-1+cuda11.2
ARG ros_desktop_version=1.5.0-1focal.20220512.132246
ARG python3_pip_version=22.1.2

################################################################################

# Install Ubuntu Focal Fossa (20.04) with CUDA 11 and OpenGL support.
# https://hub.docker.com/r/nvidia/cudagl/
# https://gitlab.com/nvidia/container-images/cudagl
# https://github.com/NVIDIA/libglvnd

# Pull the official parent image from the Nvidia repository.
FROM nvidia/cudagl:${nvidia_cudagl_version}
ARG nvidia_cudagl_version
ENV NVIDIA_CUDAGL_VERSION=${nvidia_cudagl_version}

# Install the Nvidia cuDNN 8 library missing in the parent image.
# https://gitlab.com/nvidia/container-images/cuda/-/blob/master/dist/11.2.0/ubuntu2004/devel/cudnn8/Dockerfile
ARG nvidia_cudnn_version
ENV NVIDIA_CUDNN_VERSION=${nvidia_cudnn_version}
RUN apt-get update && apt-get install -y --no-install-recommends \
  libcudnn8=${NVIDIA_CUDNN_VERSION} \
  libcudnn8-dev=${NVIDIA_CUDNN_VERSION} \
  && apt-mark hold libcudnn8 \
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

# Set up the time zone to avoid using 'DEBIAN_FRONTEND=noninteractive' inside 'Dockerfile' (best practice).
# This removes the interactive installation of 'tzdata' included in 'ros-noetic-desktop-full'.
ARG TIMEZONE="Asia/Tokyo"
RUN echo ${TIMEZONE} > /etc/timezone \
  && ln -s /usr/share/zoneinfo/${TIMEZONE} /etc/localtime

# Install 'tzdata' without interaction.
RUN apt-get update && apt-get install -y --no-install-recommends \
  tzdata=2022a-0ubuntu0.20.04 \
  && apt-get clean && rm -rf /var/lib/apt/lists/*

################################################################################

# Install ROS Noetic Ninjemys.
# http://wiki.ros.org/noetic/Installation/Ubuntu
# http://packages.ros.org/ros/ubuntu/dists/focal/main/binary-amd64/Packages

# Install prerequisite packages/tools.
RUN apt-get update && apt-get install -y --no-install-recommends \
  curl \
  && rm -rf /var/lib/apt/lists/*

# Update the package list.
RUN echo "deb http://packages.ros.org/ros/ubuntu focal main" > /etc/apt/sources.list.d/ros-latest.list

# Add the package keys.
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -

# Install the 'ros-noetic-desktop-full' packages (including ROS, Rqt, Rviz, and more).
ARG ros_desktop_version
ENV ROS_DESKTOP_VERSION=${ros_desktop_version}
RUN apt-get update && apt-get install -y --no-install-recommends \
  ros-noetic-desktop-full=${ROS_DESKTOP_VERSION} \
  python3-catkin-tools=0.8.5-1 \
  python3-rosdep=0.21.0-1 \
  && rm -rf /var/lib/apt/lists/*

# Initialize 'rosdep'.
# http://wiki.ros.org/rosdep
RUN rosdep init

################################################################################

# Install the TurtleBot3 ROS dependencies.
# https://emanual.robotis.com/docs/en/platform/turtlebot3/quick-start/#pc-setup

# Gather the dependencies from the ROS repositories.
RUN apt-get update && apt-get install -y --no-install-recommends \
  ros-noetic-joy=1.15.1-1focal.20220107.001731 \
  ros-noetic-teleop-twist-joy=0.1.3-1focal.20220107.002011 \
  ros-noetic-teleop-twist-keyboard=1.0.0-1focal.20220107.000515 \
  ros-noetic-laser-proc=0.1.6-1focal.20220107.001558 \
  ros-noetic-rgbd-launch=2.3.0-1focal.20220221.111123 \
  ros-noetic-rosserial-arduino=0.9.2-1focal.20220107.003948 \
  ros-noetic-rosserial-python=0.9.2-1focal.20220107.000351 \
  ros-noetic-rosserial-client=0.9.2-1focal.20220107.003714 \
  ros-noetic-rosserial-msgs=0.9.2-1focal.20210423.223234 \
  ros-noetic-amcl=1.17.1-1focal.20220221.101410 \
  ros-noetic-map-server=1.17.1-1focal.20220106.234201 \
  ros-noetic-move-base=1.17.1-1focal.20220221.111339 \
  ros-noetic-urdf=1.13.2-1focal.20220106.235206 \
  ros-noetic-xacro=1.14.13-1focal.20220212.160007 \
  ros-noetic-compressed-image-transport=1.14.0-1focal.20220221.102331 \
  ros-noetic-rqt=0.5.3-1focal.20220328.225111 \
  ros-noetic-rviz=1.14.14-1focal.20220212.160156 \
  ros-noetic-gmapping=1.4.2-1focal.20220107.002506 \
  ros-noetic-navigation=1.17.1-1focal.20220221.111820 \
  ros-noetic-interactive-markers=1.12.0-1focal.20220107.001131 \
  ros-noetic-multirobot-map-merge=2.1.4-1focal.20220107.005543 \
  && rm -rf /var/lib/apt/lists/*

################################################################################

# Install common optional packages/tools.
RUN apt-get update && apt-get install -y --no-install-recommends \
  git=1:2.25.1-1ubuntu3.4 \
  inetutils-ping=2:1.9.4-11ubuntu0.1 \
  iproute2=5.5.0-1ubuntu1 \
  mesa-utils=8.4.0-1build1 \
  nano=4.8-1ubuntu1 \
  net-tools=1.60+git20180626.aebd88e-1ubuntu1 \
  openssh-client=1:8.2p1-4ubuntu0.5 \
  unzip=6.0-25ubuntu1 \
  wget=1.20.3-1ubuntu2 \
  && rm -rf /var/lib/apt/lists/*

################################################################################

# Install common Python 3 packages/tools.

# Gather the dependencies from the Ubuntu repositories.
RUN apt-get update && apt-get install -y --no-install-recommends \
  python3-pip=20.0.2-5ubuntu1.6 \
  python3-setuptools=45.2.0-1 \
  python3-tk=3.8.10-0ubuntu1~20.04 \
  && rm -rf /var/lib/apt/lists/*

################################################################################

# Install the Python 3 dependencies of the Serket library.
# http://serket.naka-lab.org/getting_started.html
# https://github.com/naka-lab/Serket

# TODO:
# TensorFlow with GPU support was removed to upgrade to CUDA 11.
# NN, VAE, MVAE should not work anymore.

# Gather the dependencies from the Pip repositories.
ARG python3_pip_version
ENV PYTHON3_PIP_VERSION=${python3_pip_version}
RUN pip3 install --upgrade pip==${PYTHON3_PIP_VERSION}
RUN pip3 install \
  matplotlib==3.3.4 \
  numba==0.53.1 \
  scikit-learn==0.24.2

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
