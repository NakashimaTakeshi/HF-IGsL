# TurtleBot3 Software Development Environment (SDE)

This project aims to develop a shared platform for collaborative research and development on the TurtleBot3 robot at the [Emergent Systems Laboratory](http://www.em.ci.ritsumei.ac.jp/) of Ritsumeikan University by providing a robust, scalable, virtualized, and documented environment to all contributors.

## Content

[[_TOC_]]

## External Resources

Consult the links below to access external resources related to the TurtleBot3 project.

### Official Websites

*   TurtleBot3 official manual: [https://emanual.robotis.com/docs/en/platform/turtlebot3/overview/](https://emanual.robotis.com/docs/en/platform/turtlebot3/overview/).
*   TurtleBot3 GitHub repository: [https://github.com/ROBOTIS-GIT/turtlebot3](https://github.com/ROBOTIS-GIT/turtlebot3).
*   TurtleBot3 ROS wiki: [http://wiki.ros.org/turtlebot3](http://wiki.ros.org/turtlebot3).

### Related Publications

Summaries of the development history, requirements, vision, and goals of the SDE have been published in the following papers:

*   L. El Hafi, G. A. Garcia Ricardez, F. von Drigalski, Y. Inoue, M. Yamamoto, and T. Yamamoto, "**Software Development Environment for Collaborative Research Workflow in Robotic System Integration**", in *RSJ Advanced Robotics (AR), Special Issue on Software Framework for Robot System Integration*, vol. 36, no. 11, pp. 533-547, Jun. 3, 2022. DOI: [https://doi.org/10.1080/01691864.2022.2068353](https://doi.org/10.1080/01691864.2022.2068353)
*   L. El Hafi and T. Yamamoto, "**Toward the Public Release of a Software Development Environment for Human Support Robots**", in *Proceedings of 2020 Annual Conference of the Robotics Society of Japan (RSJ 2020)*, ref. RSJ2020AC3E1-01, pp. 1-2, (Virtual), Oct. 9, 2020.
*   L. El Hafi, S. Matsuzaki, S. Itadera, and T. Yamamoto, "**Deployment of a Containerized Software Development Environment for Human Support Robots**", in *Proceedings of 2019 Annual Conference of the Robotics Society of Japan (RSJ 2019)*, ref. RSJ2019AC3K1-03, pp. 1-2, Tokyo, Japan, Sep. 3, 2019.
*   L. El Hafi, Y. Hagiwara, and T. Taniguchi, "**Abstraction-Rich Workflow for Agile Collaborative Development and Deployment of Robotic Solutions**", in *Proceedings of 2018 Annual Conference of the Robotics Society of Japan (RSJ 2018)*, ref. RSJ2018AC3D3-02, pp. 1-3, Kasugai, Japan, Sep. 5, 2018.

> **Note:**
  Cite these papers if you are using the SDE to implement your research!
  It is crucially important to desseminate the SDE accros the robotics research community.

## Contribution Guidelines

Carefully read the contribution guidelines before pushing new code or requesting a merge.
Details can be found in the contribution guide: [CONTRIBUTING.md](CONTRIBUTING.md).

## Reporting Issues

Please report any issue or future work using the GitLab issue tracker: [https://gitlab.com/emlab/TurtleBot3/issues](https://gitlab.com/emlab/TurtleBot3/issues).

## Getting Started

Follow this step-by-step guide to perform the initial setup of the TurtleBot3 project on a development machine running [Ubuntu](https://www.ubuntu.com/).

> **Note:**
  The whole development environment of the TurtleBot3 project is containerized/virtualized using [Docker](https://www.docker.com/).
  This ensures that all the contributors work in the exact same software environment.

> **Note:**
  The TurtleBot3 robot and simulator interfaces are primarily implemented with [ROS](http://www.ros.org/).

> **Note:**
  The TurtleBot3 project relies heavily on both hardware 3D graphic acceleration and Nvidia [CUDA](https://developer.nvidia.com/cuda-toolkit), thus a discrete Nvidia GPU is highly recommended although Intel CPU graphic acceleration is also supported.
  In addition, virtual machines are only partially supported: 3D accelerated tools such as [Rviz](https://github.com/ros-visualization/rviz) and [Gazebo](http://gazebosim.org/) will crash at runtime unless the same virtual machine drivers are installed in both the Docker container and the virtual machine.

> **Note:**
  Although this initial setup is meant to be performed only once, you can run it again would you want to reset the development environment as the project evolves.
  However, make sure to backup your latest changes beforehand.

### Step 0: Verify the Prerequisites

**Mandatory:**

*   A development machine running Ubuntu 20.04 LTS (Focal Fossa) based on the AMD64 architecture.
*   Access to administrator privileges (`sudo`) on the Ubuntu machine.
*   Access to developer privileges on the GitLab project at [https://gitlab.com/emlab/TurtleBot3](https://gitlab.com/emlab/TurtleBot3).

**Recommended:**

*   A [Robotis TurtleBot3](https://emanual.robotis.com/docs/en/platform/turtlebot3/overview/) robot for full operability.
    If not, the TurtleBot3 simulator provides support for basic operations.
*   An Nvidia GPU capable of running CUDA 11.2, or newer, to accelerate 3D graphics and deep learning computing.
*   A properly configured gitlab.com account linked with your personal SSH key to push contributions to the project repository: https://docs.gitlab.com/ee/ssh/.

### Step 1: Set up the Development Environment

Set up the environment of the development machine with the following instructions.

1.   Install [Git](https://git-scm.com/) if necessary:

     ```shell
     sudo apt-get update && sudo apt-get install -y git
     ```

2.   Clone the TurtleBot3 project repository in your home folder:

     ```shell
     git clone -b HF-PGM_MRSSM-otake https://gitlab.com/emlab/TurtleBot3.git
     ```

     Enter your GitLab developer credentials if prompted.
3.   Configure the system environment:

     ```shell
     cd ./TurtleBot3/ && bash ./SETUP-DEVEL-MACHINE.bash
     ```

     The execution of `SETUP-DEVEL-MACHINE.bash` requires `sudo` permissions to install the tools that allow virtualization, i.e., Docker, [Docker Compose](https://github.com/docker/compose), and [Nvidia Docker 2](https://github.com/NVIDIA/nvidia-docker).
     System changes made with `sudo` are kept to a strict minimum.
4.   Reboot the system (or log out and back in) for the changes to users and groups to take effect:

     ```shell
     sudo reboot
     ```

> **Note:**
  The `SETUP-DEVEL-MACHINE.bash` script is actually divided into `INCL-SUDO-ENV.bash` and `INCL-USER-ENV.bash`.
  The execution of `INCL-SUDO-ENV.bash` makes system-wide changes and thus requires `sudo` permissions.
  However, if your system has already all the necessary tools installed, you can directly set up your local user environment with `cd ~/TurtleBot3/ && bash ./INCL-USER-ENV.bash` which does not require `sudo` permissions.

> **Note:**
  You do not need to reboot if your local user has already been added to the `docker` group.
  If so, executing `docker --version` should not ask for `sudo`.
  In principle, you only need to reboot after the very first time you run `SETUP-DEVEL-MACHINE.bash`.

### Step 2: Build the Docker Image

Create a virtual environment using Docker (= Docker image) on the development machine with the following instructions.

1.   Build the Docker image:

     ```shell
     cd ./TurtleBot3/ && bash ./BUILD-DOCKER-IMAGE.bash gitlab-ci
     ```

     This script builds the image following the instructions found in `~/TurtleBot3/docker/turtlebot3-devel/Dockerfile`.
     Enter your GitLab developer credentials if prompted.

> **Note:**
  The script `BUILD-DOCKER-IMAGE.bash` first tries to download the image from the [container registry](https://gitlab.com/emlab/TurtleBot3/container_registry) of the project.
  If, for whatever reason, the image cannot be downloaded, the script will build it locally.
  The later process is slow and can take up to 1 hour.
  In both cases, avoid using a Wi-Fi connection to greatly accelerate the process.
  Note that future local builds will reuse cached data whenever possible.

### Step 3: Run the Docker Container

Enter a virtual instance of the Docker image (= Docker container) on the development machine with the following instructions.

1.   Run the Docker container:

     ```shell
     cd ~/TurtleBot3/ && bash ./RUN-DOCKER-CONTAINER.bash
     ```

     This script creates or updates the container following the instructions found in `~/TurtleBot3/docker/docker-compose.yml`.
     It allows the container to share system resources, such as volumes and devices, with the development machine.
2.   Configure the freshly built Docker container by executing the following Bash function inside it:

     ```shell
     sde-get-fully-started
     ```

     This function calls several scripts to download the required datasets and build the ROS environment (= Catkin workspace) inside the Docker container.
     It will remove any existing Catkin workspace and build the new one inside `/root/TurtleBot3/catkin_ws/`.
     It will also automatically source the newly build ROS environment.
4.   From there, you have everything ready to start using the SDE to develop new features.
3.   Use `ctrl+d` to exit the container at any time.

> **Note:**
  If no Nvidia drivers are present, the script `RUN-DOCKER-CONTAINER.bash` sets the Docker runtime to `runc`, instead of `nvidia`, to bypass `nvidia-docker2` when entering the container.
  In this case, most 3D accelerated tools, including the TurtleBot3 simulator, will be extremely slow to run.

> **Note:**
  The script `RUN-DOCKER-CONTAINER.bash` will try to resolve host name `turtlebot3-01.local` of the TurtleBot3 robot and add its IP address to `/etc/hosts`.
  You can confirm the result with `ping turtlebot3-01.local` from inside the container.
  Note that you can ignore the errors if you do not plan to use the TurtleBot3 robot at that time.

> **Note:**
  Be careful if you need to modify `docker-compose.yml` as the container will be recreated from scratch the next time you run `RUN-DOCKER-CONTAINER.bash`.

> **Note:**
  Note that the Bash function `sde-get-fully-started` should be used only once after instantiating a new container from the Docker image.

### Step 4: Learn the Advanced Functions

The development environment inside the Docker container offers several useful functions that you should be aware of.
These advanced functions will help you increase both the convenience and the quality of your work for the TurtleBot3 project.

#### Custom Bash Functions

The development environment contains several useful Bash functions, all starting with the prefix `sde-`, to make your work more convenient.
Including, but not limited to:

*   `sde-download-model-data`: Download from the cloud all the large binary files required at runtime (models, datasets, dictionaries, weights, etc.).
*   `sde-build-catkin-workspace`: Build and source the Catkin workspace on top of the system ROS environment.
*   `sde-reset-catkin-workspace`: Remove built artifacts, then cleanly rebuild and source the Catkin workspace on top of the system ROS environment (to use after, for example, switching branches).
*   `sde-fix-permission-issues`: Fix the various permission issues that may appear when manipulating, on the development machine, files generated by the `root` user of the Docker container.
*   `sde-cleanup-output-data`: Delete files and data output at runtime to clean up the environment between experiment trials.
*   `sde-get-fully-started`: Execute several of the aforementioned functions to quickly get started when entering a freshly built Docker container.

> **Note:**
  These Bash functions are based on helper scripts that can be found in `/root/TurtleBot3/docker/turtlebot3-devel/scripts/` in the Docker container or in `~/TurtleBot3/docker/turtlebot3-devel/scripts/` in the development machine.
  You can see their definitions in `~/.bashrc` inside the container.

> **Note:**
  The script `build-catkin-workspace.bash` will build the Catkin workspace using `catkin build` instead of the older but still officially default `catkin_make`.
  Please be sure to build using `catkin build` to avoid strange issues.

#### Multiple Terminal Operation

You can simultaneously run multiple terminals using `RUN-TERMINATOR-TERMINAL.bash`.
This script opens [Terminator](https://gnometerminator.blogspot.com/) with the default layout configuration stored in `~/TurtleBot3/terminator/config`.
Each sub-terminal automatically executes `RUN-DOCKER-CONTAINER.bash` with a predefined ROS launch file for convenience.
You can then select execution options by pressing specific keys, as shown in the example below:

```
Run 'example_roslaunch_file.launch'? Press:
'r' to run with the robot,
's' to run in the simulator,
'c' to enter a child shell,
'q' to quit.
```

#### Configuration of the Docker Container

A cascade of scripts performs the initialization of the Docker container.
Although the boundaries between them can sometimes be blurry, each one has a specific function, and future implementations/revisions should keep these functions separated as much as possible.
They can be found on the development machine at:

*   `~/TurtleBot3/docker/turtlebot3-devel/Dockerfile`: Used by `BUILD-DOCKER-IMAGE.bash` to create the image of the shared development environment.
    It mainly describes the installations of the project tools/dependencies required inside the container.
*   `~/TurtleBot3/docker/docker-compose.yml`: Describes the interface between the development machine and the container.
    It includes external information from outside the container, such as device host names or network configuration.
    It is invoked when the container is (re)started, most frequently the first time that `RUN-DOCKER-CONTAINER.bash` is run after the container is either stopped (development machine reboot) or rebuilt.
*   `~/TurtleBot3/docker/turtlebot3-devel/scripts/initialize-docker-container.bash`: A startup script referenced in `docker-compose.yml` and thus also invoked when (re)starting the container, most frequently the first time that `RUN-DOCKER-CONTAINER.bash` is run after the container is either stopped (development machine reboot) or rebuilt.
    It executes the internal commands needed at the startup of the container (for example, to maintain the container running in the background).
*   `~/TurtleBot3/docker/turtlebot3-devel/scripts/initialize-bash-shell.bash`: A shell script appended to `/root/.bashrc` at the end of the `Dockerfile` to set up the Bash shell environment inside the container (terminal colors, shell functions, symbolic links, binary paths, environment sourcing, etc.).
    Theoretically, its content could be written directly in the `Dockerfile`, but this would result in complex lines of code as helpful Bash syntaxes like here-documents are not supported by Docker.
    It is automatically sourced by `/root/.bashrc` every time the user enters the Bash shell inside the container, most likely every time `RUN-DOCKER-CONTAINER.bash` is run.

### Step 5: Develop on the Simulator/Robot

From here, you can continue with one or both of the following options depending on the presence of a TurtleBot3 robot within the same local network as the development machine.

1.   Enter the Docker container from a new terminal window:

     ```shell
     cd ~/TurtleBot3/ && bash ./RUN-DOCKER-CONTAINER.bash
     ```

     Then connect or switch to `roscore` used by the Gazebo simulator or by the TurtleBot3 robot with the following aliases:

     ```shell
     simulation_mode
     robot_mode
     ```

2.   You can also conveniently run multiple terminals simultaneously using Terminator by running:

     ```shell
     cd ~/TurtleBot3/ && bash ./RUN-TERMINATOR-TERMINAL.bash simulation
     ```

     Or:

     ```shell
     cd ~/TurtleBot3/ && bash ./RUN-TERMINATOR-TERMINAL.bash robot
     ```

Finally, before writing any new code, please make sure to have read the contribution guidelines in: [CONTRIBUTING.md](CONTRIBUTING.md).

> **Note:**
  The most important rule is to avoid pushing large binary files (datasets, weights, etc.) in the repository.
  Instead, you need to provide a link to download all your necessary large binary files from the cloud with `sde-download-model-data`.
  Ideally, all these files should be centralized in an online storage.

## Robot Configuration

This project is configured to use up to 3 TurtleBot3 robots simultaneously.
The default configuration of the TurtleBot3 on-board computers is as follows, where `x` is the number corresponding to a specific robot:

*   User name: `pi`
*   Host name: `turtlebot3-0x.local`
*   Password: `turtlebot`

To connect to the Raspberry Pi on-board computer of the TurtleBot3, run the following in a new terminal on the development machine:

```shell
ping turtlebot3-0x.local
ssh pi@turtlebot3-0x.local
```

You can also bring up the TurtleBot3 robots by executing the following script on the development machine:

```shell
cd ~/TurtleBot3/ && bash ./BRINGUP-TURTLEBOT3-ROBOT.bash
```

> **Note:**
  Do not, under any circumstance, directly modify the system of the on-board computer of the TurtleBot3 robot.
  Always use the environment of the development machine to interact with the robot using ROS over the local network.

## Robot Installation

Follow this step-by-step guide to set up or reinitialize a TurtleBot3 robot.
`x` is the number corresponding to a specific robot.

> **Note:**
  The instructions below contains only minor modifications from the TurtleBot3 official manual, mainly to support mutiple robots and host names.

> **Note:**
  Each TurtleBot3 robot is set up using the older ROS Kinetic distribution because the ROS package that interfaces the Camera Module v2 is not compatible with the newer ROS distributions.
  Therefore, the ROS distribution in running in the robots does not match the ROS distribution of the development machine.
  This is not an optimal solution, but so far, it did not cause important issues.

1.   Read the following page of the TurtleBot3 official manual: [https://emanual.robotis.com/docs/en/platform/turtlebot3/sbc_setup/#sbc-setup](https://emanual.robotis.com/docs/en/platform/turtlebot3/sbc_setup/#sbc-setup).
1.   Download the following TurtleBot3 SBC Image: Raspberry Pi 3B+ ROS Kinetic on Raspberry Pi OS (Raspbian OS).  
Link: [http://www.robotis.com/service/download.php?no=1738](http://www.robotis.com/service/download.php?no=1738).  
SHA256: `eb8173f3727db08087990b2c4e2bb211e70bd54644644834771fc8b971856b97`.
1.   Flash the image on the microSD card of the TurtleBot3.
1.   Insert the microSD card into the Raspberry Pi.
1.   Connect a monitor to the HDMI port of Raspberry Pi.
1.   Connect input devices (keyboard, mouse) to the USB ports of Raspberry Pi
1.   Connect the power (either with USB or OpenCR) to turn on the Raspberry Pi.
1.   After booting up the Raspberry Pi, use the GUI and input devices to connect to the Wi-Fi.
     The Raspberry Pi will automatically connect to known Wi-Fi networks after reboot.
1.   Open the "Raspberry Pi Configuration" menu and set the host name to `turtlebot3-0x`.
1.   In the same "Raspberry Pi Configuration" menu, verify that the overscan is disabled, and that both SSH and the camera are enabled.
1.   Reboot the Raspberry Pi and disconnect the monitor and input devices as the next steps can be performed from a remote development machine via SSH.
1.   Connect to the Raspberry Pi via SSH (password: `turtlebot`):

     ```shell
     ping turtlebot3-0x.local
     ssh pi@turtlebot3-0x.local
     ```

1.   Inside the Raspberry Pi, synchronize the time:

     ```shell
     sudo apt-get install ntpdate
     sudo ntpdate ntp.ubuntu.com
     ```

1.   Inside the Raspberry Pi, fix the ROS GPG key issue (see: [https://discourse.ros.org/t/ros-gpg-key-expiration-incident/20669](https://discourse.ros.org/t/ros-gpg-key-expiration-incident/20669)):

     ```shell
     curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
     ```

1.   Inside the Raspberry Pi, expand the filesystem:

     ```shell
     sudo raspi-config
     ```
     
     Select "7 Advanced Options" > "A1 Expand Filesystem" then exit and reboot.
1.   Connect once more to the Raspberry Pi via SSH (password: `turtlebot`):

     ```shell
     ping turtlebot3-0x.local
     ssh pi@turtlebot3-0x.local
     ```

1.   Inside the Raspberry Pi, add the ROS network information to the Bash shell by running `nano ~/.bashrc` and adding the following lines at the bottom of the file:

     ```shell
     export ROS_MASTER_URI=http://master.local:11311
     export ROS_HOSTNAME=turtlebot3-0x.local
     export TURTLEBOT3_MODEL=waffle_pi
     ```

     Note that `x` is the number corresponding to the robot and `master` is the host name of the development machine.
     Note also that `TURTLEBOT3_MODEL=waffle_pi`.
     Save the file and load the new settings into the current Bash shell with `source ~/.bashrc`.
1.   Do not install the "NEW LDS-02 Configuration" unless you use TurtleBot3 robots from 2022 or later.
1.   Read the following page of the TurtleBot3 official manual: [https://emanual.robotis.com/docs/en/platform/turtlebot3/opencr_setup/](https://emanual.robotis.com/docs/en/platform/turtlebot3/opencr_setup/).
1.   Inside the Raspberry Pi, update the OpenCR firmware with:

     ```shell
     sudo dpkg --add-architecture armhf
     sudo apt-get update
     sudo apt-get install libc6:armhf
     export OPENCR_PORT=/dev/ttyACM0
     export OPENCR_MODEL=waffle
     rm -rf ./opencr_update.tar.bz2
     wget https://github.com/ROBOTIS-GIT/OpenCR-Binaries/raw/master/turtlebot3/ROS1/latest/opencr_update.tar.bz2 
     tar -xvf opencr_update.tar.bz2 
     cd ./opencr_update
     ./update.sh $OPENCR_PORT $OPENCR_MODEL.opencr
     cd ~/
     ```

     Note that `OPENCR_MODEL=waffle`.
1.   Read the following page of the TurtleBot3 official manual: [https://emanual.robotis.com/docs/en/platform/turtlebot3/appendix_raspi_cam/](https://emanual.robotis.com/docs/en/platform/turtlebot3/appendix_raspi_cam/).
1.   Inside the Raspberry Pi, test the Camera Module v2 and install its related ROS packages with:

     ```shell
     cd ~/
     raspistill -v -o test.jpg
     ls
     rm test.jpg
     cd ~/catkin_ws/src
     git clone https://github.com/UbiquityRobotics/raspicam_node.git
     sudo apt-get install ros-kinetic-compressed-image-transport ros-kinetic-camera-info-manager
     cd ~/catkin_ws && catkin_make
     cd ~/
     source ~/.bashrc
     ```
