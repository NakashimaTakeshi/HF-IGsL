# TurtleBot3 Software Development Environment (SDE)

This project aims to develop a shared platform for collabrative resesearch and development on the TurtleBot3 robot at the [Emergent Systems Laboratory](http://www.em.ci.ritsumei.ac.jp/) of Ritsumeikan University by providing a robust, scalable, virtualized, and documented environment to all contributors.

## Content

[[_TOC_]]

## Contribution Guidelines

Carefully read the contribution guidelines before pushing new code or requesting a merge.
Details can be found in the contribution guide: [CONTRIBUTING.md](CONTRIBUTING.md).

## Reporting Issues

Please report any issue or future work by using the GitLab issue tracker: [https://gitlab.com/emlab/TurtleBot3/issues](https://gitlab.com/emlab/TurtleBot3/issues).

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
*   A Nvidia GPU capable of running CUDA 10.0 (compute capability >= 3.0), or newer, to accelerate 3D graphics and deep learning computing.
*   A properly configured gitlab.com account linked with your personal SSH key to push contributions to the project repository: https://docs.gitlab.com/ee/ssh/.

### Step 1: Set up the Development Environment

Set up the environment of the development machine with the following instructions.

1.   Install [Git](https://git-scm.com/) if necessary:

     ```shell
     sudo apt-get update && sudo apt-get install -y git
     ```

2.   Clone the TurtleBot3 project repository in your home folder:

     ```shell
     cd ~/ && git clone https://gitlab.com/emlab/TurtleBot3.git
     ```

     Enter your GitLab developer credentials if prompted.
3.   Configure the system environment:

     ```shell
     cd ~/TurtleBot3/ && bash ./SETUP-DEVEL-MACHINE.bash
     ```

     The execution of `SETUP-DEVEL-MACHINE.bash` requires `sudo` permissions to install the tools that allow virtualization, i.e. Docker, [Docker Compose](https://github.com/docker/compose), and [Nvidia Docker 2](https://github.com/NVIDIA/nvidia-docker).
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
     cd ~/TurtleBot3/ && bash ./BUILD-DOCKER-IMAGE.bash
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
     It allows the container to share system resources, such as volumes and devices, with the host machine.
2.   Execute the Bash function `sde-get-fully-started` to configure the freshly built Docker container.
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

The development environment contains the several useful Bash functions, all starting with the prefix `sde-`, to make your work more convenient.
Including, but not limited to:

*   `sde-download-model-data`: Download from the cloud all the large binary files required at runtime (models, datasets, dictionaries, weights, etc.).
*   `sde-build-catkin-workspace`: Build and source the Catkin workspace on top of the system ROS environment.
*   `sde-reset-catkin-workspace`: Remove built artifacts, then cleanly rebuild and source the Catkin workspace on top of the system ROS environment (to use after, for example, switching branches).
*   `sde-fix-permission-issues`: Fix the various permission issues that may appear when manipulating, on the host machine, files generated by the `root` user of the Docker container.
*   `sde-get-fully-started`: Execute several of the aforementioned functions to quickly get started when entering a freshly built Docker container.

> **Note:**
  These Bash functions are based on helper scripts that can be found in `/root/TurtleBot3/docker/turtlebot3-devel/scripts/` in the Docker container or in `~/TurtleBot3/docker/turtlebot3-devel/scripts/` in the host machine.
  You can see their definitions in `~/.bashrc` inside the container.

> **Note:**
  The script `build-catkin-workspace.bash` will build the Catkin workspace using `catkin build` instead of the older but still officially default `catkin_make`.
  Please be sure to build using `catkin build` to avoid strange issues.

#### Multiple Terminal Operation

You can simultaneously run multiple terminals using `RUN-TERMINATOR-TERMINAL.bash`.
This script opens [Terminator](https://gnometerminator.blogspot.com/) with the default layout configuration stored in `~/TurtleBot3/terminator/config`.
Each sub-terminal automatically executes `RUN-DOCKER-CONTAINER.bash` with a predefined ROS launch file for convenience.
You can then select execution options by pressing specific keys as shown in the example below:

```
Run 'example_roslaunch_file.launch'? Press:
'r' to run with the robot,
's' to run in the simulator,
'c' to enter a child shell,
'q' to quit.
```

#### Configuration of the Docker Container

A cascade of scripts performs the initialization of the Docker container.
Although the boundaries between them can sometimes be blurry, each one has a specific function and future implementations/revisions should keep these functions separated as much as possible.
They can be found on the host machine at:

*   `~/TurtleBot3/docker/turtlebot3-devel/Dockerfile`: Used by `BUILD-DOCKER-IMAGE.bash` to create the image of the shared development environment.
    It mainly describes the installations of the project tools/dependencies required inside the container.
*   `~/TurtleBot3/docker/docker-compose.yml`: Describes the interface between the host machine and the container.
    It includes external information from outside the container, such as device host names or network configuration.
    It is invoked when the container is (re)started, most frequently the first time that `RUN-DOCKER-CONTAINER.bash` is run after the container is either stopped (host reboot) or rebuilt.
*   `~/TurtleBot3/docker/turtlebot3-devel/scripts/initialize-docker-container.bash`: A startup script referenced in `docker-compose.yml` and thus also invoked when (re)starting the container, most frequently the first time that `RUN-DOCKER-CONTAINER.bash` is run after the container is either stopped (host reboot) or rebuilt.
    It executes the internal commands needed at the startup of the container (for example, to maintain the container running in the background).
*   `~/TurtleBot3/docker/turtlebot3-devel/scripts/initialize-bash-shell.bash`: A shell script appended to `/root/.bashrc` at the end of the `Dockerfile` to set up the Bash shell environment inside the container (terminal colors, shell functions, symbolic links, binary paths, environment sourcing, etc.).
    Theoretically, its content could be written directly in the `Dockerfile` but this would result in complex lines of code as helpful Bash syntax like here-documents are not supported by Docker.
    It is automatically sourced by `/root/.bashrc` every time the user enters the Bash shell inside the container, most likely every time `RUN-DOCKER-CONTAINER.bash` is run.

### Step 5: Develop on the Simulator/Robot

From here, you can continue with either one, or both, of the following options depending on the presence of a TurtleBot3 robot within the same local network as the development machine.

1.   Enter the Docker container from a new terminal window:

     ```shell
     cd ~/TurtleBot3/ && bash ./RUN-DOCKER-CONTAINER.bash
     ```

2.   Connect to `roscore` that is running inside the Gazebo simulator with the following alias:

     ```shell
     simulation_mode
     ```

     Or connect to `roscore` that is running inside the TurtleBot3 robot with the following alias:

     ```shell
     robot_mode
     ```

Finally, before writing any new code, please make sure to have read the contribution guidelines in: [CONTRIBUTING.md](CONTRIBUTING.md).

> **Note:**
  The most important rule is to avoid pushing large binary files (datasets, weights, etc.) in the repository.
  Instead, you need to provide a link to download all your necessary large binary files from the cloud with `turtlebot3-download-model-data`.
  Ideally, all these files should be centralized in an online storage.

## Robot Configuration

The default configuration of the TurtleBot3 on-board computer is as follows:

*   User name: `pi`
*   Host name: `turtlebot3-01.local`
*   Password: `turtlebot`

To connect to the TurtleBot3 on-board computer, run the following in a new terminal on the host system:

```shell
ping turtlebot3-01.local
ssh pi@turtlebot3-01.local
```
