# 'rulo_chatter' Package

The `rulo_chatter` package allows the quick test of a ROS setup by providing two simple nodes that communicate to each other.

*   Maintainer: Lotfi El Hafi ([lotfi.elhafi@gmail.com](mailto:lotfi.elhafi@gmail.com)).
*   Author: Lotfi El Hafi ([lotfi.elhafi@gmail.com](mailto:lotfi.elhafi@gmail.com)).

**Content:**

*   [Purpose](#purpose)
*   [Launch](#launch)

## Purpose

The `rulo_chatter` package goals are:

*    Provide a quick test of the ROS setup with two simple nodes that communicate to each other.
*    Serve as a reference/template when creating `CMakeLists.txt` and `package.xml` files.

The nodes were inspired by the official ROS tutorial *"[Writing a Simple Publisher and Subscriber (C++)](http://wiki.ros.org/ROS/Tutorials/WritingPublisherSubscriber%28c%2B%2B%29)"*.

## Launch

*   `rulo_chatter_default.launch`: Runs the nodes `rulo_talker.cpp` and `rulo_listener.cpp`, and output the results in the terminal.
*   `rulo_chatter_rosbridge.launch`: Extends `rulo_chatter_default.launch` by opening a WebSocket to visualize the results with `rulo_chatter.html` through [Rosbridge](http://wiki.ros.org/rosbridge_suite).
*   `rulo_chatter_rqt.launch`: Extends `rulo_chatter_default.launch` by displaying the results in a [Rqt](http://wiki.ros.org/rqt) GUI.
