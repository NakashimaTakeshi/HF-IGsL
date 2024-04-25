# #!/usr/bin/env python

import rosbag
import pandas as pd
from nav_msgs.msg import Odometry

# rosbag file
bag = rosbag.Bag('/root/TurtleBot3/dataset/experiment_target/dataset12_2023-04-05-16-50-38_filtered.bag')

# Output CSV file
csv_file = 'output.csv'

# Topic of interest
topic = "/tracker"

# Container for the extracted data
data = []

for topic, msg, t in bag.read_messages(topics=[topic]):
    # if isinstance(msg, Odometry):
    if msg._type == 'nav_msgs/Odometry':
        data.append({
            'time': t.to_sec(),
            'position_x': msg.pose.pose.position.x,
            'position_y': msg.pose.pose.position.y,
            'position_z': msg.pose.pose.position.z,
            'orientation_x': msg.pose.pose.orientation.x,
            'orientation_y': msg.pose.pose.orientation.y,
            'orientation_z': msg.pose.pose.orientation.z,
            'orientation_w': msg.pose.pose.orientation.w,
        })

bag.close()

# Convert to a pandas DataFrame
df = pd.DataFrame(data)
df = df.rename(columns={'time': '#time'})

# Write to CSV file
df.to_csv(csv_file, index=False)