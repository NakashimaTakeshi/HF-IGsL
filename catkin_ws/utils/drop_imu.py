import rosbag

outbag = rosbag.Bag('dataset12_2023-04-05-16-50-38__filtered.bag', 'w')
input_bag = '/root/TurtleBot3/dataset/experiment_target/dataset12_2023-04-05-16-50-38_filtered.bag'

try:
    for topic, msg, t in rosbag.Bag(input_bag).read_messages():
        # If the topic is what we are interested in, filter based on nsecs
        if topic == "/imu":
            # Get the first three digits of nsecs
            first_three_digits = int(str(msg.header.stamp.nsecs)[:3])

            # If the first three digits is not divisible by 5, skip this message
            if first_three_digits % 5 != 0:
                continue

        # If not, just write it back into the bag.
        outbag.write(topic, msg, t)
finally:
    outbag.close()