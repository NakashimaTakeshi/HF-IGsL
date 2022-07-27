#!/bin/bash

################################################################################

# Download the missing Fuel models in the 'aws_robomaker_hospital_world' ROS package.
# /root/TurtleBot3/catkin_ws/src/aws_robomaker_hospital_world/setup.sh

python3 -m pip install -r /root/TurtleBot3/catkin_ws/src/aws_robomaker_hospital_world/requirements.txt
python3 /root/TurtleBot3/catkin_ws/src/aws_robomaker_hospital_world/fuel_utility.py download \
-m XRayMachine -m IVStand -m BloodPressureMonitor -m BPCart -m BMWCart \
-m CGMClassic -m StorageRack -m Chair \
-m InstrumentCart1 -m Scrubs -m PatientWheelChair \
-m WhiteChipChair -m TrolleyBed -m SurgicalTrolley \
-m PotatoChipChair -m VisitorKidSit -m FemaleVisitorSit \
-m AdjTable -m MopCart3 -m MaleVisitorSit -m Drawer \
-m OfficeChairBlack -m ElderLadyPatient -m ElderMalePatient \
-m InstrumentCart2 -m MetalCabinet -m BedTable -m BedsideTable \
-m AnesthesiaMachine -m TrolleyBedPatient -m Shower \
-m SurgicalTrolleyMed -m StorageRackCovered -m KitchenSink \
-m Toilet -m VendingMachine -m ParkingTrolleyMin -m PatientFSit \
-m MaleVisitorOnPhone -m FemaleVisitor -m MalePatientBed \
-m StorageRackCoverOpen -m ParkingTrolleyMax \
-d /root/TurtleBot3/catkin_ws/src/aws_robomaker_hospital_world/fuel_models --verbose
