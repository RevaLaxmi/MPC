import numpy as np
from casadi import SX, vertcat, Function, nlpsol
from pymavlink import mavutil
import time


master = mavutil.mavlink_connection('tcp:127.0.0.1:14550')  # Replace with the actual connection string

while True:
    master.mav.rc_channels_override_send(
                master.target_system,
                master.target_component,
                1900,                   # Throttle (leave unchanged)
                0,  # Adjusted Roll PWM
                0, # Adjusted Pitch PWM
                0,                   # Yaw (leave unchanged)
                0,                   # Channel 5 (auxiliary)
                0,                   # Channel 6 (auxiliary)
                0,                   # Channel 7 (auxiliary)
                0                    # Channel 8 (auxiliary)
            )