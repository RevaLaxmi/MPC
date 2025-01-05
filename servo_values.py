from pymavlink import mavutil
import time
import matplotlib.pyplot as plt

# Connection string
connection_string = "tcp:127.0.0.1:14550"
master = mavutil.mavlink_connection(connection_string)
print(f"Connecting to SITL on {connection_string}...")

# Wait for heartbeat
print("Waiting for heartbeat...")
master.wait_heartbeat()
print(f"Heartbeat received from system {master.target_system}, component {master.target_component}")

# Continuously fetch servo output messages
while True:
    msg_control = master.recv_match(type="SERVO_OUTPUT_RAW", blocking=True, timeout=1)  # Wait with timeout
    if msg_control:
        # Print all servo output values to determine which corresponds to each surface
        print(f"Servo 5: {msg_control.servo5}")
        print(f"Servo 6: {msg_control.servo6}")
        print(f"Servo 7: {msg_control.servo7}")
        print(f"Servo 8: {msg_control.servo8}")
        print(f"Servo 9: {msg_control.servo9}")
        print(f"Servo 10: {msg_control.servo10}")
        print(f"Servo 11: {msg_control.servo11}")
        print(f"Servo 12: {msg_control.servo12}")
        print(f"Servo 13: {msg_control.servo13}")
        print(f"Servo 14: {msg_control.servo14}")
        print(f"Servo 15: {msg_control.servo15}")
        print(f"Servo 16: {msg_control.servo16}")

    time.sleep(0.5)  # Sleep for a short time to avoid excessive CPU usage
