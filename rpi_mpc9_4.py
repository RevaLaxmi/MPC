'''MPC: PITCH AND ROLL'''
'''in process: optimization of the code in rpi_mpc9_5'''

from casadi import SX, vertcat, Function, nlpsol
import numpy as np
import cv2
import time
import csv
from pymavlink import mavutil
from screeninfo import get_monitors


# MISSION PLANNER CONNECTION
master = mavutil.mavlink_connection('tcp:127.0.0.1:14550')
print("Waiting for heartbeat...")
master.wait_heartbeat()
print("Heartbeat received. Connected to the vehicle.")


# ADJUSTING CAMERA DISPLAY SCREEN DIMENSIONS
# Determine the screen dimensions
screen = get_monitors()[0]  # Adjust the index if using a secondary monitor
screen_width, screen_height = screen.width, screen.height
# Set a fixed display resolution for the camera feed
display_width = int(screen_width * 0.4)  # 40% of screen width
display_height = int(screen_height * 0.4)  # 40% of screen height


# CAMERA CONNECTION
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # For Windows
                                              # Adjust to 0 to use inbuilt camera. 
cv2.namedWindow("Camera Feed")
tracker = cv2.TrackerKCF_create()
bbox = None
tracking = False


# MOUSE CLICK FOR TRACKING -> USING KCF
def click_event(event, x, y, flags, param):
    global bbox, tracker, tracking
    if event == cv2.EVENT_LBUTTONDOWN:
        bbox = (x - 25, y - 25, 50, 50)
        tracker = cv2.TrackerKCF_create()
        tracker.init(frame, bbox)
        tracking = True
        print(f"Initialized tracker at position: {bbox}")
cv2.setMouseCallback("Camera Feed", click_event)


# MPC PARAMETERS -> these effect optimization
pwm_center = 1500                                                       # Neutral PWM value
pwm_limit = 1000                                                        # PWM limit, so deviation from 1500 is -+1000
'''CAN ADJUST: Shorter time steps allow for precide modelling BUT increased computation time.'''
dt = 0.02                                                               # Timestep for the MPC, in this case 20milliseconds
'''CAN ADJUST: horizon value. Increase prediction horizon to anticipate the future errors over a long time period. BUT it increases computation time.'''
N = 30                                                                  # The horizon we're keeping 
'''CAN ADJUST: Increasing the values gives us precise tracking but extremely aggresive changes. So we need to find a balance between the two.'''
Q_phi = 50                                                              # Weight for roll error in the cost function. Higher value means more aggressive changes to roll angle.
Q_theta = 100                                                           # Weight for pitch error in the cost function. Higher value means more aggressive changes to roll angle.


# PLACEHOLDERS FOR OPTIMIZATION PROCESS
phi = SX.sym('phi')                                                     # Roll angle
theta = SX.sym('theta')                                                 # Pitch angle
u_phi = SX.sym('u_phi')                                                 # Control input for roll
u_theta = SX.sym('u_theta')                                             # Control input for pitch
state = vertcat(phi, theta)                                             # Current state of the system
control = vertcat(u_phi, u_theta)                                       # These control inputs will be optimized to minimize the system's error


# HOW ROLL AND PITCH ANGLES CHANGE WITH TIME BASED ON THE CURRENT STATE AND CONTROL INPUTS
phi_dot = -0.1 * phi + u_phi
theta_dot = -0.1 * theta + u_theta
state_dot = vertcat(phi_dot, theta_dot)
'''Explanation of the 0.1 value: This is the damping factor for the system. So as we increase the value, the pitch and roll factors settle faster. We don't want the system to overshoot and oscillate -> that's why we've kept this value to begin with.'''


# Dynamics function
f = Function('f', [state, control], [state_dot])
phi_0 = SX.sym('phi_0')
theta_0 = SX.sym('theta_0')
init_state = vertcat(phi_0, theta_0) 


# MPC Variables
X = [init_state]                                                       # Initial state passed as a parameter
U = []                                                                 # Control inputs of the horizon being calculated 
cost = 0                                                               # Cost function to be minimized, declared here for our opti calculations
constraints = []


'''Nothing to be changed in this code block. Just initializing the MPC variables and creating the cost function and constraints.'''
for t in range(N):
    u_phi_t = SX.sym(f'u_phi_{t}')
    u_theta_t = SX.sym(f'u_theta_{t}')
    U.append(vertcat(u_phi_t, u_theta_t))
    
    # Predict next state
    X_next = X[-1] + dt * f(X[-1], vertcat(u_phi_t, u_theta_t))
    X.append(X_next)
    
    # Cost function: penalize roll and pitch errors
    cost += Q_phi * X_next[0]**2 + Q_theta * X_next[1]**2
    
    # Constraints for control inputs
    constraints += [-pwm_limit <= u_phi_t, u_phi_t <= pwm_limit]
    constraints += [-pwm_limit <= u_theta_t, u_theta_t <= pwm_limit]
U_flat = vertcat(*[u for u in U])

'''This code block is the same as the above, its just reducing all the calculation outputs we were getting in the terminal.'''
opti = nlpsol(
    'opti', 
    'ipopt', 
    {
        'x': U_flat,
        'f': cost,
        'g': vertcat(*constraints),
        'p': init_state
    }, 
    {
        'ipopt.print_level': 0,                                        # Suppress Ipopt log output
        'ipopt.sb': 'yes',                                             # Suppress barrier output
        'print_time': 0                                                # Suppress timing information
    }
)


# CAN WITHIN 1 FUNCTION CALL FOR PITCH/ROLL ERROR, INSTEAD OF 2 DIFFERENT FUNCTIONS
'''Nothing to be changed in this code block. Just calculating the roll error'''
def get_roll_error_from_object_position(frame):
    global bbox
    frame_height, frame_width, _ = frame.shape
    frame_center_x = frame_width // 2
    if tracking:
        success, bbox = tracker.update(frame)
        if success:
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            object_center_x = int(bbox[0] + bbox[2] // 2)
            return (object_center_x - frame_center_x) / frame_width
    return 0

def get_pitch_error_from_object_position(frame):
    global bbox
    frame_height, frame_width, _ = frame.shape
    frame_center_y = frame_height // 2
    if tracking:
        success, bbox = tracker.update(frame)
        if success:
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            object_center_y = int(bbox[1] + bbox[3] // 2)
            return (frame_center_y - object_center_y) / frame_height
    return 0


'''CSV FILE: for self reference. Can comment out the entire block if not needed.'''
csv_file = open('mpc_data_log.csv', mode='w', newline='')  
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Frame', 'Roll Error', 'Pitch Error', 'PWM Roll', 'PWM Pitch'])  

frame_count = 0

'''Nothing to be changed in this code block. It's the main loop that runs the MPC and sends commands to the drone.'''
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break

        # Resize the frame to fit the display dimensions
        frame = cv2.resize(frame, (display_width, display_height))

        frame_count += 1

        # Calculate errors
        roll_error = get_roll_error_from_object_position(frame)
        pitch_error = get_pitch_error_from_object_position(frame)

        # Solve MPC
        current_state = np.array([roll_error, pitch_error])  # Initialize state
        solution = opti(x0=np.zeros(U_flat.shape), lbg=-pwm_limit, ubg=pwm_limit, p=current_state)
        optimal_controls = np.array(solution['x']).flatten()

        # Extract control inputs
        optimal_pwm_roll = int(pwm_center + 10 * optimal_controls[0])
        optimal_pwm_pitch = int(pwm_center + 10 * optimal_controls[1])

        # Clamp values to PWM limits
        optimal_pwm_roll = max(min(optimal_pwm_roll, pwm_center + pwm_limit), pwm_center - pwm_limit)
        optimal_pwm_pitch = max(min(optimal_pwm_pitch, pwm_center + pwm_limit), pwm_center - pwm_limit)

        # Send commands to drone
        master.mav.rc_channels_override_send(
            master.target_system,
            master.target_component,
            optimal_pwm_roll, optimal_pwm_pitch, 0, 0, 0, 0, 0, 0
        )

        ''' printing in the terminal aswell'''
        print(f"Roll Error: {roll_error:.3f}, Roll PWM: {optimal_pwm_roll}")
        print(f"Pitch Error: {pitch_error:.3f}, Pitch PWM: {optimal_pwm_pitch}")
        '''CSV FILE: for self reference. Can comment out the entire block if not needed.'''
        csv_writer.writerow([frame_count, roll_error, pitch_error, optimal_pwm_roll, optimal_pwm_pitch])

        cv2.imshow("Camera Feed", frame)

        if cv2.waitKey(int(dt * 1000)) & 0xFF == 27:
            break

except KeyboardInterrupt:
    print("MPC control terminated.")
finally:
    cap.release()
    cv2.destroyAllWindows()


''' END '''