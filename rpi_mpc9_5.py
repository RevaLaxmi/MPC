# OPTIMIZED CODE -> using threading here
'''
Optimization done on the basis of the following factors in the RPi.
1. CPU: Quad Core -> Using Multi-threading for faster processing.
2. RAM: Limited RAM for processing -> Using shared variables for faster processing.
'''

'''
TO DO:

Use the multiprocessing library for true parallelism, especially for CPU-heavy tasks like MPC computation.
Split the compute-intensive tasks (like compute_mpc) into separate processes if they significantly burden the CPU.

Set the camera capture resolution to the desired size using cap.set(cv2.CAP_PROP_FRAME_WIDTH, display_width) and cap.set(cv2.CAP_PROP_FRAME_HEIGHT, display_height) to avoid resizing frames in software.

Precompute and Reuse: Precompute parts of the MPC problem (e.g., symbolic graph) if possible.

CSV FILE: Buffer the data in memory and write it to the file periodically (e.g., every 50 frames).
But will most likely just be removing the entire CSV file part so ignore ^ 

Issue: Sending commands at a fixed interval might cause conflicts with other threads.
Use a priority queue or shared buffer to decouple command generation from command sending.
Monitor the time drift of send_commands using time.perf_counter() to ensure precise timing.

Memory Usage: Issue: Continuous copying of frames (local_frame = frame.copy()) increases memory usage.
Work directly on the frame with proper locks.


Can do towards the end: Profile the code using tools like cProfile or line_profiler to identify bottlenecks.
'''

from casadi import SX, vertcat, Function, nlpsol
import numpy as np
import cv2
import time
import csv
from pymavlink import mavutil
from screeninfo import get_monitors
from threading import Thread, Lock, Event

# MISSION PLANNER CONNECTION
master = mavutil.mavlink_connection('tcp:127.0.0.1:14550')
print("Waiting for heartbeat...")
master.wait_heartbeat()
print("Heartbeat received. Connected to the vehicle.")

# ADJUSTING CAMERA DISPLAY SCREEN DIMENSIONS
screen = get_monitors()[0]
screen_width, screen_height = screen.width, screen.height
display_width = int(screen_width * 0.4)
display_height = int(screen_height * 0.4)

# CAMERA CONNECTION
cap = cv2.VideoCapture(0)
cv2.namedWindow("Camera Feed")
tracker = None
bbox = None
tracking = False

# THREADING VARIABLES
lock = Lock()
stop_event = Event()
frame = None  # Shared frame variable
roll_error = 0
pitch_error = 0
optimal_pwm_roll = 1500
optimal_pwm_pitch = 1500

# MOUSE CLICK FOR TRACKING -> USING KCF
def click_event(event, x, y, flags, param):
    global frame, bbox, tracker, tracking
    if event == cv2.EVENT_LBUTTONDOWN:
        with lock:
            if frame is not None and frame.size != 0:
                bbox_width, bbox_height = 50,50 
                x_min = max(0, x-bbox_width//2)
                y_min = max(0, y-bbox_width//2)
                bbox = (x_min, y_min, bbox_width, bbox_height)
                print(f"Click detected. Initializing tracker with bbox: {bbox}")
                
                tracker = cv2.TrackerKCF_create()
                try:
                    tracker.init(frame, bbox)
                    tracking = True
                    print("Tracker initialized successfully.")
                except cv2.error as e:
                    print(f"Error initializing tracker: {e}")
            else:
                print("Error: Frame is None or invalid during tracker initialization.")
cv2.setMouseCallback("Camera Feed", click_event)

# MPC PARAMETERS
pwm_center = 1500
pwm_limit = 1000
dt = 0.02
N = 30
Q_phi = 50
Q_theta = 100

# PLACEHOLDERS FOR OPTIMIZATION PROCESS
phi = SX.sym('phi')
theta = SX.sym('theta')
u_phi = SX.sym('u_phi')
u_theta = SX.sym('u_theta')
state = vertcat(phi, theta)
control = vertcat(u_phi, u_theta)
phi_dot = -0.1 * phi + u_phi
theta_dot = -0.1 * theta + u_theta
state_dot = vertcat(phi_dot, theta_dot)

# Dynamics function
f = Function('f', [state, control], [state_dot])
phi_0 = SX.sym('phi_0')
theta_0 = SX.sym('theta_0')
init_state = vertcat(phi_0, theta_0)

# MPC Variables
X = [init_state]
U = []
cost = 0
constraints = []
for t in range(N):
    u_phi_t = SX.sym(f'u_phi_{t}')
    u_theta_t = SX.sym(f'u_theta_{t}')
    U.append(vertcat(u_phi_t, u_theta_t))
    X_next = X[-1] + dt * f(X[-1], vertcat(u_phi_t, u_theta_t))
    X.append(X_next)
    cost += Q_phi * X_next[0]**2 + Q_theta * X_next[1]**2
    constraints += [-pwm_limit <= u_phi_t, u_phi_t <= pwm_limit]
    constraints += [-pwm_limit <= u_theta_t, u_theta_t <= pwm_limit]
U_flat = vertcat(*[u for u in U])

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
        'ipopt.print_level': 0,
        'ipopt.sb': 'yes',
        'print_time': 0
    }
)

def capture_frames():
    global frame
    while not stop_event.is_set():
        ret, raw_frame = cap.read()
        if ret:
            resized_frame = cv2.resize(raw_frame, (display_width, display_height))
            with lock:
                frame = resized_frame

                # Draw the bounding box for reference
                if bbox is not None:
                    x, y, w, h = [int(v) for v in bbox]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue bounding box
                    cv2.putText(frame, "Target", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


def compute_mpc():
    global roll_error, pitch_error, optimal_pwm_roll, optimal_pwm_pitch
    frame_count = 0
    csv_file = open('mpc_data_log.csv', mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Frame', 'Roll Error', 'Pitch Error', 'PWM Roll', 'PWM Pitch'])

    try:
        while not stop_event.is_set():
            with lock:
                if frame is None:
                    continue
                local_frame = frame.copy()

            frame_count += 1
            roll_error, pitch_error = compute_object_errors(local_frame)


            current_state = np.array([roll_error, pitch_error])
            solution = opti(x0=np.zeros(U_flat.shape), lbg=-pwm_limit, ubg=pwm_limit, p=current_state)
            optimal_controls = np.array(solution['x']).flatten()

            optimal_pwm_roll = int(pwm_center + 10 * optimal_controls[0])
            optimal_pwm_pitch = int(pwm_center + 10 * optimal_controls[1])
            optimal_pwm_roll = max(min(optimal_pwm_roll, pwm_center + pwm_limit), pwm_center - pwm_limit)
            optimal_pwm_pitch = max(min(optimal_pwm_pitch, pwm_center + pwm_limit), pwm_center - pwm_limit)

            csv_writer.writerow([frame_count, roll_error, pitch_error, optimal_pwm_roll, optimal_pwm_pitch])

    finally:
        csv_file.close()


def send_commands():
    while not stop_event.is_set():
        master.mav.rc_channels_override_send(
            master.target_system,
            master.target_component,
            optimal_pwm_roll, optimal_pwm_pitch, 0, 0, 0, 0, 0, 0
        )
        time.sleep(dt)


def compute_object_errors(local_frame):
    global bbox
    frame_height, frame_width, _ = local_frame.shape
    frame_center_x, frame_center_y = frame_width // 2, frame_height // 2

    if tracking:
        success, bbox = tracker.update(local_frame)
        if success:
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(local_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            object_center_x = int(bbox[0] + bbox[2] // 2)
            object_center_y = int(bbox[1] + bbox[3] // 2)

            roll_error = (object_center_x - frame_center_x) / frame_width
            pitch_error = (frame_center_y - object_center_y) / frame_height
            return roll_error, pitch_error
    return 0, 0


# Start threads
capture_thread = Thread(target=capture_frames, daemon=True) # share frame
mpc_thread = Thread(target=compute_mpc, daemon=True)        # share frame, roll/pitch/optimal_pwm_roll/optimal_pwm_pitch
command_thread = Thread(target=send_commands, daemon=True)  # share roll/pitch/optimal_pwm_roll/optimal_pwm_pitch
'''sends PWM values at fixed intervals (dt) regardless of other operations. This ensures the drone receives consistent control inputs without delays caused by frame processing or MPC computation.'''


capture_thread.start()
mpc_thread.start()
command_thread.start()

try:
    while True:
        with lock:
            if frame is not None:
                cv2.imshow("Camera Feed", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
except KeyboardInterrupt:
    print("MPC control terminated.")
finally:
    stop_event.set()
    capture_thread.join()
    mpc_thread.join()
    command_thread.join()
    cap.release()
    cv2.destroyAllWindows()

''' END '''
