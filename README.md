# MPC vs PID Controller for Optimized Drone Control

## Project Overview

This project evaluates and compares Model Predictive Control (MPC) and Proportional-Integral-Derivative (PID) control strategies using MAVLink communication and drone flight simulation. The primary goal is to optimize flight control by centering a drone’s camera gimbal on a target object. The approach highlights PID for basic system response and MPC for more dynamic, anticipatory control with error correction.

---

## Key Components

### Technologies and Tools

- **Python**: Programming language for controller implementation
- **CasADi**: Framework for numerical optimization
- **OpenCV**: Computer vision library for object tracking
- **MAVLink**: Protocol for drone communication
- **Raspberry Pi 4 (RPI4)**: Multi-core processing for real-time operations

### MAVLink Connection Setup

We use MAVLink for communication with the drone:

```python
master = mavutil.mavlink_connection('tcp:127.0.0.1:14550')
print("Waiting for heartbeat...")
master.wait_heartbeat()
print("Heartbeat received. Connected to the vehicle.")
```

This establishes a connection between the script and the simulated or physical drone.

### Camera and Display Configuration

To track objects in the frame:

```python
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cv2.namedWindow("Camera Feed")
```

- **Object Tracking**: Using `cv2.TrackerKCF_create()` for mouse-click-initiated tracking.
- **Display Scaling**: Resizing the video feed to 40% of the screen dimensions for optimal viewing.

### MPC Implementation

#### MPC Model Dynamics

The MPC code models the drone's roll and pitch dynamics as a simplified linear system:

```python
phi_dot = -0.1 * phi + u_phi
theta_dot = -0.1 * theta + u_theta
```

- **State Variables**: `phi` (roll) and `theta` (pitch) represent the current orientation.
- **Control Inputs**: `u_phi` and `u_theta` adjust roll and pitch to minimize errors.
- **Damping Factor**: The value `0.1` represents a damping coefficient that helps stabilize the system, preventing overshooting and oscillations.

This linear model is chosen for its simplicity and real-time feasibility on embedded systems like Raspberry Pi.

#### Cost Function and Constraints

The cost function penalizes deviations from desired roll and pitch angles:

```python
cost += Q_phi * X_next[0]**2 + Q_theta * X_next[1]**2
```

- **Weighting Factors**: `Q_phi` and `Q_theta` influence how aggressively the controller corrects errors. Higher values result in faster responses but may cause more abrupt movements.

Control inputs are constrained to ensure safe operation:

```python
constraints += [-pwm_limit <= u_phi_t, u_phi_t <= pwm_limit]
constraints += [-pwm_limit <= u_theta_t, u_theta_t <= pwm_limit]
```

These constraints limit the maximum control actions, preventing damage or instability.

### Error Calculation

Errors in object tracking are calculated relative to the frame’s center:

```python
def get_roll_error_from_object_position(frame):
    frame_center_x = frame_width // 2
    object_center_x = int(bbox[0] + bbox[2] // 2)
    return (object_center_x - frame_center_x) / frame_width
```

This function returns a normalized roll error. A similar function computes the pitch error.

### Optimization

The `nlpsol` solver from CasADi finds the optimal control inputs:

```python
opti = nlpsol('opti', 'ipopt', {
    'x': U_flat, 'f': cost, 'g': vertcat(*constraints), 'p': init_state
}, {
    'ipopt.print_level': 0, 'ipopt.sb': 'yes', 'print_time': 0
})
```

This solver efficiently handles the nonlinear constraints and cost function.

### Multi-threading and Multi-processing

Since this project runs on Raspberry Pi 4 (RPI4) with multi-core architecture, **multi-threading** and **multi-processing** are utilized:

- **Parallel Video Processing**: Captures and processes frames concurrently.
- **MPC Computation in Parallel with MAVLink Communication**: Ensures timely control updates without interrupting data streams.

These techniques leverage the RPI4’s multi-core CPU to achieve real-time performance.

### Data Logging

MPC data (roll/pitch errors and PWM values) are logged to a CSV file for analysis:

```python
csv_writer.writerow([frame_count, roll_error, pitch_error, optimal_pwm_roll, optimal_pwm_pitch])
```

The logged data helps evaluate the controller’s performance.

---

## Comparison of MPC and PID

- **PID**: Simple implementation with direct error correction, but limited in anticipating future states.
- **MPC**: Uses predictive models and constraints for dynamic adjustments, yielding smoother and more precise control.

