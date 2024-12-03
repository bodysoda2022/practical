import numpy as np
import matplotlib.pyplot as plt

def inverse_kinematics_3dof(x, y, z, l1, l2, l3):
    """
    Calculates the joint angles for a 3-DOF robot arm to reach a given target point.

    Args:
        x: The x-coordinate of the target point.
        y: The y-coordinate of the target point.
        z: The z-coordinate of the target point (end-effector orientation).
        l1: Length of the first link.
        l2: Length of the second link.
        l3: Length of the third link (end-effector).

    Returns:
        A tuple containing the joint angles (theta1, theta2, theta3).
    """
    # Distance in the XY plane
    d_xy = np.sqrt(x**2 + y**2)

    # Total distance from the base to the target
    d_total = np.sqrt(d_xy**2 + z**2)

    # Check if the target is within reach
    if d_total > (l1 + l2 + l3) or d_total < abs(l1 - l2 - l3):
        raise ValueError("Target is out of reach for the robot arm.")

    # Base joint angle
    theta1 = np.arctan2(y, x)

    # Projected distance in the XY plane for link calculations
    d_proj = np.sqrt(d_xy**2 + z**2)

    # Angle calculations using cosine law
    alpha = np.arctan2(z, d_xy)  # Elevation angle to the target
    beta = np.arccos((l1**2 + d_proj**2 - l2**2) / (2 * l1 * d_proj))
    theta2 = alpha + beta  # Shoulder joint angle

    gamma = np.arccos((l1**2 + l2**2 - d_proj**2) / (2 * l1 * l2))
    theta3 = np.pi - gamma  # Elbow joint angle

    return theta1, theta2, theta3

# Input target position and link lengths
x = float(input("Enter the x-coordinate of the target point: "))
y = float(input("Enter the y-coordinate of the target point: "))
z = float(input("Enter the z-coordinate (orientation) of the target point: "))

l1 = 1.0  # Length of the first link
l2 = 1.0  # Length of the second link
l3 = 0.5  # Length of the third link (end-effector)

try:
    # Calculate joint angles
    theta1, theta2, theta3 = inverse_kinematics_3dof(x, y, z, l1, l2, l3)

    # Convert to degrees for easier interpretation
    theta1_deg = np.degrees(theta1)
    theta2_deg = np.degrees(theta2)
    theta3_deg = np.degrees(theta3)

    # Print the results
    print(f"Joint angle θ1 (base): {theta1_deg:.2f} degrees")
    print(f"Joint angle θ2 (shoulder): {theta2_deg:.2f} degrees")
    print(f"Joint angle θ3 (elbow): {theta3_deg:.2f} degrees")

    # Calculate the positions of each joint
    x1 = l1 * np.cos(theta1) * np.cos(theta2)
    y1 = l1 * np.sin(theta1) * np.cos(theta2)
    z1 = l1 * np.sin(theta2)

    x2 = x1 + l2 * np.cos(theta1) * np.cos(theta2 + theta3)
    y2 = y1 + l2 * np.sin(theta1) * np.cos(theta2 + theta3)
    z2 = z1 + l2 * np.sin(theta2 + theta3)

    x3 = x2 + l3 * np.cos(theta1) * np.cos(theta2 + theta3)
    y3 = y2 + l3 * np.sin(theta1) * np.cos(theta2 + theta3)
    z3 = z2 + l3 * np.sin(theta2 + theta3)

    # Plot the robot arm
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot([0, x1, x2, x3], [0, y1, y2, y3], [0, z1, z2, z3], '-o', linewidth=3, label="Robot Arm")
    ax.scatter(x, y, z, color='r', label="Target Point")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    ax.set_title("3-DOF Robot Arm Inverse Kinematics")
    ax.legend()
    plt.show()

except ValueError as e:
    print(e)
