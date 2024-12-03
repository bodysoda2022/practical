import numpy as np
import matplotlib.pyplot as plt

def inverse_kinematics(x, y, theta_total, l1, l2):
    """
    Calculates the joint angles for a 2-DOF arm to reach a given target point.

    Args:
        x: The x-coordinate of the target point.
        y: The y-coordinate of the target point.
        theta_total: The total angle of the end-effector with respect to the base.
        l1: Length of the first link.
        l2: Length of the second link.

    Returns:
        A tuple containing the joint angles (theta1, theta2) in radians.
    """
    d = np.sqrt(x**2 + y**2)  # Distance to the target

    # Check if the target is reachable
    if d > l1 + l2:
        raise ValueError("No solution: the given target is unreachable.")
    if d < abs(l1 - l2):
        raise ValueError("No solution: the given target is inside the dead zone.")

    # Calculate angles
    alpha = np.arctan2(y, x)  # Angle between x-axis and target point
    beta = np.arccos((l1**2 + d**2 - l2**2) / (2 * l1 * d))  # Angle at the first joint
    theta1 = alpha + beta

    gamma = np.arccos((l1**2 + l2**2 - d**2) / (2 * l1 * l2))  # Angle at the second joint
    theta2 = np.pi - gamma

    return theta1, theta2

# Get input from the user
x = float(input("Enter the x-coordinate of the target point: "))
y = float(input("Enter the y-coordinate of the target point: "))
theta_total_deg = float(input("Enter the total angle of the end-effector with respect to the base (in degrees): "))
l1 = float(input("Enter the length of the first link: "))
l2 = float(input("Enter the length of the second link: "))

# Convert angles to radians
theta_total = np.radians(theta_total_deg)

try:
    # Calculate joint angles
    theta1, theta2 = inverse_kinematics(x, y, theta_total, l1, l2)

    # Convert angles to degrees for easier interpretation
    theta1_deg = np.degrees(theta1)
    theta2_deg = np.degrees(theta2)

    print(f"Joint angle θ1: {theta1_deg:.2f} degrees")
    print(f"Joint angle θ2: {theta2_deg:.2f} degrees")

    # Calculate joint positions
    x1 = l1 * np.cos(theta1)
    y1 = l1 * np.sin(theta1)
    x2 = x1 + l2 * np.cos(theta1 + theta2)
    y2 = y1 + l2 * np.sin(theta1 + theta2)

    print(f"Position of joint 2: ({x1:.2f}, {y1:.2f})")
    print(f"Position of end-effector: ({x2:.2f}, {y2:.2f})")

    # Plot the robot arm
    plt.figure(figsize=(6, 6))
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.minorticks_on()
    plt.plot([0, x1], [0, y1], 'r', linewidth=3, label='Link 1')  # First link
    plt.plot([x1, x2], [y1, y2], 'b', linewidth=3, label='Link 2')  # Second link
    plt.scatter([0, x1, x2], [0, y1, y2], c=['black', 'blue', 'green'], zorder=5)  # Joints
    plt.scatter(x, y, c='red', marker='x', label='Target Point', zorder=5)  # Target point
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Inverse Kinematics of a 2-DOF Robot Arm')
    plt.xlim([-l1 - l2 - 1, l1 + l2 + 1])
    plt.ylim([-l1 - l2 - 1, l1 + l2 + 1])
    plt.legend()
    plt.show()

except ValueError as e:
    print(e)
