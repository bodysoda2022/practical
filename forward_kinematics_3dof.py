import numpy as np
import matplotlib.pyplot as plt

def forward_kinematics_3dof(theta1, theta2, theta3, l1, l2, l3):
    """
    Calculates the end-effector position for a 3-DOF robot arm.

    Args:
        theta1: Angle of the first joint in radians.
        theta2: Angle of the second joint in radians.
        theta3: Angle of the third joint in radians.
        l1: Length of the first link.
        l2: Length of the second link.
        l3: Length of the third link.

    Returns:
        A tuple containing the x and y coordinates of the end-effector.
    """
    # Position of the first joint (base)
    x1 = l1 * np.cos(theta1)
    y1 = l1 * np.sin(theta1)

    # Position of the second joint (shoulder)
    x2 = x1 + l2 * np.cos(theta1 + theta2)
    y2 = y1 + l2 * np.sin(theta1 + theta2)

    # Position of the end-effector (wrist)
    x3 = x2 + l3 * np.cos(theta1 + theta2 + theta3)
    y3 = y2 + l3 * np.sin(theta1 + theta2 + theta3)

    return x3, y3

# Get input from the user
theta1 = float(input("Enter the angle of the first joint (in degrees): "))
theta2 = float(input("Enter the angle of the second joint (in degrees): "))
theta3 = float(input("Enter the angle of the third joint (in degrees): "))
l1 = float(input("Enter the length of the first link: "))
l2 = float(input("Enter the length of the second link: "))
l3 = float(input("Enter the length of the third link: "))

# Convert angles to radians
theta1 = np.radians(theta1)
theta2 = np.radians(theta2)
theta3 = np.radians(theta3)

# Calculate the end-effector position using forward kinematics
x, y = forward_kinematics_3dof(theta1, theta2, theta3, l1, l2, l3)

# Print the result
print("End-effector position: ({:.2f}, {:.2f})".format(x, y))

# Plot the robot arm
plt.figure(figsize=(8, 6))
plt.title("3-DOF Robot Arm")
plt.xlabel("X")
plt.ylabel("Y")

# Calculate the coordinates of the joints
x1 = l1 * np.cos(theta1)
y1 = l1 * np.sin(theta1)
x2 = x1 + l2 * np.cos(theta1 + theta2)
y2 = y1 + l2 * np.sin(theta1 + theta2)

# Plot the robot arm
plt.plot([0, x1], [0, y1], 'b-', linewidth=3, label="Link 1")
plt.plot([x1, x2], [y1, y2], 'g-', linewidth=3, label="Link 2")
plt.plot([x2, x], [y2, y], 'r-', linewidth=3, label="Link 3")

# Mark the joints
plt.plot(0, 0, 'ko', markersize=10)  # Base
plt.plot(x1, y1, 'ko', markersize=10)  # Shoulder
plt.plot(x2, y2, 'go', markersize=10)  # Elbow
plt.plot(x, y, 'ro', markersize=10)  # End-effector (Target)

# Set axis limits
plt.xlim([-l1-l2-l3, l1+l2+l3])
plt.ylim([-l1-l2-l3, l1+l2+l3])

# Display grid and legend
plt.grid(True)
plt.legend()


# Show the plot
plt.show()
