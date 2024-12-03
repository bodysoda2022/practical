import numpy as np
import matplotlib.pyplot as plt

# Function to calculate the end-effector position using forward kinematics
def forward_kinematics(thetal, theta2, l1, l2):
    # Calculate the position of the end-effector based on joint angles and link lengths
    x = l1 * np.cos(thetal) + l2 * np.cos(thetal + theta2)
    y = l1 * np.sin(thetal) + l2 * np.sin(thetal + theta2)
    return x, y

# Get input from the user for joint angles and link lengths
thetal_deg = float(input("Enter the angle of the first joint (in degrees): "))
theta2_deg = float(input("Enter the angle of the second joint (in degrees): "))
l1 = float(input("Enter the length of the first link: "))
l2 = float(input("Enter the length of the second link: "))

# Convert angles from degrees to radians
thetal = np.radians(thetal_deg)
theta2 = np.radians(theta2_deg)

# Calculate the end-effector position
x, y = forward_kinematics(thetal, theta2, l1, l2)

# Print the results
print(f"End-effector position: (x: {x:.2f}, y: {y:.2f})")

# Plot the robot arm
plt.figure(figsize=(6, 6))

# Calculate the coordinates of the joints
x1 = l1 * np.cos(thetal)
y1 = l1 * np.sin(thetal)
x2 = x  # End-effector position
y2 = y

# Plot the links
plt.plot([0, x1], [0, y1], 'r', linewidth=3)  # First link
plt.plot([x1, x2], [y1, y2], 'b', linewidth=3)  # Second link

# Plot the joints
plt.plot(0, 0, 'ko', markersize=10)  # Base joint
plt.plot(x1, y1, 'bo', markersize=10)  # First joint
plt.plot(x2, y2, 'ro', markersize=10)  # End-effector

# Set plot limits and labels
plt.xlim([-l1 - l2, l1 + l2])
plt.ylim([-l1 - l2, l1 + l2])
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Two-Link Robot Arm")

# Display grid
plt.grid(True)

# Show the plot
plt.show()
