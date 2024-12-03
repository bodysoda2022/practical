import numpy as np
import matplotlib.pyplot as plt

# Constants
g = 9.81  # Gravity in m/s^2

# Input Parameters
m1 = float(input("Enter mass of Link 1 (kg): "))
m2 = float(input("Enter mass of Link 2 (kg): "))
m_load = float(input("Enter mass of Load (kg): "))
L1 = float(input("Enter length of Link 1 (m): "))
L2 = float(input("Enter length of Link 2 (m): "))

# Range for theta values for plotting (in degrees)
thetal_range = np.linspace(0, 180, 180)  # Joint 1 angles
theta2_range = np.linspace(0, 180, 180)  # Joint 2 angles

# Initialize lists to store results for plotting
taul_values = []
tau2_values = []
Fg1_values = []
Fg2_values = []
F_ext_values = []

# Function to calculate forces and torques
def calculate_forces_and_torques(thetal, theta2):
    """
    Calculate gravitational forces and torques for the robotic arm.

    Args:
        thetal: Angle of joint 1 (in degrees).
        theta2: Angle of joint 2 (in degrees).

    Returns:
        Gravitational forces and torques.
    """
    # Convert angles to radians
    thetal_rad = np.radians(thetal)
    theta2_rad = np.radians(theta2)

    # Gravitational forces for each link and load
    Fg1 = m1 * g
    Fg2 = m2 * g
    F_ext = m_load * g

    # Torque at Joint 2
    tau2 = Fg2 * (L2 / 2) * np.sin(theta2_rad) + F_ext * L2 * np.sin(theta2_rad)

    # Torque at Joint 1
    taul = (
        Fg1 * (L1 / 2) * np.sin(thetal_rad)
        + Fg2 * L1 * np.sin(thetal_rad)
        + Fg2 * (L2 / 2) * np.sin(thetal_rad + theta2_rad)
        + F_ext * (L1 + L2) * np.sin(thetal_rad + theta2_rad)
    )

    return Fg1, Fg2, F_ext, taul, tau2

# Calculate forces and torques for a range of angles
for thetal in thetal_range:
    for theta2 in theta2_range:
        Fg1, Fg2, F_ext, taul, tau2 = calculate_forces_and_torques(thetal, theta2)
        Fg1_values.append(Fg1)
        Fg2_values.append(Fg2)
        F_ext_values.append(F_ext)
        taul_values.append(taul)
        tau2_values.append(tau2)

# Convert lists to arrays for plotting
taul_values = np.array(taul_values)
tau2_values = np.array(tau2_values)
Fg1_values = np.array(Fg1_values)
Fg2_values = np.array(Fg2_values)
F_ext_values = np.array(F_ext_values)

# Plotting Force and Torque vs Joint Angles
plt.figure(figsize=(12, 10))

# Plot Torque at Joint 1 vs. Joint Angle
plt.subplot(2, 2, 1)
plt.plot(thetal_range, taul_values[:len(thetal_range)], label=r'$\tau_1$ (Torque at Joint 1)')
plt.xlabel(r'$\theta_1$ (degrees)')
plt.ylabel("Torque (Nm)")
plt.title("Torque at Joint 1 vs. Joint Angle")
plt.legend()

# Plot Gravitational Force on Link 1 vs. Joint Angle
plt.subplot(2, 2, 2)
plt.plot(thetal_range, Fg1_values[:len(thetal_range)], label=r'$F_{g1}$ (Force on Link 1)')
plt.xlabel(r'$\theta_1$ (degrees)')
plt.ylabel('Force (N)')
plt.title('Gravitational Force on Link 1 vs. Joint Angle')
plt.legend()

# Plot Torque at Joint 2 vs. Joint Angle
plt.subplot(2, 2, 3)
plt.plot(theta2_range, tau2_values[:len(theta2_range)], label=r'$\tau_2$ (Torque at Joint 2)')
plt.xlabel(r'$\theta_2$ (degrees)')
plt.ylabel("Torque (Nm)")
plt.title("Torque at Joint 2 vs. Joint Angle")
plt.legend()

# Plot Gravitational Forces on Link 2 and Load vs. Joint Angle
plt.subplot(2, 2, 4)
plt.plot(theta2_range, Fg2_values[:len(theta2_range)], label=r'$F_{g2}$ (Force on Link 2)')
plt.plot(theta2_range, F_ext_values[:len(theta2_range)], label=r'$F_{\text{ext}}$ (Force on Load)')
plt.xlabel(r'$\theta_2$ (degrees)')
plt.ylabel('Force (N)')
plt.title('Gravitational Forces on Link 2 and Load vs. Joint Angle')
plt.legend()

plt.tight_layout()
plt.show()
