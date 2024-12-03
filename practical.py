import streamlit as st
import requests

# App title
st.title("AIR Practical")

# Dictionary of subtitles and their corresponding GitHub raw links
files = {
    "ros install and packages ": "https://raw.githubusercontent.com/bodysoda2022/practical/refs/heads/main/ros_install.sh",
    "publisher and subscriber ": "https://raw.githubusercontent.com/bodysoda2022/practical/refs/heads/main/air_publisher.sh",
    "Foward kinematics 2 DOF ": "https://raw.githubusercontent.com/bodysoda2022/practical/refs/heads/main/forward_kinematics_2dof.py",
    "Inverse kinematics 2 DOF ": "https://raw.githubusercontent.com/bodysoda2022/practical/refs/heads/main/inverse_kinematics_2dof.py",
    "Forward kinematics 3 DOF ": "https://raw.githubusercontent.com/bodysoda2022/practical/refs/heads/main/forward_kinematics_3dof.py",
    "Inverse Kinematics 3 DOF ": "https://raw.githubusercontent.com/bodysoda2022/practical/refs/heads/main/inverse_kinematics_2dof.py",
    "Four Bar Linkage ": "https://raw.githubusercontent.com/bodysoda2022/practical/refs/heads/main/four%20bar%20inkage.linkage2",
    "Pick and Place ": "https://raw.githubusercontent.com/bodysoda2022/practical/refs/heads/main/pick%20and%20place.linkage2",
    "Sliding crank ": "https://raw.githubusercontent.com/bodysoda2022/practical/refs/heads/main/sliding%20cranck.linkage2",
    "velocity and force ": "https://raw.githubusercontent.com/bodysoda2022/practical/refs/heads/main/velocity_force.py",
}


# Loop through each subtitle and create a direct download button
for title, url in files.items():
    st.subheader(title)
    try:
        # Fetch the file directly
        response = requests.get(url)
        response.raise_for_status()

        # Extract the file name from the URL
        file_name = url.split("/")[-1]

        # Display the download button directly
        st.download_button(
            label=f"Download {title}",
            data=response.content,
            file_name=file_name,
            mime="application/octet-stream"
        )
    except Exception as e:
        st.error(f"An error occurred while preparing {title}: {e}")
