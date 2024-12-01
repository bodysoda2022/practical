import streamlit as st
import requests

# App title
st.title("ML Practical")

# Dictionary of subtitles and their corresponding GitHub raw links
files = {
    "Basic Operations - experiment 1": "https://raw.githubusercontent.com/bodysoda2022/practical/refs/heads/main/basic%20operations.py",
    "Handle missing data - experiment 2": "https://raw.githubusercontent.com/bodysoda2022/practical/refs/heads/main/exp2/exp2.py",
    "Feature engineering, model selection, cross-validation - experiment 3": "https://raw.githubusercontent.com/bodysoda2022/practical/refs/heads/main/exp3.py",
    "Linear Regression - experiment 4": "https://raw.githubusercontent.com/bodysoda2022/practical/refs/heads/main/exp4.py",
    "Classification problems and decision tree algorithms - experiment 5": "https://raw.githubusercontent.com/bodysoda2022/practical/refs/heads/main/exp5.py",
    "Convolutional Neural Network (CNN) - experiment 6 (download the training and testing data from kaggle)": "https://raw.githubusercontent.com/bodysoda2022/practical/refs/heads/main/exp%206.py",
    "Sentiment analysis - experiment 7": "https://raw.githubusercontent.com/bodysoda2022/practical/refs/heads/main/exp7.py",
    "Fake news detection - experiment 9": "https://raw.githubusercontent.com/bodysoda2022/practical/refs/heads/main/exp9/exp9.py",
}

# Add the zip download link for experiment 8
zip_url = "https://github.com/bodysoda2022/practical/blob/main/ML%20Model%20Deply.zip?raw=true"

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

# Experiment 8: Download the ZIP file
st.subheader("Model deployment in cloud - experiment 8")
try:
    # Fetch the zip file directly
    response = requests.get(zip_url)
    response.raise_for_status()

    # Set the file name for the ZIP file
    zip_file_name = "ML_Model_Deploy.zip"

    # Display the download button for the zip file
    st.download_button(
        label="Download Model Deployment ZIP",
        data=response.content,
        file_name=zip_file_name,
        mime="application/zip"
    )
except Exception as e:
    st.error(f"An error occurred while preparing the ZIP download: {e}")
