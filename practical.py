import streamlit as st
import requests

# App title
st.title("ML Practical")

# Dictionary of subtitles and their corresponding GitHub raw links
files = {
    "Basic Operations - experiment 1": "https://raw.githubusercontent.com/bodysoda2022/pract/refs/heads/main/basic%20operations.py",
    "Handle missing data - experiment 2": "https://raw.githubusercontent.com/bodysoda2022/pract/refs/heads/main/exp2/exp2.py",
    "Feature engineering,model selection,cross-validation - experiment 3": "https://raw.githubusercontent.com/bodysoda2022/pract/refs/heads/main/exp3.py",
    "Linear Regression - experiment 4 ": "https://raw.githubusercontent.com/bodysoda2022/pract/refs/heads/main/exp4.py",
    "Classification problems and decision tree algorithms - experiment 5": "https://raw.githubusercontent.com/Pradeesh890/pract/refs/heads/main/clustering.py",
    " Convolutional Neural Network (CNN) - experiment 6(download the training and testing data from kaggle)": "https://raw.githubusercontent.com/Pradeesh890/pract/refs/heads/main/decision_trees.py",
    "sentimen analysis - experiment 7 ": "https://raw.githubusercontent.com/Pradeesh890/pract/refs/heads/main/random_forest.py",
    "Model deployment in cloud only model code !!- experiment 8": "https://raw.githubusercontent.com/Pradeesh890/pract/refs/heads/main/svm.py",
    "fake news detection - experiment 9": "https://raw.githubusercontent.com/Pradeesh890/pract/refs/heads/main/neural_networks.py",
    
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
