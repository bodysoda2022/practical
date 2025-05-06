import streamlit as st
import requests

# Set page title and configuration
st.set_page_config(page_title="practicals", layout="wide")

# Add custom CSS for styling
st.markdown("""
<style>
div.stDownloadButton > button {
    width: 100%;
    background-color: #4CAF50;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# Title for the application
st.title("NLP Praxtical")

# GitHub file links and titles
# Replace these with your actual GitHub raw links and titles
github_files = [
    {"title": "Text Classification using Naïve Bayes and SVM for Sentiment Analysis", "url": "https://raw.githubusercontent.com/bodysoda2022/practical/refs/heads/main/exp1.ipynb"},
    {"title": "vBuilding a Text Categorization System using TF-IDF and KNN", "url": "https://raw.githubusercontent.com/bodysoda2022/practical/refs/heads/main/exp2.ipynb"},
    {"title": "Speech-to-Text Conversion using RNNbased Models.", "url": "https://raw.githubusercontent.com/bodysoda2022/practical/refs/heads/main/exp3.ipynb"},
    {"title": "Sequence Labeling for Named Entity Recognition (NER) using LSTM.", "url": "https://raw.githubusercontent.com/bodysoda2022/practical/refs/heads/main/exp4.ipynb"},
    {"title": "Document Similarity using TF-IDF and Cosine Similarity", "url": "https://raw.githubusercontent.com/bodysoda2022/practical/refs/heads/main/exp5.ipynb"},
    {"title": "Building a Text Summarizer using BERT embedding and Extractive Summarization", "url": "https://raw.githubusercontent.com/bodysoda2022/practical/refs/heads/main/exp6.ipynb"},
    {"title": "Implementing Question Answering System using BERT-based Models", "url": "https://raw.githubusercontent.com/bodysoda2022/practical/refs/heads/main/exp7.ipynb"},
    {"title": "Text Generation for Storytelling using GPT-3 or GPT-Neo", "url": "https://raw.githubusercontent.com/bodysoda2022/practical/refs/heads/main/exp8.ipynb"},
    {"title": "Optimizing a Model for Edge Devices using Symmetric Quantization ", "url": "https://raw.githubusercontent.com/bodysoda2022/practical/refs/heads/main/exp9.ipynb"},
    {"title": "Building an AI-based Customer Support Agent using Multi-Agent Systems", "url": "https://raw.githubusercontent.com/bodysoda2022/practical/refs/heads/main/exp10.ipynb"}
]

# Function to download file content from GitHub
@st.cache_data  # Cache the function to improve performance
def get_file_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.text
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching file: {e}")
        return None

# Display files with direct download buttons
for file in github_files:
    col1, col2 = st.columns([4, 1])
    
    with col1:
        st.subheader(file["title"])
    
    with col2:
        # Get file name from URL
        file_name = file["url"].split("/")[-1]
        
        # Pre-fetch the file content
        file_content = get_file_content(file["url"])
        
        if file_content is not None:
            # Use streamlit's built-in download_button for direct download
            st.download_button(
                label="Download",
                data=file_content,
                file_name=file_name,
                mime="text/plain",
                key=f"download_{file['title']}",
            )
        else:
            st.error("File not available")
    
