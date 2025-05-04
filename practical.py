import streamlit as st
import requests
import base64

# Set page title and configuration
st.set_page_config(page_title="GitHub File Downloader", layout="wide")

# Add custom CSS to align download buttons to the right
st.markdown("""
<style>
.download-button {
    text-align: right !important;
    display: flex;
    justify-content: flex-end;
}
</style>
""", unsafe_allow_html=True)

# Title for the application
st.title("GitHub File Downloader")

# GitHub file links and titles
# Replace these with your actual GitHub raw links and titles
github_files = [
    {
        "title": "File 1",
        "url": "https://raw.githubusercontent.com/username/repository/main/file1.txt"
    },
    {
        "title": "File 2",
        "url": "https://raw.githubusercontent.com/username/repository/main/file2.csv"
    },
    {
        "title": "File 3",
        "url": "https://raw.githubusercontent.com/username/repository/main/file3.json"
    },
    {
        "title": "File 4",
        "url": "https://raw.githubusercontent.com/username/repository/main/file4.py"
    },
    {
        "title": "File 5",
        "url": "https://raw.githubusercontent.com/username/repository/main/file5.md"
    }
]

# Function to download file content from GitHub
def get_file_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.content
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching file: {e}")
        return None

# Function to create a download link
def get_download_link(file_content, file_name, file_label):
    b64 = base64.b64encode(file_content).decode()
    return f'<div class="download-button"><a href="data:application/octet-stream;base64,{b64}" download="{file_name}" class="btn">Download {file_label}</a></div>'

# Display files with download buttons on the right
for file in github_files:
    col1, col2 = st.columns([4, 1])
    
    with col1:
        st.subheader(file["title"])
    
    with col2:
        # Get file name from URL
        file_name = file["url"].split("/")[-1]
        
        # Create a button to trigger download
        if st.button(f"Download", key=file["title"]):
            file_content = get_file_content(file["url"])
            if file_content:
                # Create download link
                dl_link = get_download_link(file_content, file_name, file["title"])
                st.markdown(dl_link, unsafe_allow_html=True)
                st.success(f"{file['title']} download initiated!")
    
    # Add a separator between files
    st.markdown("---")

# Instructions for the user
st.markdown("""
### Instructions
1. Click the download button next to the file you want to download
2. The file will be downloaded directly to your computer
""")