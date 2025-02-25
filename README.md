# Web-QA

A simple web-based tool that allows users to:
1. Input one or more URLs
2. Ask questions about the content from those URLs
3. Get accurate answers based solely on the ingested information


## Features

- Clean, user-friendly interface
- Multi-URL support
- Context-aware answers from URL content
- Source citations with each answer
- Simple deployment options

## Requirements

- Python 3.8+
- Google API key with Gemini access

## Local Setup Instructions

1. Clone the repository or download the files

2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your Google API key:
   ```
   GOOGLE_API_KEY=your_google_api_key_here
   ```

4. Run the application:
   ```
   streamlit run app.py
   ```

5. Access the application in your browser at `http://localhost:8501`

## How to Use

1. Enter URLs in the text field and click "Add URL" for each URL you want to include
2. Click "Process URLs" to load and analyze the content
3. Enter your question about the URL content
4. Click "Get Answer" to receive a response based only on the information from the URLs

