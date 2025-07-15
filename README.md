# DocuTalk

A Streamlit app that allows you to chat with the contents of your PDF files using Google’s Gemini API. Just upload your PDFs, ask questions, and get intelligent, context-based answers!

 Features
-  Secure input for Gemini API Key
-  Upload and read multiple PDF files
-  Automatically splits and processes text
-  Embedding with `embedding-001` model
-  Question-answering with `gemini-2.0-flash-thinking-exp-01-21`
-  Select Top-K relevant chunks for response

Tech Stack
- Python
- [Streamlit](https://streamlit.io/)
- [Google Generative AI](https://ai.google.dev/)
- [PyPDF2](https://pypi.org/project/PyPDF2/)
- [LangChain](https://www.langchain.com/)
- [scikit-learn](https://scikit-learn.org/stable/)

Getting Started

1. Clone the repository
git clone https://github.com/titiksha62/Chat_with_PDF.git
cd Chat_with_PDF

3. Install dependencies
pip install -r requirements.txt

4. Run the app
streamlit run pdf_reader.py

API Key Setup
To use Gemini:
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Generate an API key
3. Paste it in the sidebar of the app when prompted

Example Use Case
Upload your college notes or research PDFs, then ask:
"Summarize the key points from Chapter 3"*
"What are the applications of AI discussed in this document?"
