# Smart Document QA App

A Streamlit-based Question & Answer application that uses LangChain, FAISS, and OpenAI to answer questions based on PDF documents.

## Deployment Instructions

This app is deployed on Streamlit Cloud.

### Setup on Streamlit Cloud

1. Fork/clone this repository
2. Go to https://share.streamlit.io/
3. Click "New app"
4. Select this repository
5. Set main file path to `app.py`
6. Add your OpenAI API key in Secrets:
```toml
   OPENAI_API_KEY = "your-key-here"
```
7. Deploy!

## Local Testing
```bash
pip install -r requirements.txt
streamlit run app.py
```

Make sure to set OPENAI_API_KEY in your environment.
