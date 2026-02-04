import os
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain

def get_qa_chain():
    """
    Creates and returns a QA chain that uses a FAISS vector store for retrieval.
    
    Returns:
        A retrieval chain that can answer questions based on the indexed documents.
    """
    # Get API key from Streamlit secrets (for deployment) or environment variables
    api_key = None
    
    # Try to get from Streamlit secrets first (for Streamlit Cloud deployment)
    try:
        import streamlit as st
        api_key = st.secrets.get("OPENAI_API_KEY")
    except:
        pass
    
    # Fallback to environment variable
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found. Please set it in Streamlit secrets or environment variables."
        )

    # Load the FAISS index
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Path to FAISS index (relative to app.py)
    db_path = os.path.join(os.path.dirname(__file__), "data", "faiss_index")
    
    db = FAISS.load_local(
        folder_path=db_path,
        embeddings=embeddings,
        allow_dangerous_deserialization=True  # Required when loading from local file
    )

    # Initialize the LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=api_key, temperature=0)

    # Create the prompt template
    prompt = ChatPromptTemplate.from_template(
        """Answer the user's question based only on the provided context. If the answer is not in the context,
        clearly state that you don't know the answer and do not make up information.

        Context: {context}
        Question: {input}
        """
    )

    # Create the document chain and retrieval chain
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = db.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    return retrieval_chain
