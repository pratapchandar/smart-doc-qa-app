import streamlit as st
from qa_engine import get_qa_chain

st.set_page_config(page_title="Smart Document QA", page_icon="📄")
st.title("📄 Smart Document QA App")

st.markdown("""
Welcome to the Smart Document QA App! This application answers questions based on pre-processed PDF documents.
""")

# Initialize the QA chain only once
@st.cache_resource
def load_qa_chain():
    return get_qa_chain()

try:
    qa_chain = load_qa_chain()
    
    st.write("### Ask a question about the documents")
    
    # User input
    question = st.text_input("Your question:", placeholder="e.g., What is the main topic of the document?")
    
    if question:
        with st.spinner("Finding answer..."):
            try:
                response = qa_chain.invoke({"input": question})
                st.write("**Answer:**")
                st.write(response["answer"])
                
                # Optional: Show source documents
                if "context" in response:
                    with st.expander("View source context"):
                        st.write(response["context"])
                        
            except Exception as e:
                st.error(f"An error occurred while processing your question: {e}")
                
except Exception as e:
    st.error(f"Failed to initialize QA engine: {e}")
    st.info("Please make sure the FAISS index is properly set up and the API key is configured.")
