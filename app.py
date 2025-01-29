import streamlit as st
from scripts.pdf_processing import process_pdf
from scripts.rag_pipeline import get_answer

def main():
    st.title("Local RAG System with DeepSeek R1")

    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    
    if uploaded_file:
        st.write("Processing the PDF...")
        docs = process_pdf(uploaded_file)

        user_input = st.text_input("Ask your PDF a question:")
        
        if user_input:
            st.write("Fetching answer...")
            response = get_answer(user_input, docs)
            st.write(response)

if __name__ == "__main__":
    main()

