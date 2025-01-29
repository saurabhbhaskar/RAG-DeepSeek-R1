import io
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings

def process_pdf(uploaded_file):
    # Save PDF temporarily
    with open("data/temp.pdf", "wb") as f:
        f.write(uploaded_file.getvalue())

    # Load PDF text
    loader = PDFPlumberLoader("data/temp.pdf")
    docs = loader.load()

    # Split text into semantic chunks
    text_splitter = SemanticChunker(HuggingFaceEmbeddings())
    documents = text_splitter.split_documents(docs)

    return documents
