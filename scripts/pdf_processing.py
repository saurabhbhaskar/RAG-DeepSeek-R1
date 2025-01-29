import io
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings

def process_pdf(uploaded_file):
    with open("data/temp.pdf", "wb") as f:
        f.write(uploaded_file.getvalue())

    loader = PDFPlumberLoader("data/temp.pdf")
    docs = loader.load()

    text_splitter = SemanticChunker(HuggingFaceEmbeddings())
    documents = text_splitter.split_documents(docs)

    return documents
