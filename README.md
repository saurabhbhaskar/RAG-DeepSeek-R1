### **RAG-DeepSeek-R1**  
A **Retrieval-Augmented Generation (RAG) pipeline** using **DeepSeek-R1** LLM for document-based Q&A.  

## 🚀 **How It Works**  
1. **Upload a PDF** – The app processes and splits the document into chunks.  
2. **Embed & Store** – Chunks are converted into embeddings using **HuggingFaceEmbeddings** and stored in **FAISS**.  
3. **Retrieve & Generate** – On user input, relevant chunks are retrieved and passed to **DeepSeek-R1** to generate an answer.  

## 📌 **Setup & Run**  
### **1️⃣ Install Dependencies**  
```sh
pip install -r requirements.txt
```

### **2️⃣ Run the App**  
```sh
streamlit run app.py
```

## 📂 **Project Structure**  
```
RAG-DeepSeek-R1/
│── app.py                    # Main Streamlit app  
│── scripts/
│   ├── pdf_processing.py      # Processes PDF files  
│   ├── rag_pipeline.py        # Handles retrieval & generation  
│── data/                      # Stores uploaded PDFs  
│── requirements.txt           # Required dependencies  
│── README.md                  # Project info  
```

## 🔥 **Tech Stack**  
- **LLM:** DeepSeek-R1  
- **Vector DB:** FAISS  
- **Framework:** Streamlit  
- **Embeddings:** HuggingFace  

📢 **Contributions & feedback are welcome!** 🚀  
