from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, RetrievalQA, StuffDocumentsChain
from langchain_community.embeddings import HuggingFaceEmbeddings

def get_answer(user_input, documents):
    # Create embeddings and vector store
    embeddings = HuggingFaceEmbeddings()
    vector_store = FAISS.from_documents(documents, embeddings)

    # Connect retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})  # Fetch top 3 chunks

    # Set up DeepSeek model (1.5B variant)
    llm = Ollama(model="deepseek-r1")

    # Create the prompt template
    prompt = """
    1. Use ONLY the context below.
    2. If unsure, say "I don’t know".
    3. Keep answers under 4 sentences.

    Context: {context}

    Question: {question}

    Answer:
    """
    QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt)

    # Create the chain for generating answers
    llm_chain = LLMChain(llm=llm, prompt=QA_CHAIN_PROMPT)

    # Combine document chunks into a final answer
    document_prompt = PromptTemplate(
        template="Context:\ncontent:{page_content}\nsource:{source}",
        input_variables=["page_content", "source"]
    )

    qa = RetrievalQA(
        combine_documents_chain=StuffDocumentsChain(
            llm_chain=llm_chain,
            document_prompt=document_prompt,
            document_variable_name="context"  # ✅ This fixes the issue
        ),
        retriever=retriever
    )

    # Get the response based on the user input
    response = qa(user_input)["result"]

    return response
