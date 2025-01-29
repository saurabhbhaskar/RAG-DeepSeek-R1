from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, RetrievalQA, StuffDocumentsChain
from langchain_community.embeddings import HuggingFaceEmbeddings

def get_answer(user_input, documents):
    embeddings = HuggingFaceEmbeddings()
    vector_store = FAISS.from_documents(documents, embeddings)

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    llm = Ollama(model="deepseek-r1")

    prompt = """
    1. Use ONLY the context below.
    2. If unsure, say "I don’t know".
    3. Keep answers under 4 sentences.

    Context: {context}

    Question: {question}

    Answer:
    """
    QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt)

    llm_chain = LLMChain(llm=llm, prompt=QA_CHAIN_PROMPT)

    document_prompt = PromptTemplate(
        template="Context:\ncontent:{page_content}\nsource:{source}",
        input_variables=["page_content", "source"]
    )

    qa = RetrievalQA(
        combine_documents_chain=StuffDocumentsChain(
            llm_chain=llm_chain,
            document_prompt=document_prompt,
            document_variable_name="context"
        ),
        retriever=retriever
    )

    response = qa(user_input)["result"]

    return response
