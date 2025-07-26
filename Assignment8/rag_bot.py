from langchain_community.llms import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, RetrievalQA
from langchain.chains.combine_documents_stuff import StuffDocumentsChain
import os

def get_qa_chain(vectorstore):
    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.1",
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        temperature=0.3,
        max_new_tokens=256
    )
    retriever = vectorstore.as_retriever()

    prompt = PromptTemplate.from_template(
        "You are a helpful assistant. Use the following context to answer the question accurately.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )

    llm_chain = LLMChain(llm=llm, prompt=prompt)
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_variable_name="context"
    )
    qa_chain = RetrievalQA(
        retriever=retriever,
        combine_documents_chain=combine_documents_chain,
        return_source_documents=True
    )

    return qa_chain
