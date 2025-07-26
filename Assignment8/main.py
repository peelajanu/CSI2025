import streamlit as st
from ragchatbot.data_loader import load_csv_as_documents
from ragchatbot.vectorstore import create_vectorstore
from ragchatbot.rag_bot import get_qa_chain

st.set_page_config(page_title="Loan Approval Chatbot", layout="wide")
st.title("🤖 Loan Approval RAG Chatbot")

# Cache setup so the vectorstore is created only once
@st.cache_resource
def setup():
    st.info("Loading and indexing documents. Please wait...", icon="📄")
    docs = load_csv_as_documents("data/Training Dataset.csv")
    vectordb = create_vectorstore(docs)
    return get_qa_chain(vectordb)

qa_chain = setup()

# Text input for user's question
question = st.text_input("Ask a question from the dataset:")

if question:
    with st.spinner("Searching for the answer..."):
        try:
            response = qa_chain.invoke({"query": question})
            st.success(response["result"])

            # Optionally display source documents
            with st.expander("Show source documents"):
                for i, doc in enumerate(response["source_documents"], 1):
                    st.markdown(f"**Document {i}:**")
                    st.text(doc.page_content)

        except Exception as e:
            st.error(f"⚠️ An error occurred: {e}")
