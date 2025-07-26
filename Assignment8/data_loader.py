import pandas as pd
from langchain.docstore.document import Document

def load_csv_as_documents(csv_path):
    df = pd.read_csv(csv_path)
    documents = []
    for _, row in df.iterrows():
        content = "\n".join([f"{col}: {row[col]}" for col in df.columns])
        documents.append(Document(page_content=content))
    return documents
