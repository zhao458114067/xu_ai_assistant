import os
from langchain_community.vectorstores import FAISS


def load_or_create_faiss(store_path, embeddings):
    faiss_index_path = os.path.join(store_path, "index.faiss")
    if not os.path.exists(faiss_index_path):
        return None, {}

    try:
        vectorstore = FAISS.load_local(store_path, embeddings=embeddings, allow_dangerous_deserialization=True)
        docs = vectorstore.similarity_search("passage", k=999999)
        existing_hashes = {doc.metadata.get("source"): doc.metadata.get("file_hash") for doc in docs}
        return vectorstore, existing_hashes
    except Exception as e:
        print(f"加载 FAISS 失败，原因：{e}")
        return None, {}
