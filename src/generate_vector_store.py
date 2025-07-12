import logging
import sys

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS

from src.common.constants import model_name, VECTOR_STORE_PATH, DATA_PATH
from src.embedding.onnx_embeddings import OnnxEmbeddings
from src.loader.document_loader import load_all_documents
from src.util.chunker import chunk_documents
from src.util.vector_store import load_or_create_faiss


def main():
    load_dotenv()
    sys.setrecursionlimit(20000)

    logging.info("开始加载文档...")
    documents = load_all_documents(DATA_PATH)
    logging.info(f"共加载原始文档：{len(documents)}")

    logging.info("加载或初始化向量库...")
    embeddings = OnnxEmbeddings(model_name=model_name)
    vectorstore, existing_hashes = load_or_create_faiss(VECTOR_STORE_PATH, embeddings)

    logging.info("开始切分文档...")
    chunks = chunk_documents(documents, existing_hashes)
    logging.info(f"新增 {len(chunks)} 个文本块")

    if not chunks:
        logging.info("无新增内容，向量库保持不变")
        return

    if vectorstore:
        vectorstore.add_documents(chunks, batch_size=64)
    else:
        vectorstore = FAISS.from_documents(chunks, embeddings)

    logging.info("保存向量库...")
    vectorstore.save_local(VECTOR_STORE_PATH)
    logging.info("完成")


if __name__ == "__main__":
    main()
