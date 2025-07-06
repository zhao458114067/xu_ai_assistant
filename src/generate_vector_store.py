import os
from concurrent.futures import ThreadPoolExecutor

import nltk
import torch
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, Docx2txtLoader, PyPDFLoader
from tqdm import tqdm
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.loader.doc_textract_loader import DocTextractLoader
from src.loader.ppt_text_loader import PPTXTextLoader

VECTOR_STORE_PATH = "../vector_store"
DATA_PATH = [
    "D:\\pycharm_workspace\\octopus_vector_repo",
]
EXCLUDE_DIRS = {"__pycache__", "target", "logs", "log", "node_modules", "lib", ".git", ".github", "build", "dist"}
INCLUDE_FILES_SOURCES = [".py", ".java", ".vue", ".js", ".ts", ".tsx", ".cjs", ".mjs", ".json", ".ini", ".sh",
                         "dockerfile", ".properties", ".doc", ".docx", ".pdf", ".ppt", ".pptx", ".vbs"]


def get_loader(filepath: str):
    filename = os.path.basename(filepath).lower()
    if any(keyword in filename for keyword in [".docx"]):
        return Docx2txtLoader(filepath)
    elif any(keyword in filename for keyword in [".doc"]):
        return DocTextractLoader(filepath)
    elif any(keyword in filename for keyword in [".pdf"]):
        return PyPDFLoader(filepath)
    elif any(keyword in filename for keyword in [".ppt", "pptx"]):
        return PPTXTextLoader(filepath)
    elif any(keyword in filename for keyword in [".vbs"]):
        return TextLoader(filepath, encoding="gb18030")
    else:
        return TextLoader(filepath, encoding="utf-8")


def load_documents(path_list: [str]):
    documents = []
    file_list = []

    for path in path_list:
        for root, dirs, files in os.walk(path):
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not d.startswith('.')]
            for file in files:
                filepath = os.path.join(root, file)
                filename = os.path.basename(filepath).lower()
                if any(keyword in filename for keyword in INCLUDE_FILES_SOURCES):
                    file_list.append(filepath)

    print(f"共发现 {len(file_list)} 个文件，开始加载…")

    for filepath in tqdm(file_list, desc="加载文档"):
        try:
            loader = get_loader(filepath)
            documents.extend(loader.load())
        except Exception as e:
            print(f"跳过 {filepath}（原因：{e}）")

    return documents


def main():
    if os.path.exists(VECTOR_STORE_PATH):
        print(f"检测到向量库已存在于 {VECTOR_STORE_PATH}，无需重复生成")
        return

    print("正在加载文档...")
    documents = load_documents(DATA_PATH)

    print("正在切分文本...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", "。", "，", ",", " "]
    )
    chunks = splitter.split_documents(documents)
    print(f"共切分为 {len(chunks)} 个文本块")

    print("正在创建向量模型...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large",
        model_kwargs={"device": device}
    )
    texts = [doc.page_content for doc in chunks]

    print("正在生成嵌入向量...")
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        vectors = list(tqdm(executor.map(embeddings.embed_query, texts), total=len(texts)))
    text_vector_pairs = list(zip(texts, vectors))
    vectorstore = FAISS.from_embeddings(text_vector_pairs, embeddings)

    # batch_size = 64
    # vectors = []
    #
    # for i in tqdm(range(0, len(texts), batch_size), desc=f"Embedding with {device}"):
    #     batch = texts[i:i + batch_size]
    #     batch_vecs = embeddings.embed_documents(batch)
    #     vectors.extend(batch_vecs)
    #
    # text_vector_pairs = list(zip(texts, vectors))
    # vectorstore = FAISS.from_embeddings(text_vector_pairs, embeddings)

    print("正在保存向量数据库...")
    vectorstore.save_local(VECTOR_STORE_PATH)
    print("向量库已生成，你现在可以开始提问了")


if __name__ == "__main__":
    load_dotenv()
    nltk.download('punkt')
    main()
