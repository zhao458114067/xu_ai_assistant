import os
import sys

import nltk
import torch
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, Docx2txtLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from tqdm import tqdm

from src.embedding.onnx_embeddings import OnnxEmbeddings
from src.loader.doc_textract_loader import DocTextractLoader
from src.loader.ppt_text_loader import PPTXTextLoader
from src.loader.tree_sitter_splitter import split_code_with_tree_sitter, LANGUAGE_MAPPING
from src.onnx_export import output_dir, onnx_model_name, model_name

VECTOR_STORE_PATH = "/vector_store"
os.makedirs(VECTOR_STORE_PATH, exist_ok=True)

DATA_PATH = [
    "/vector_repo"
]
EXCLUDE_DIRS = {"__pycache__", "target", "logs", "log", "node_modules", "lib", ".git", ".github", "build", "dist",
                "resources", "test"}
INCLUDE_FILES_SOURCES = [".py", ".java", ".vue", ".js", ".ts", ".tsx", ".cjs", ".mjs", ".json", ".sh",
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
    print("正在加载文档...")
    documents = load_documents(DATA_PATH)

    # 加载 ONNX 向量模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"正在使用 {device} 创建向量模型...")
    embeddings = OnnxEmbeddings(
        onnx_path=os.path.join(output_dir, onnx_model_name),
        model_name=model_name
    )

    # 加载已有向量库（如果存在）
    vectorstore = None
    existing_sources = set()
    faiss_index_path = os.path.join(VECTOR_STORE_PATH, "index.faiss")
    if os.path.exists(faiss_index_path):
        print("检测到向量库已存在，尝试加载增量更新...")
        try:
            vectorstore = FAISS.load_local(VECTOR_STORE_PATH, embeddings=embeddings,
                                           allow_dangerous_deserialization=True)
            # 使用 dummy 查询获取所有 source
            docs = vectorstore.similarity_search("passage", k=9999999)
            existing_sources = {doc.metadata.get("source") for doc in docs if "source" in doc.metadata}
            print(f"已加载 {len(existing_sources)} 个历史文档源")
        except Exception as e:
            print(f"加载向量库失败，将重新生成（原因：{e}）")

    print("正在切分文本...")
    chunks = []
    new_file_count = 0
    for doc in documents:
        source = doc.metadata.get("source", "")
        if source in existing_sources:
            continue  # 跳过已处理文件

        new_file_count += 1
        if any(source.endswith(ext) for ext in LANGUAGE_MAPPING.keys()):
            try:
                chunks.extend(split_code_with_tree_sitter(source))
            except Exception as e:
                print(f"代码解析失败 {source}：{e}")
        else:
            splits = RecursiveCharacterTextSplitter(
                chunk_size=600,
                chunk_overlap=100,
                separators=["\n\n", "\n", "。", ".", ";", "{", "}", "#", "//", " "]
            ).split_text(doc.page_content)
            for chunk_text in splits:
                chunks.append(Document(page_content="passage: " + chunk_text,
                                       metadata={"source": source}))

    print(f"共新增 {new_file_count} 个文件，生成 {len(chunks)} 个文本块")

    if not chunks:
        print("无新增内容，向量库保持不变")
        return

    if vectorstore:
        vectorstore.add_documents(chunks)
    else:
        vectorstore = FAISS.from_documents(chunks, embeddings)

    print("保存向量库中...")
    vectorstore.save_local(VECTOR_STORE_PATH)
    print("增量构建完成，现在可以开始提问了")


if __name__ == "__main__":
    sys.setrecursionlimit(20000)
    load_dotenv()
    nltk.download('punkt')
    main()
