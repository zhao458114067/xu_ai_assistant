import os
import nltk
from tqdm import tqdm
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

DATA_PATH = "D:\idea_workspace\zx-rich\src"
VECTOR_STORE_PATH = "../vector_store"

def load_documents(path: str):
    documents = []
    file_list = []

    for root, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != "__pycache__"]
        for file in files:
            if file.startswith('.'):
                continue
            file_list.append(os.path.join(root, file))

    print(f"📄 共发现 {len(file_list)} 个文件，开始加载…")

    for filepath in tqdm(file_list, desc="加载文档"):
        try:
            loader = UnstructuredFileLoader(filepath)
            documents.extend(loader.load())
        except Exception as e:
            print(f"⚠️ 跳过 {filepath}（原因：{e}）")
    return documents

def main():
    print("正在加载文档...")
    documents = load_documents(DATA_PATH)

    print("正在切分文本...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(documents)
    print(f"共切分为 {len(chunks)} 个文本块")

    print("正在创建向量库（使用 bge-small-zh）...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh",
        model_kwargs={"device": "cpu"}
    )

    # 加上嵌入进度条
    vectors = []
    for chunk in tqdm(chunks, desc="生成嵌入"):
        vectors.append(chunk)
    vectorstore = FAISS.from_documents(vectors, embeddings)

    print("正在保存向量数据库...")
    vectorstore.save_local(VECTOR_STORE_PATH)
    print("完成！你现在可以开始提问了")

if __name__ == "__main__":
    nltk.download('punkt')
    main()
