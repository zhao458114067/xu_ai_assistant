import hashlib
import os
from langchain_community.document_loaders import TextLoader, Docx2txtLoader, PyPDFLoader
from src.loader.doc_textract_loader import DocTextractLoader
from src.loader.ppt_text_loader import PPTXTextLoader
from src.common.constants import EXCLUDE_DIRS, INCLUDE_FILES_SOURCES, DATA_PATH

LOADER_MAP = {
    ".docx": Docx2txtLoader,
    ".doc": DocTextractLoader,
    ".pdf": PyPDFLoader,
    ".ppt": PPTXTextLoader,
    ".pptx": PPTXTextLoader,
    ".vbs": lambda path: TextLoader(path, encoding="gb18030"),
}


def get_loader(filepath: str):
    ext = os.path.splitext(filepath)[1].lower()
    loader_cls = LOADER_MAP.get(ext, lambda path: TextLoader(path, encoding="utf-8"))
    return loader_cls(filepath)


def load_all_documents(path):
    from tqdm import tqdm
    documents, file_list = [], []

    for root, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not d.startswith('.')]
        for file in files:
            if any(file.lower().endswith(ext) for ext in INCLUDE_FILES_SOURCES):
                file_list.append(os.path.join(root, file))

    for filepath in tqdm(file_list, desc="加载文档"):
        try:
            file_hash = compute_file_hash(filepath)
            loader = get_loader(filepath)
            docs = loader.load()
            relative_path = os.path.relpath(filepath, DATA_PATH)
            for doc in docs:
                doc.metadata["source"] = relative_path
                doc.metadata["file_hash"] = file_hash
            documents.extend(docs)
        except Exception as e:
            print(f"跳过 {filepath}（原因：{e}）")
    return documents


def compute_file_hash(filepath, algo="sha256"):
    hasher = hashlib.new(algo)
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()
