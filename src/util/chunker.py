import logging
import os
import hashlib
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.loader.tree_sitter_splitter import split_code_with_tree_sitter, LANGUAGE_MAPPING


def hash_text(text: str):
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def chunk_documents(documents, existing_hashes):
    chunks, seen_hashes = [], set()

    for doc in documents:
        source = doc.metadata.get("source", "")
        file_hash = doc.metadata.get("file_hash", "")
        # 内容没变，跳过
        if existing_hashes.get(source) == file_hash:
            continue
        if any(source.endswith(ext) for ext in LANGUAGE_MAPPING.keys()):
            try:
                code_chunks = split_code_with_tree_sitter(source)
                for c in code_chunks:
                    content_hash = hash_text(c.page_content)
                    if content_hash not in seen_hashes:
                        seen_hashes.add(content_hash)
                        chunks.append(c)
            except Exception as e:
                logging.exception(f"Tree-sitter 失败 {source}")
        else:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=600,
                chunk_overlap=150,
                separators=["\n\n", "\n", "。", ".", ";", "{", "}", "#", "//", " "]
            )
            splits = splitter.split_text(doc.page_content)
            for chunk_text in splits:
                full_text = f"passage: {chunk_text.strip()}"
                content_hash = hash_text(full_text)
                if content_hash not in seen_hashes:
                    seen_hashes.add(content_hash)
                    chunks.append(Document(
                        page_content=full_text,
                        metadata={"source": source}
                    ))
    return chunks
