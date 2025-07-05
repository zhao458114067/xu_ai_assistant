from langchain.document_loaders.base import BaseLoader
from langchain.schema import Document
import textract


class DocTextractLoader(BaseLoader):
    def __init__(self, file_path: str, encoding: str = "utf-8"):
        self.file_path = file_path
        self.encoding = encoding

    def load(self) -> list[Document]:
        try:
            text = textract.process(self.file_path).decode(self.encoding)
            return [Document(page_content=text, metadata={"source": self.file_path})]
        except Exception as e:
            raise RuntimeError(f"读取 {self.file_path} 失败：{e}")
