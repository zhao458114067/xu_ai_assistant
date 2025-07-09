from tree_sitter import Parser
from tree_sitter_languages import get_language
from langchain_core.documents import Document
import os

LANGUAGE_MAPPING = {
    ".py": "python",
    ".java": "java",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".cjs": "javascript",
    ".mjs": "javascript"
}

TARGET_NODES = {
    "python": ["function_definition", "class_definition"],
    "java": ["method_declaration", "class_declaration"],
    "javascript": ["function_declaration", "method_definition"],
    "typescript": ["function_declaration", "method_definition"],
    "tsx": ["function_declaration", "method_definition"]
}


def split_code_with_tree_sitter(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    lang = LANGUAGE_MAPPING.get(ext)
    if not lang:
        return []

    language_obj = get_language(lang)
    parser = Parser()
    parser.set_language(language_obj)

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        code = f.read()

    tree = parser.parse(bytes(code, "utf8"))
    root = tree.root_node
    chunks = []

    def walk_nodes(node):
        if node.type in TARGET_NODES.get(lang, []):
            start, end = node.start_byte, node.end_byte
            content = bytes(code, 'utf-8')[start:end].decode('utf-8', errors='ignore').strip()
            if len(content) > 20:
                chunks.append(Document(
                    page_content=f"File: {filepath}\nCode:\n{content}",
                    metadata={
                        "source": filepath,
                        "node_type": node.type,
                        "line_start": node.start_point[0] + 1,
                        "line_end": node.end_point[0] + 1
                    }
                ))
        for child in node.children:
            walk_nodes(child)

    walk_nodes(root)

    if not chunks:
        chunks.append(Document(
            page_content=f"File: {filepath}\nContent:\n{code[:2000]}",
            metadata={"source": filepath, "fallback": True}
        ))
    return chunks
