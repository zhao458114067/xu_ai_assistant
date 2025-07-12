from langchain_core.documents import Document
import os

from tree_sitter_languages import get_language
from tree_sitter_languages.core import Parser

from src.common.constants import DATA_PATH

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


def extract_file_context(code: str, lang: str):
    """
    提取文件开头连续的import语句和注释作为上下文
    """
    lines = code.splitlines()
    context_lines = []
    if lang == "python":
        for line in lines:
            s = line.strip()
            if s.startswith(("import ", "from ", "#")) or s == "":
                context_lines.append(line)
            else:
                break
    elif lang == "java":
        for line in lines:
            s = line.strip()
            if s.startswith(("import ", "//", "/*", "*")) or s == "":
                context_lines.append(line)
            else:
                break
    elif lang in ("javascript", "typescript", "tsx"):
        for line in lines:
            s = line.strip()
            if s.startswith(("import ", "//", "/*", "*")) or s == "":
                context_lines.append(line)
            else:
                break
    else:
        # 其他语言暂时不提取上下文
        return ""

    return "\n".join(context_lines)


def split_code_with_tree_sitter(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    lang = LANGUAGE_MAPPING.get(ext)
    if not lang:
        return []

    language_obj = get_language(lang)
    parser = Parser()
    parser.set_language(language_obj)

    with open(os.path.join(DATA_PATH, filepath), "r", encoding="utf-8", errors="ignore") as f:
        code = f.read()

    file_context = extract_file_context(code, lang)

    tree = parser.parse(bytes(code, "utf8"))
    root = tree.root_node
    chunks = []

    def walk_nodes(node):
        if node.type in TARGET_NODES.get(lang, []):
            start, end = node.start_byte, node.end_byte
            code_block = bytes(code, 'utf-8')[start:end].decode('utf-8', errors='ignore').strip()
            if len(code_block) > 20:
                # 拼接文件头上下文和代码块
                full_content = (file_context + "\n\n" + code_block).strip()
                chunks.append(Document(
                    page_content=f"File: {filepath}\nCode:\n{full_content}",
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
