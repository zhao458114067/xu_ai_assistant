import os

# 配置输出目录
common_dir = "/ai_assistant"
model_name = "BAAI/bge-m3"
output_dir = common_dir + "/onnx_models/" + model_name
os.makedirs(output_dir, exist_ok=True)
onnx_model_name = model_name.replace("/", "_") + ".onnx"
onnx_path = f"{output_dir}/{onnx_model_name}"

# 持久化位置
VECTOR_STORE_PATH = common_dir + "/vector_store"
os.makedirs(VECTOR_STORE_PATH, exist_ok=True)

RETRIEVER_PATH = common_dir + "/retriever"
os.makedirs(RETRIEVER_PATH, exist_ok=True)

DATA_PATH = [
    common_dir + "/vector_repo"
]

# 向量仓库过滤
EXCLUDE_DIRS = {"__pycache__", "target", "logs", "log", "node_modules", "lib", ".git", ".github", "build", "dist",
                "resources", "test"}
INCLUDE_FILES_SOURCES = [".py", ".java", ".vue", ".js", ".ts", ".tsx", ".cjs", ".mjs", ".json", ".sh",
                         "dockerfile", ".properties", ".doc", ".docx", ".pdf", ".ppt", ".pptx", ".vbs"]