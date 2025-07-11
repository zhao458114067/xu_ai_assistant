from optimum.exporters.onnx import main_export
from transformers import AutoModel, AutoTokenizer
import os
import shutil

# 配置输出目录
model_name = "BAAI/bge-m3"
output_dir = "/onnx_models/" + model_name
onnx_model_name = model_name.replace("/", "_") + ".onnx"
onnx_path = f"{output_dir}/{onnx_model_name}"

os.makedirs(output_dir, exist_ok=True)

if __name__ == '__main__':
    # 保存 tokenizer
    print("正在导出tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)

    # 导出 ONNX 模型
    print("正在导出ONNX 模型")
    main_export(
        model_name_or_path=model_name,
        output=output_dir,
        task="feature-extraction",
        opset=17
    )

    # 重命名模型文件
    os.rename(f"{output_dir}/model.onnx", f"{output_dir}/{onnx_model_name}")

    print(f"模型和 tokenizer 已成功导出到: {output_dir}")
