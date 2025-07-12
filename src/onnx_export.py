from optimum.exporters.onnx import main_export
from transformers import AutoModel, AutoTokenizer
import os
from src.common.constants import model_name, output_dir, onnx_model_name

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
