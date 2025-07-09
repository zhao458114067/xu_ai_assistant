from optimum.exporters.onnx import main_export
from transformers import AutoModel
import os

# 配置输出目录
model_name = "BAAI/bge-m3"
onnx_model_name = model_name.replace("/", "_") + ".onnx"
output_dir = "/onnx_models/" + model_name
os.makedirs(output_dir, exist_ok=True)  # 确保目录存在

if __name__ == '__main__':
    model = AutoModel.from_pretrained(model_name)

    # 导出模型
    main_export(
        model_name_or_path=model_name,
        output=output_dir,
        task="feature-extraction",
        opset=17
    )
    os.rename(f"{output_dir}/model.onnx", f"{output_dir}/{onnx_model_name}")

    print(f"模型已成功导出到: {onnx_model_name}")