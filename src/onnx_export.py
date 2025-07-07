from transformers import AutoModel, AutoTokenizer
import torch
import os

# 配置输出目录
output_dir = "/onnx_models"  # 可以修改为任何您想要的路径
os.makedirs(output_dir, exist_ok=True)  # 确保目录存在

if __name__ == '__main__':
    model_name = "intfloat/multilingual-e5-large"
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 创建示例输入
    inputs = tokenizer("passage: dummy text", return_tensors="pt")

    # 构建输出路径（处理模型名称中的斜杠）
    output_path = os.path.join(output_dir, model_name.replace("/", "_") + ".onnx")

    # 导出模型
    torch.onnx.export(
        model,
        (inputs["input_ids"], inputs["attention_mask"]),
        output_path,  # 使用完整路径
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "attention_mask": {0: "batch_size", 1: "seq_len"},
            "last_hidden_state": {0: "batch_size", 1: "seq_len"},
        },
        opset_version=17,
        do_constant_folding=True  # 启用常量折叠优化
    )

    print(f"模型已成功导出到: {os.path.abspath(output_path)}")