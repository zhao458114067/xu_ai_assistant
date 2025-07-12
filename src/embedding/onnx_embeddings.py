import os
import warnings
import numpy as np
from tqdm import tqdm
import onnxruntime as ort
from transformers import AutoTokenizer, AutoModel
from langchain_core.embeddings import Embeddings

from src.common.constants import onnx_path, output_dir


class OnnxEmbeddings(Embeddings):
    def __init__(self, model_name, batch_size=1, device_type="auto"):
        self.batch_size = batch_size
        self.use_onnx = os.path.exists(onnx_path)

        if self.use_onnx:
            self.session = self._init_onnx_session(onnx_path, device_type)
            self.output_name = self.session.get_outputs()[0].name
            self.tokenizer = AutoTokenizer.from_pretrained(output_dir, trust_remote_code=True)

        else:
            import torch
            warnings.warn("未找到 ONNX 模型，将使用 Hugging Face PyTorch 模型，推理速度较慢")
            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model.eval()
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)

    def _init_onnx_session(self, onnx_path, device_type):
        if device_type == "auto":
            providers = self._detect_providers()
        else:
            provider_map = {
                "amd": ["DmlExecutionProvider"],
                "cuda": ["CUDAExecutionProvider"],
                "cpu": ["CPUExecutionProvider"]
            }
            providers = provider_map.get(device_type.lower(), ["CPUExecutionProvider"])

        try:
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            return ort.InferenceSession(onnx_path, providers=providers, sess_options=sess_options)
        except Exception as e:
            warnings.warn(f"无法使用硬件加速: {e}，回退到 CPU")
            return ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    def _detect_providers(self):
        available = ort.get_available_providers()
        priority = ["DmlExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
        return [p for p in priority if p in available]

    def embed_documents(self, texts):
        embeddings = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc="生成嵌入向量"):
            batch_texts = texts[i:i + self.batch_size]
            batch_embeddings = self._embed_batch(batch_texts)
            embeddings.extend(batch_embeddings)
        return embeddings

    def _embed_batch(self, texts):
        # 预处理所有文本
        tokens = self.tokenizer(texts, return_tensors="np",
                                padding="max_length",
                                truncation=True,
                                max_length=512)
        ort_inputs = {
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
        }
        outputs = self.session.run([self.output_name], ort_inputs)[0]
        cls_embedding = outputs[:, 0, :]  # [batch, hidden]
        norm = np.linalg.norm(cls_embedding, axis=1, keepdims=True)
        normed = cls_embedding / (norm + 1e-10)
        return normed.tolist()

    def embed_query(self, text):
        return self._embed(text)

    def _embed(self, text):
        embeddings = self._embed_batch([text])
        return embeddings[0]
