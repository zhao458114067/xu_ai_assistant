import os
import warnings
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import onnxruntime as ort
from transformers import AutoTokenizer, AutoModel
from langchain_core.embeddings import Embeddings

from build.xu_ai_assistant.src.onnx_export import onnx_path
from src.onnx_export import output_dir


class OnnxEmbeddings(Embeddings):
    def __init__(self, model_name, device_type="auto"):
        """
        :param onnx_path: ONNX 文件路径或目录
        :param model_name: HF 模型名称
        :param device_type: auto / cuda / cpu / amd
        """
        self.use_onnx = os.path.exists(onnx_path)

        if self.use_onnx:
            self.session = self._init_onnx_session(onnx_path, device_type)
            self.output_name = self.session.get_outputs()[0].name
            self.tokenizer = AutoTokenizer.from_pretrained(output_dir, trust_remote_code=True)

        else:
            warnings.warn("未找到 ONNX 模型，将使用 Hugging Face PyTorch 模型，推理速度较慢")
            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model.eval()
            self.device = torch.device("cuda" if  torch.cuda.is_available() else "cpu")
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
        return [self._embed(text) for text in tqdm(texts, desc="生成嵌入向量")]

    def embed_query(self, text):
        return self._embed(text)

    def _embed(self, text):
        text = text.strip().replace("\n", " ")
        tokens = self.tokenizer(text, return_tensors="pt" if not self.use_onnx else "np", padding="max_length", truncation=True, max_length=512)

        if self.use_onnx:
            # position_ids = np.arange(tokens["input_ids"].shape[1])[None, :].astype("int64")
            ort_inputs = {
                "input_ids": tokens["input_ids"],
                "attention_mask": tokens["attention_mask"],
                # "position_ids": position_ids,
            }
            outputs = self.session.run([self.output_name], ort_inputs)[0]
            cls_embedding = outputs[:, 0, :]
            normed = F.normalize(torch.tensor(cls_embedding), p=2, dim=1).numpy()
            return normed[0].tolist()
        else:
            tokens = {k: v.to(self.device) for k, v in tokens.items()}
            with torch.no_grad():
                outputs = self.model(**tokens)
            cls_embedding = outputs.last_hidden_state[:, 0, :]
            normed = F.normalize(cls_embedding, p=2, dim=1).cpu().numpy()
            return normed[0].tolist()
