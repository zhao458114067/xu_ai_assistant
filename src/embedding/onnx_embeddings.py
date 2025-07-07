import onnxruntime as ort
import torch
import torch.nn.functional as F
from langchain_core.embeddings import Embeddings
from tqdm import tqdm
from transformers import AutoTokenizer
import warnings


class OnnxEmbeddings(Embeddings):
    def __init__(self, onnx_path, model_name, device_type="auto"):
        """
        :param device_type: auto/amd/cuda/cpu
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.session = self._init_onnx_session(onnx_path, device_type)

    def _init_onnx_session(self, onnx_path, device_type):
        # 自动检测可用硬件
        if device_type == "auto":
            providers = self._detect_providers()
        else:
            provider_map = {
                "amd": ["DmlExecutionProvider"],
                "cuda": ["CUDAExecutionProvider"],
                "cpu": ["CPUExecutionProvider"]
            }
            providers = provider_map.get(device_type.lower(), ["CPUExecutionProvider"])

        # 初始化ONNX会话
        try:
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session = ort.InferenceSession(
                onnx_path,
                providers=providers,
                sess_options=sess_options
            )
            return session
        except Exception as e:
            warnings.warn(f"无法使用硬件加速: {e}，回退到CPU")
            return ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    def _detect_providers(self):
        """自动检测可用硬件"""
        available = ort.get_available_providers()
        priority = [
            "DmlExecutionProvider",  # AMD GPU (Windows)
            "CUDAExecutionProvider",  # NVIDIA GPU
            "CPUExecutionProvider"  # 保底
        ]
        return [p for p in priority if p in available]

    def embed_documents(self, texts):
        return [self._embed(text) for text in tqdm(texts, desc="生成嵌入向量")]

    def embed_query(self, text):
        return self._embed(text)

    def _embed(self, text):
        text = "passage: " + text.strip().replace("\n", " ")
        tokens = self.tokenizer(text, return_tensors="np", padding="max_length", truncation=True, max_length=512)
        ort_inputs = {
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"]
        }
        outputs = self.session.run(["last_hidden_state"], ort_inputs)[0]
        cls_embedding = outputs[:, 0, :]
        normed = F.normalize(torch.tensor(cls_embedding), p=2, dim=1).numpy()
        return normed[0].tolist()
