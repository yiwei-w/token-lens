from abc import ABC, abstractmethod


class InferenceBackend(ABC):
    @abstractmethod
    def generate(self, prompt: str, temperature: float, max_new_tokens: int, top_k: int) -> dict:
        pass
    
    @abstractmethod
    def generate_tree(self, prompt: str, temperature: float, max_depth: int, top_k: int, top_p: float, min_p: float) -> dict:
        pass


def get_backend(name: str):
    if name == "huggingface":
        from .huggingface_backend import HuggingFaceBackend
        return HuggingFaceBackend()
    elif name == "mock":
        from .mock_backend import MockBackend
        return MockBackend()
    elif name == "vllm":
        from .vllm_backend import VLLMBackend
        return VLLMBackend()
    else:
        raise ValueError(f"Unknown backend: {name}")