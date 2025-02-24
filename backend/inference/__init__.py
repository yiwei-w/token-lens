from abc import ABC, abstractmethod


class InferenceBackend(ABC):
    @abstractmethod
    def generate(self, prompt: str, temperature: float, max_new_tokens: int, top_k: int) -> dict:
        pass


def get_backend(name: str):
    if name == "huggingface":
        from .huggingface import HuggingFaceBackend
        return HuggingFaceBackend()
    elif name == "mock":
        from .mock import MockBackend
        return MockBackend()
    else:
        raise ValueError(f"Unknown backend: {name}")