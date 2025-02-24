# backend/inference/huggingface.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from . import InferenceBackend


class HuggingFaceBackend(InferenceBackend):
    def __init__(self):
        self.model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def generate(self, prompt: str, temperature: float, max_new_tokens: int, top_k: int) -> dict:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        output = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=self.tokenizer.pad_token_id
        )
        full_ids = output.sequences[0]
        prompt_len = input_ids.shape[1]
        completion_ids = full_ids[prompt_len:]
        tokens_data = []
        for i, token_id in enumerate(completion_ids):
            logits = output.scores[i][0]  # shape: (vocab_size,)
            probs = torch.softmax(logits, dim=-1)
            token_prob = probs[token_id].item()
            log_prob = torch.log(probs[token_id]).item()
            entropy = -torch.sum(probs * torch.log(probs)).item()
            topk = torch.topk(logits, k=top_k)
            tokens_data.append({
                "text": self.tokenizer.decode([token_id]).strip(),
                "prob": token_prob,
                "log_prob": log_prob,
                "entropy": entropy,
                "top_tokens": [self.tokenizer.decode([tid]).strip() for tid in topk.indices.tolist()],
                "top_logits": topk.values.tolist()
            })
        return {
            "full_text": self.tokenizer.decode(full_ids, skip_special_tokens=True),
            "tokens": tokens_data
        }