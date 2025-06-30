import torch
from lm_eval.api.model import LM
from activation_steering.leash_layer import LeashLayer
from tqdm import tqdm


class MyLeashedModel(LM):
    def __init__(self, model, tokenizer, device="cuda"):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def generate(self, requests):
        """
        requests: list of lm_eval.api.task.Instance for generation.
        Each request.arguments[0]: prompt string,
        request.arguments[1]: max generation tokens (int).
        """
        responses = []
        for request in requests:
            prompt = request.arguments[0]
            max_new_tokens = request.arguments[1]

            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            outputs = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )
            generated_tokens = outputs[0][inputs["input_ids"].shape[-1] :]
            decoded = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            responses.append(decoded)

            LeashLayer.clean_leashes()

        return responses

    def generate_until(self, requests):
        # Just use the same generate behavior
        return self.generate(requests)

    def loglikelihood(self, requests):
        """
        requests: list of lm_eval.api.task.Instance for loglikelihood.
        request.arguments[0]: prompt/context string
        request.arguments[1]: continuation/target string
        Returns: list of tuples (logprob, is_greedy)
        """
        results = []

        for request in tqdm(requests):
            prompt = request.arguments[0]
            continuation = request.arguments[1]

            enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            full = self.tokenizer(prompt + continuation, return_tensors="pt").to(
                self.device
            )

            with torch.no_grad():
                outputs = self.model(**full)
                logits = outputs.logits

            prompt_len = enc.input_ids.shape[-1]
            target_ids = full.input_ids[0][prompt_len:]
            pred_logits = logits[0, prompt_len - 1 : -1]  # shifted

            log_probs = torch.nn.functional.log_softmax(pred_logits, dim=-1)
            selected = log_probs[range(len(target_ids)), target_ids]
            total_logprob = selected.sum().item()

            greedy_tokens = log_probs.argmax(dim=-1)
            is_greedy = (greedy_tokens == target_ids).all().item()

            results.append((total_logprob, is_greedy))

            LeashLayer.clean_leashes()

        return results

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError()
