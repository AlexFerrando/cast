# For licensing see accompanying LICENSE file.
# Copyright (C) 2025Apple Inc. All Rights Reserved.

import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer


class SentenceDataset(torch.utils.data.Dataset):
    def __init__(self, sentences: list[str], num_sentences: int = 20000):
        self.sentences = sentences
        if len(self.sentences) > num_sentences:
            self.sentences = self.sentences[:num_sentences]

    @staticmethod
    def dataset_names(cache_dir: Path = None) -> dict[str, Path]:
        return {
            "wikipedia": "/hhome/aferrando/ml-lineas/data/datasets/wikipedia/wikipedia_sentences.csv"
        }

    def __getitem__(self, item):
        return self.sentences[item]

    def __len__(self):
        return len(self.sentences)


@torch.no_grad()
def perplexity_batch(
    sentences: list[str],
    prompts: list[str] | None,
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    device: str,
    max_context_length: int | None = 128,
    max_generation_length: int | None = 50,
    autoregressive: bool = False,
) -> torch.Tensor:
    """
    Compute the perplexity of the passed ``sentences`` according to a specific ``model``.
    Args:
        sentences: A list of sentences
        prompts: A list of prompts
        tokenizer: Huggingface transformers tokenizer
        model: Huggingface transformers model
        device: Device identifier
        max_context_length: Max number of tokens considered. If the sentence is shorter, pad tokens are added.
        max_generation_length: Maximum number of newly generated tokens allowed.
        autoregressive: If True, use autoregressive decoding, otherwise use parallel decoding with causal masking.
    Returns:
        Perplexity per sentence in the batch
    """
    if autoregressive:
        return _autoregressive_perplexity_batch(
            sentences=sentences,
            prompts=prompts,
            tokenizer=tokenizer,
            model=model,
            device=device,
            max_context_length=max_context_length,
            max_generation_length=max_generation_length,
        )
    else:
        return _parallel_perplexity_batch(
            sentences=sentences,
            prompts=prompts,
            tokenizer=tokenizer,
            model=model,
            device=device,
            max_context_length=max_context_length,
            max_generation_length=max_generation_length,
        )


@torch.no_grad()
def _autoregressive_perplexity_batch(
    sentences: list[str],
    prompts: list[str] | None,
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    device: str,
    max_context_length: int | None = 128,
    max_generation_length: int | None = 50,
) -> torch.Tensor:
    """
    Autoregressive perplexity that matches the batched (parallel) version.
    """
    truncation = max_context_length is not None
    tokenizer.padding_side = "right"
    tok_s = tokenizer(
        text=sentences,
        return_tensors="pt",
        truncation=truncation,
        padding=truncation,
        max_length=max_generation_length,
        add_special_tokens=(prompts is None),
    ).to(device)
    tokenizer.padding_side = "left"

    if prompts is not None:
        side = tokenizer.truncation_side
        tokenizer.truncation_side = "left"
        tok_p = tokenizer(
            text=prompts,
            return_tensors="pt",
            truncation=truncation,
            padding=True,
            add_special_tokens=True,
            max_length=max_context_length,
        ).to(device)
        tokenizer.truncation_side = side
        tok_all = {k: torch.cat([tok_p[k], tok_s[k]], -1) for k in tok_p.keys()}
    else:
        tok_all = tok_s

    input_ids = tok_all["input_ids"]
    attention_mask = tok_all["attention_mask"]
    seq_lens = attention_mask.sum(-1)

    # Buffers
    ppls = torch.zeros(input_ids.shape[0], device=device, dtype=torch.float32)
    totals = torch.zeros_like(ppls)

    # Now include **all tokens after BOS** (like parallel version)
    for ctx_len in range(1, seq_lens.max()):
        mask = ctx_len < seq_lens
        _input_ids = input_ids[mask][:, :ctx_len]
        _attention_mask = attention_mask[mask][:, :ctx_len]
        logits = model(input_ids=_input_ids, attention_mask=_attention_mask).logits

        model.clean_leashes()
        loss = torch.nn.functional.cross_entropy(
            logits[:, -1],
            input_ids[mask][:, ctx_len].reshape(-1),
            reduction="none",
        )
        ppls[mask] += loss
        totals[mask] += 1

    return torch.exp(ppls / totals)


@torch.no_grad()
def _parallel_perplexity_batch(
    sentences: list[str],
    prompts: list[str] | None,
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    device: str,
    max_context_length: int | None = 128,
    max_generation_length: int | None = 50,
) -> torch.Tensor:
    """
    Compute the perplexity of the passed ``sentences`` according to a specific ``model``.
    Args:
        sentences: A list of sentences
        prompts: A list of prompts
        tokenizer: Huggingface transformers tokenizer
        model: Huggingface transformers model
        device: Device identifier
        max_context_length: Max number of tokens considered. If the sentence is shorter, pad tokens are added.
        max_generation_length: Maximum number of newly generated tokens allowed.
    Returns:
        Perplexity per sentence in the batch
    """
    truncation = max_context_length is not None
    padding_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    if prompts is not None:
        text = [p + s for p, s in zip(prompts, sentences, strict=True)]
    else:
        text = sentences
    tok_all = tokenizer(
        text=text,
        return_tensors="pt",
        truncation=truncation,
        padding=True,
        add_special_tokens=True,
        max_length=max_generation_length if prompts is None else max_context_length,
    ).to(device)
    tokenizer.padding_side = padding_side
    logits = model(
        input_ids=tok_all["input_ids"], attention_mask=tok_all["attention_mask"]
    ).logits
    model.clean_leashes()
    # Compute perplexity for last token (note that indexing at offset + ctx_len gives us the token id right after :(offset + ctx_len))
    loss = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]),
        tok_all["input_ids"][:, 1:].reshape(-1),
        reduction="none",
    )
    loss = (tok_all["attention_mask"][:, 1:] * loss.view(logits.shape[0], -1)).sum(
        -1
    ) / tok_all["attention_mask"][:, 1:].sum(-1)

    return torch.exp(loss)


def perplexity_sequential(
    sentences: list[str],
    prompts: list[str],
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    device: str,
) -> torch.Tensor:
    """
    Compute the perplexity of the passed ``sentences`` according to a specific ``model``.

    Inspired by https://colab.research.google.com/drive/1X2ZfC4y8Jx8FbkR7m-bLi8Ifrq-8MPTO#scrollTo=BehgQO-Nbvj0&line=30&uniqifier=1

    Args:
        sentences: A sequence of sentences
        prompts: A sequence of prompts
        tokenizer: Huggingface transformers tokenizer
        model: Huggingface transformers model
        device: Device identifier
    Returns:
        Perplexity per sentence in the batch
    """
    model.eval()
    ppls = []
    for s, p in zip(sentences, prompts, strict=True):
        if p is not None:
            tok_p = tokenizer(p, return_tensors="pt")
            len_p = tok_p["input_ids"].shape[1]
        else:
            len_p = 0
            p = ""

        tok = tokenizer(p + s, return_tensors="pt")

        # Build attention mask to not attend to prompt
        with torch.no_grad():
            outputs = model(**tok)

        # Make tuple of scores
        logits_cont = outputs.logits[
            :, len_p - 1 : -1, :
        ]  # shifting by one, since last token is a "future" token
        scores_cont = logits_cont.to(torch.float64).unbind(dim=1)

        # Compute transition log-likelihoods
        lls = model.compute_transition_scores(
            sequences=tok["input_ids"][:, len_p:],
            scores=scores_cont,
            normalize_logits=True,
        )
        ppl_cont = torch.exp(-torch.mean(lls)).item()
        ppls.append(ppl_cont)
    return torch.tensor(ppls, device=device)


def measure_perplexity(
    continuations: torch.utils.data.DataLoader | list[str],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: torch.utils.data.DataLoader | list[str] | None = None,
    device: str = None,
    batch_size: int | None = 128,
    autoregressive: bool = False,
) -> np.ndarray:
    ppl = []

    if prompts is not None:
        if isinstance(prompts, list):
            prompts = torch.utils.data.DataLoader(
                dataset=prompts,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,  # no preprocessing happening here
            )

    if isinstance(continuations, list):
        continuations = torch.utils.data.DataLoader(
            dataset=continuations,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,  # no preprocessing happening here
        )

    if prompts is not None:
        for c, p in tqdm(zip(continuations, prompts, strict=True)):
            ppl_batch = perplexity_batch(
                sentences=c,
                prompts=p,
                model=model,
                tokenizer=tokenizer,
                device=device,
                autoregressive=autoregressive,
            )
            ppl.append(ppl_batch)
    else:
        for c in tqdm(continuations):
            ppl_batch = perplexity_batch(
                sentences=c,
                prompts=None,
                model=model,
                tokenizer=tokenizer,
                device=device,
                autoregressive=autoregressive,
            )
            ppl.append(ppl_batch)

    ppl = (
        torch.cat(ppl)
        .detach()
        .to(device=torch.device("cpu"), dtype=torch.float32)
        .numpy()
    )
    return ppl
