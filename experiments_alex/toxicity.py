import json
import torch
import pandas as pd
import numpy as np
import random
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from lm_eval.utils import make_table
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from activation_steering import SteeringDataset, SteeringVector, MalleableModel
from experiments_alex.utils import MyLeashedModel
from .perplexity import measure_perplexity, SentenceDataset

import typing as t


def load_model_and_tokenizer(model_name="Qwen/Qwen2.5-1.5B"):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def load_toxicity_data(path="/hhome/aferrando/ml-lineas/data/toxicity_rtp.json"):
    with open(path, "r") as file:
        data = json.load(file)
    target, source, control = data["target"], data["source"], data["control"]
    print(
        f"Loaded {len(target)} target examples, {len(source)} source examples, and {len(control)} control examples."
    )
    return target, source, control


def extract_or_load_vector(
    vector_path, examples, tokenizer, model, mode, method="pca_center"
):
    if mode == "load":
        return SteeringVector.load(vector_path)

    dataset = SteeringDataset(
        tokenizer=tokenizer,
        examples=examples,
        suffixes=None,
        disable_suffixes=True,
    )

    vector = SteeringVector.train(
        model=model,
        tokenizer=tokenizer,
        steering_dataset=dataset,
        method=method,
        accumulate_last_x_tokens="all",
        batch_size=32,
    )
    vector.save(vector_path)
    return vector


def find_best_condition_point(
    malleable_model, toxic, control, condition_vector, save_path
):
    return malleable_model.find_best_condition_point(
        positive_strings=toxic,
        negative_strings=control,
        condition_vector=condition_vector,
        layer_range=(0, 13),  # (0, 13)
        max_layers_to_combine=3,
        threshold_range=(0.0, 0.05),
        threshold_step=0.0005,
        save_analysis=True,
        file_path=save_path,
    )


def load_tet_prompts(
    path="/hhome/aferrando/ml-lineas/data/datasets/thoroughly_engineered_toxicity/thoroughly_engineered_toxicity.jsonl",
    n_examples=100,
    seed=42,
):
    toxic_prompts = []
    with open(path, "r") as file:
        for line in file.readlines():
            json_line = json.loads(line)
            if not any(["TOXIC" in metric for metric in json_line["metrics"]]):
                continue
            toxic_prompts.append(line)
    random.seed(seed)
    random.shuffle(toxic_prompts)
    return [json.loads(line)["query"] for line in toxic_prompts[:n_examples]]


def load_rtp_prompts(
    path="/hhome/aferrando/ml-lineas/data/datasets/prompted_gens_gpt2.jsonl",
    n_examples=100,
    seed=42,
):
    rtp_data = []
    with open(path) as fp:
        for line in fp:
            # Skip all sentences without toxicity score
            if json.loads(line)["prompt"]["toxicity"] is None:
                continue
            rtp_data.append(line)

    random.seed(seed)
    random.shuffle(rtp_data)
    return [json.loads(line)["prompt"]["text"] for line in rtp_data[:n_examples]]


def run_generation(
    malleable_model: MalleableModel,
    prompts: t.List[str],
    tokenizer: AutoTokenizer,
) -> t.List[str]:
    settings = {
        "pad_token_id": tokenizer.eos_token_id,
        "do_sample": True,
        "max_new_tokens": 50,
        "repetition_penalty": 1.2,
        "temperature": 1.0,
        "top_p": 0.3,
    }

    return malleable_model.respond_batch_sequential(
        prompts=prompts, settings=settings, use_chat_template=False
    )


def compute_toxicity_score(completions, batch_size=32):
    classifier = pipeline(
        "text-classification",
        "s-nlp/roberta_toxicity_classifier",
        device=torch.device("cuda"),
    )
    batched_preds = []
    for i in range(0, len(completions), batch_size):
        batch = completions[i : i + batch_size]
        preds = classifier(batch, padding=True, truncation=True)
        batched_preds.extend(preds)

    toxic_count = sum(1 for pred in batched_preds if pred["label"] == "toxic")
    return toxic_count / len(completions)


def find_cast_model(model_name="Qwen/Qwen2.5-1.5B", mode="load") -> MalleableModel:
    model, tokenizer = load_model_and_tokenizer(model_name)
    target, source, control = load_toxicity_data()

    refusal_behavior_vector = extract_or_load_vector(
        "./outputs/vectors/refusal_behavior_vector",
        examples=[(a, t) for a in target for t in source],
        tokenizer=tokenizer,
        model=model,
        mode=mode,
    )

    condition_vector = extract_or_load_vector(
        "./outputs/vectors/toxicity_condition_vector",
        examples=[(t, c) for t in source for c in control],
        tokenizer=tokenizer,
        model=model,
        mode=mode,
    )

    malleable_model = MalleableModel(model=model, tokenizer=tokenizer)

    # best_layers, best_threshold, best_direction, f1 = find_best_condition_point(
    #     malleable_model,
    #     source,
    #     control,
    #     condition_vector,
    #     save_path="./outputs/optimal_condition_points/optimal_condition_point_toxicity_condition_vector.json",
    # )

    # print(
    #     f"Best condition point found at layer {best_layers}, threshold {best_threshold}, "
    #     f"direction {best_direction} (f1={f1:.4f})"
    # )

    # Steer model
    malleable_model.steer(
        behavior_vector=refusal_behavior_vector,
        behavior_layer_ids=[15, 16, 17, 18, 19, 20, 21, 22, 23],
        behavior_vector_strength=1.0,
        condition_vector=condition_vector,
        condition_layer_ids=[3, 4],
        condition_vector_threshold=0.024,
        condition_comparator_threshold_is="larger",
    )

    return malleable_model


def generate_and_compute_toxicity(model: MalleableModel, prompts: t.List[str]) -> float:
    completions = run_generation(
        model,
        prompts,
        model.tokenizer,
    )
    score = compute_toxicity_score(completions)
    return score


def evaluate_toxicity(
    model: MalleableModel, n_tet_examples: int = 100, n_rtp_examples: int = 100
):
    if n_tet_examples > 0:
        tet_prompts = load_tet_prompts(n_examples=n_tet_examples)
        print(f"Loaded {len(tet_prompts)} TET prompts for toxicity evaluation.")
        tet_score = generate_and_compute_toxicity(model, tet_prompts)
        print(f"TET toxicity score: {tet_score:.4f}")
    if n_rtp_examples > 0:
        rtp_prompts = load_rtp_prompts(n_examples=n_rtp_examples)
        print(f"Loaded {len(rtp_prompts)} RTP prompts for toxicity evaluation.")
        rtp_score = generate_and_compute_toxicity(model, rtp_prompts)
        print(f"RTP toxicity score: {rtp_score:.4f}")


def evaluate_perplexity(model: MalleableModel):
    wikipedia_path = SentenceDataset.dataset_names()["wikipedia"]
    df = pd.read_csv(wikipedia_path)
    if len(df) > 20000:
        df = df.iloc[:20000]
    sentences = df.text.values.tolist()
    print(
        f"Loaded {len(sentences)} sentences from Wikipedia for perplexity evaluation."
    )
    perplexities = measure_perplexity(
        continuations=sentences,
        prompts=None,
        batch_size=128,
        model=model,
        tokenizer=model.tokenizer,
        device=torch.device("cuda"),
        autoregressive=True,
    )
    ppl_results = {
        f"perplexity": float(np.nanmean(perplexities)),
        f"perplexity-std": float(np.nanstd(perplexities)),
    }
    print(
        f"Perplexity score: {ppl_results['perplexity']:.4f} "
        f"(std: {ppl_results['perplexity-std']:.4f})"
    )


def evaluate_mmlu(model: MalleableModel):
    lm = MyLeashedModel(
        model=model,
        tokenizer=model.tokenizer,
        device=torch.device("cuda"),
    )

    results = evaluator.simple_evaluate(
        lm,
        tasks=["mmlu"],
        num_fewshot=5,
        limit=None,
        bootstrap_iters=100000,
        random_seed=42,
        numpy_random_seed=42,
        torch_random_seed=42,
        device=torch.device("cuda"),
        batch_size="auto",
        cache_requests=True,
    )
    results.pop("samples", None)  # Remove samples from results
    print(make_table(results))


def main(mode=""):
    malleable_model = find_cast_model(mode=mode)
    evaluate_perplexity(malleable_model)
    evaluate_toxicity(malleable_model, n_tet_examples=1230, n_rtp_examples=0)
    evaluate_mmlu(malleable_model)


if __name__ == "__main__":
    main(mode="")  # or "load"
