import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from activation_steering import SteeringDataset, SteeringVector, MalleableModel

### Extract Refusal Behavior Vector and Save ###

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B", device_map="auto", torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")

# Load data
with open("/hhome/aferrando/cast/docs/demo-data/alpaca.json", "r") as file:
    alpaca_data = json.load(file)

with open("/hhome/aferrando/cast/docs/demo-data/behavior_refusal.json", "r") as file:
    refusal_data = json.load(file)

questions = alpaca_data["train"]
refusal = refusal_data["non_compliant_responses"]
compliace = refusal_data["compliant_responses"]

print(
    f"Loaded {len(questions)} questions, {len(refusal)} refusal responses and {len(compliace)} compliant responses."
)

# Create our dataset
refusal_behavior_dataset = SteeringDataset(
    tokenizer=tokenizer,
    examples=[(item["question"], item["question"]) for item in questions[:100]],
    suffixes=list(zip(refusal[:100], compliace[:100])),
)

# Extract behavior vector for this setup with 8B model, 10000 examples, a100 GPU, batch size 16 -> should take around 6 minutes
# To mimic setup from Representation Engineering: A Top-Down Approach to AI Transparency, do method = "pca_diff" amd accumulate_last_x_tokens=1
refusal_behavior_vector = SteeringVector.train(
    model=model,
    tokenizer=tokenizer,
    steering_dataset=refusal_behavior_dataset,
    method="pca_center",
    accumulate_last_x_tokens="suffix-only",
    batch_size=16,
)

# Let's save this behavior vector for later use
refusal_behavior_vector.save("./outputs/vectors/refusal_behavior_vector")


#### Extract Legal Condition Vector and Save ###

# Load data
with open("/hhome/aferrando/cast/docs/demo-data/condition_multiple.json", "r") as file:
    condition_data = json.load(file)

data = {"base": [], "sexual": [], "legal": [], "hate": [], "crime": [], "health": []}
for instance in condition_data["train"]:
    data["base"].append(instance["base"])
    data["sexual"].append(instance["sexual_content"])
    data["legal"].append(instance["legal_opinion"])
    data["hate"].append(instance["hate_speech"])
    data["crime"].append(instance["crime_planning"])
    data["health"].append(instance["health_consultation"])

# Here, we contrast one category of prompts to the other five categories and create a legal condition vector
target_condition = "legal"
other_conditions = ["base", "sexual", "hate", "crime", "health"]

positive_instructions = []
negative_instructions = []
for other_condition in other_conditions:
    positive_instructions.extend(data[target_condition])
    negative_instructions.extend(data[other_condition])

# Create our dataset
condition_dataset = SteeringDataset(
    tokenizer=tokenizer,
    examples=list(zip(positive_instructions, negative_instructions)),
    suffixes=None,
    disable_suffixes=True,
)

# Extract condition vector for this setup with 8B model, 4050 examples, a100 GPU -> should take around 90 seconds
condition_vector = SteeringVector.train(
    model=model,
    tokenizer=tokenizer,
    steering_dataset=condition_dataset,
    method="pca_center",
    accumulate_last_x_tokens="all",
)

# Let's save this condition vector for later use
condition_vector.save(f"./outputs/{target_condition}_condition_vector")

# Let's find the best condition points too
malleable_model = MalleableModel(model=model, tokenizer=tokenizer)

# Feel free to adjust the setup
best_layer, best_threshold, best_direction, _ = (
    malleable_model.find_best_condition_point(
        positive_strings=positive_instructions,
        negative_strings=negative_instructions,
        condition_vector=condition_vector,
        layer_range=(1, 10),
        max_layers_to_combine=1,
        threshold_range=(0.0, 0.06),
        threshold_step=0.0001,
        save_analysis=True,
        file_path=f"optimal_condition_point_{target_condition}_condition_vector.json",
    )
)
