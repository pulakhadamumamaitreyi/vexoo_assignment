from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
import torch

# -----------------------------
# Load Dataset
# -----------------------------

dataset = load_dataset("openai/gsm8k", "main")

train_data = dataset["train"].select(range(3000))
test_data = dataset["test"].select(range(1000))

# -----------------------------
# Tokenizer
# -----------------------------

model_name = "meta-llama/Llama-3.2-1B"  # simulate if not available

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token


def preprocess(example):
    prompt = f"Question: {example['question']}\nAnswer:"
    target = example["answer"]

    full_text = prompt + " " + target

    tokenized = tokenizer(
        full_text,
        truncation=True,
        padding="max_length",
        max_length=256
    )

    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


train_data = train_data.map(preprocess)
test_data = test_data.map(preprocess)

# -----------------------------
# Model + LoRA
# -----------------------------

model = AutoModelForCausalLM.from_pretrained(model_name)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none"
)

model = get_peft_model(model, lora_config)

# -----------------------------
# Training
# -----------------------------

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=1,
    logging_steps=50,
    evaluation_strategy="epoch",
    save_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
)

trainer.train()

# -----------------------------
# Evaluation
# -----------------------------

def evaluate(model, dataset):
    correct = 0

    for sample in dataset:
        inputs = tokenizer(sample["question"], return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=50)

        pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if sample["answer"].strip() in pred:
            correct += 1

    accuracy = correct / len(dataset)
    print("Accuracy:", accuracy)


evaluate(model, test_data)
