import argparse
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)


from datasets import load_dataset

# TODO: LORA
# TODO: QLORA
# TODO: inference
# TODO: safetensor

# ref: https://medium.com/@yxinli92/fine-tuning-large-language-models-with-deepspeed-a-step-by-step-guide-2fa6ce27f68a

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--local_rank", type=str)
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float16, trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds_config = {
        "train_micro_batch_size_per_gpu": 2,
        "gradient_accumulation_steps": 1,
        "fp16": {"enabled": True},
        "zero_optimization": {
            "stage": 3,
            "allgather_partitions": True,
            "reduce_scatter": True,
            "contiguous_gradients": True,
        },
    }

    # if args.local_rank == 0:
    # 가끔 huggingface_hub.errors.HfHubHTTPError: 429 Client Error: Too Many Requests for url: 에러가 나는데
    # 로컬 랭크 때문에 여러변 로드해서인걸로 추측
    # 테스트 데이터셋
    train_dataset = load_dataset("KorQuAD/squad_kor_v1", split="train[:100]")
    valid_dataset = load_dataset("KorQuAD/squad_kor_v1", split="validation[:10]")

    def tokenize_function(examples):
        inputs = []

        for question, answer in zip(examples["question"], examples["answers"]):

            messages = [
                {"role": "user", "content": question},
                {
                    "role": "assistant",
                    "content": answer["text"][0],
                },
            ]

            formatted_text = tokenizer.apply_chat_template(messages, tokenize=False)

            input_dict = tokenizer(
                formatted_text,
                padding="max_length",
                truncation=True,
                max_length=1024,
            )

            inputs.append(input_dict)

        return {
            "input_ids": [input["input_ids"] for input in inputs],
            "attention_mask": [input["attention_mask"] for input in inputs],
        }

    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
    tokenized_valid_dataset = valid_dataset.map(tokenize_function, batched=True)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir="./results",
        # per_device_train_batch_size=4,
        # gradient_accumulation_steps=2,
        per_device_train_batch_size=2,
        num_train_epochs=3,
        fp16=True,
        deepspeed=ds_config,
        save_safetensors=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_valid_dataset,
        data_collator=data_collator,
    )

    trainer.train()
