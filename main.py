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

if __name__ == "__main__":
    # https://medium.com/@yxinli92/fine-tuning-large-language-models-with-deepspeed-a-step-by-step-guide-2fa6ce27f68a

    # model_name = "maywell/Synatra-42dot-1.3B"
    model_name = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds_config = {
        # "train_batch_size": 8,
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

    # Load dataset and tokenize
    train_dataset = load_dataset("KorQuAD/squad_kor_v1", split="train[:100]")
    valid_dataset = load_dataset("KorQuAD/squad_kor_v1", split="validation[:10]")

    def tokenize_function(examples):
        inputs = []
        labels = []

        for question, answer in zip(examples["question"], examples["answers"]):
            input_dict = tokenizer(
                f"""
사용자: {question}
답변: {answer['text']}
""",
                padding="max_length",
                truncation=True,
                max_length=1024,
            )

            # input_dict["input_ids"] = torch.LongTensor(input_dict["input_ids"])
            # input_dict["attention_mask"] = torch.LongTensor(
            #     input_dict["attention_mask"]
            # )

            inputs.append(input_dict)

            # label = input_dict["input_ids"]
            # label = [l if l != tokenizer.pad_token_id else -100 for l in label]
            # label = label[1:] + [-100]
            # label = torch.LongTensor(label)
            # labels.append(label)

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
