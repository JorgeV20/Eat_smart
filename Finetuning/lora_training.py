import argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model


# ---------------------------
# Argument Parser
# ---------------------------
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_id", type=str,
                        default="HuggingFaceTB/SmolVLM2-2.2B-Instruct")
    parser.add_argument("--dataset_name", type=str,
                        default="Codatta/MM-Food-100K")
    parser.add_argument("--output_dir", type=str,
                        default="./smolvlm-dish-ingredients")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)

    return parser.parse_args()


def main():
    args = parse_args()

    print("Loading model in 4-bit...")

    # 4-bit quantization (QLoRA)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map="auto"
    )

    processor = AutoProcessor.from_pretrained(args.model_id)
    
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    print("Loading dataset...")
    dataset = load_dataset(args.dataset_name)
    train_dataset = dataset["train"].shuffle(seed=42).select(range(1000))

    # Format dataset for dish + ingredients
    def format_example(example):

        dish_name = example.get("dish_name", "Unknown Dish")
        ingredients = example["ingredients"]

        # JSON structured output
        output_text = (
            "{\n"
            f'  "dish": "{dish_name}",\n'
            '  "ingredients": [\n'
            + ",\n".join([f'    "{ing}"' for ing in ingredients])
            + "\n  ]\n}"
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text",
                     "text": "Identify the dish and list all visible ingredients in JSON format."}
                ]
            },
            {
                "role": "assistant",
                "content": output_text
            }
        ]

        return {
            "image": example["image_url"],
            "messages": messages
        }

    print("Formatting dataset...")
    train_dataset = train_dataset.map(format_example)

    print("Applying LoRA...")

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules="all-linear",
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()


    def collate_fn(batch):
        texts = []
        images = []
        prompt_lengths = []

        for example in batch:
            # 1. Prepare the full conversation for training
            text = processor.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False
            )
            texts.append(text)
            images.append(example["image_url"])

            # 2. Prepare ONLY the prompt to measure its length
            prompt_only = processor.apply_chat_template(
                [example["messages"][0]], # Just the user message
                tokenize=False,
                add_generation_prompt=True # Adds the hidden "Assistant:" trigger
            )
            
            # Tokenize just the prompt to see how many tokens it takes
            prompt_tokens = processor(
                text=prompt_only, 
                images=example["image_url"], 
                return_tensors="pt"
            )
            prompt_lengths.append(prompt_tokens["input_ids"].shape[1])

        # Tokenize the whole batch together
        inputs = processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True
        )

        # 3. Create labels and apply the -100 mask
        labels = inputs["input_ids"].clone()
        
        for i, prompt_len in enumerate(prompt_lengths):
            # Mask out the prompt so the model doesn't learn to generate it
            labels[i, :prompt_len] = -100 
            
            # Mask out any padding tokens added to make batch sizes equal
            labels[i, inputs["attention_mask"][i] == 0] = -100

        inputs["labels"] = labels
        return inputs

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        num_train_epochs=args.epochs,
        fp16=True,
        learning_rate=args.lr,
        logging_steps=10,
        save_steps=500,
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=True, 
        optim="paged_adamw_8bit"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn
    )

    print("Starting training...")
    trainer.train()

    print("Saving model...")
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()