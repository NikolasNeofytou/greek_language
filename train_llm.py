import argparse
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          DataCollatorForLanguageModeling, Trainer,
                          TrainingArguments)


def main(output_dir: str,
         dataset_name: str = "oscar-corpus/OSCAR-2301",
         dataset_config: str = "el",
         split: str = "train[:1%]") -> None:
    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained("gpt2")

    dataset = load_dataset(dataset_name, dataset_config, split=split)

    def tokenize_function(examples):
        return tokenizer(examples["text"])

    tokenized_datasets = dataset.map(
        tokenize_function, remove_columns=["text"], batched=True
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        logging_steps=10,
        save_steps=1000,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        data_collator=data_collator,
    )

    trainer.train()

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="greek-llm")
    parser.add_argument("--dataset_name", default="oscar-corpus/OSCAR-2301")
    parser.add_argument("--dataset_config", default="el")
    parser.add_argument("--split", default="train[:1%]")
    args = parser.parse_args()
    main(args.output_dir, args.dataset_name, args.dataset_config, args.split)
