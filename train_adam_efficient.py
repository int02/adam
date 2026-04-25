#!/usr/bin/env python3
"""
Cost-Effective ADAM Training Script
Fine-tunes smaller models for jailbreaking using LoRA
Integrates with existing ADAM system
"""

import asyncio
import json
import os
import sys
from pathlib import Path
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
from sklearn.model_selection import train_test_split
from datetime import datetime

# Cost-effective model options (free/low-cost)
COST_EFFECTIVE_MODELS = {
    "phi-2": "microsoft/phi-2",  # 2.7B params, good for jailbreaking
    "gemma-2b": "google/gemma-2b",  # 2B params, efficient
    "mistral-7b": "mistralai/Mistral-7B-v0.1",  # 7B but good compression
    "qwen-1.5b": "Qwen/Qwen1.5-1.8B",  # Very small but capable
    "qwen-0.5b": "Qwen/Qwen1.5-0.5B",  # Extremely small for testing
}


class ADAMTrainer:
    def __init__(self):
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        self.datasets_dir = Path("datasets")
        self.datasets_dir.mkdir(exist_ok=True)

    def collect_training_data(self):
        """Collect all available training data from datasets/ folders"""
        all_texts = []

        # Check advmodel folder
        advmodel_dir = self.datasets_dir / "advmodel"
        if advmodel_dir.exists():
            for json_file in advmodel_dir.glob("*.json"):
                try:
                    with open(json_file, "r") as f:
                        data = json.load(f)
                        for item in data:
                            if "text" in item:
                                # Weight by score if available
                                score = item.get("score", 5)
                                weight = max(1, int(score))
                                all_texts.extend([item["text"]] * weight)
                except Exception as e:
                    print(f"Error loading {json_file}: {e}")

        # Check evalmodel folder
        evalmodel_dir = self.datasets_dir / "evalmodel"
        if evalmodel_dir.exists():
            for json_file in evalmodel_dir.glob("*.json"):
                try:
                    with open(json_file, "r") as f:
                        data = json.load(f)
                        for item in data:
                            if "text" in item:
                                all_texts.append(item["text"])
                except Exception as e:
                    print(f"Error loading {json_file}: {e}")

        # Also check training_data from the main system
        training_data_dir = Path("training_data")
        if training_data_dir.exists():
            for json_file in training_data_dir.glob("**/*.json"):
                try:
                    with open(json_file, "r") as f:
                        data = json.load(f)
                        for item in data:
                            if "text" in item:
                                score = item.get("score", 5)
                                weight = max(1, int(score))
                                all_texts.extend([item["text"]] * weight)
                except Exception as e:
                    print(f"Error loading {json_file}: {e}")

        return all_texts

    async def train_adam_model(
        self,
        model_name="phi-2",
        output_name="adam_jailbreak",
        epochs=2,
        lora_r=16,
        lora_alpha=32,
        batch_size=2,
        use_quantization=True,
    ):
        """Train ADAM using LoRA for cost-effectiveness"""

        print(f"🚀 Training ADAM with {model_name} using LoRA...")
        print(f"📊 Collecting training data...")

        # Collect all available training data
        all_texts = self.collect_training_data()

        if len(all_texts) < 10:
            print("❌ Not enough training data. Need at least 10 samples.")
            print("💡 Add data to datasets/advmodel/ or datasets/evalmodel/ folders")
            print("💡 Or run pentests to generate training data")
            return {"status": "error", "error": "Insufficient training data"}

        print(f"📈 Found {len(all_texts)} training samples")

        # Split data
        train_texts, val_texts = train_test_split(
            all_texts, test_size=0.1, random_state=42
        )
        print(
            f"🎯 Training on {len(train_texts)} samples, validating on {len(val_texts)}"
        )

        # Use cost-effective model
        base_model = COST_EFFECTIVE_MODELS.get(model_name, model_name)

        # Load tokenizer
        print(f"🔧 Loading tokenizer for {base_model}...")
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Tokenize data
        def tokenize_function(examples):
            return tokenizer(
                examples["text"], truncation=True, padding="max_length", max_length=512
            )

        print("🔧 Tokenizing data...")
        train_dataset = Dataset.from_dict({"text": train_texts})
        val_dataset = Dataset.from_dict({"text": val_texts})

        train_dataset = train_dataset.map(tokenize_function, batched=True)
        val_dataset = val_dataset.map(tokenize_function, batched=True)

        # Model loading config
        model_kwargs = {}
        if use_quantization:
            print("💾 Using 4-bit quantization for memory efficiency...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["quantization_config"] = bnb_config
            model_kwargs["device_map"] = "auto"

        # Load model
        print(f"🧠 Loading {base_model} model...")
        model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)

        # Prepare for LoRA training
        print("🔧 Setting up LoRA fine-tuning...")
        model = prepare_model_for_kbit_training(model) if use_quantization else model

        # LoRA config (cost-effective fine-tuning)
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Attention layers
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, lora_config)
        print(f"🎯 Trainable parameters: {model.print_trainable_parameters()}")

        # Training arguments (cost-optimized)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        training_args = TrainingArguments(
            output_dir=str(self.models_dir / f"{output_name}_{timestamp}"),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=4,  # Effective batch size = batch_size * 4
            learning_rate=2e-4,
            weight_decay=0.01,
            logging_steps=10,
            evaluation_strategy="steps",
            eval_steps=50,
            save_steps=100,
            save_total_limit=2,
            load_best_model_at_end=True,
            fp16=not use_quantization,  # Mixed precision when not using quantization
            push_to_hub=False,
            report_to="none",  # No external logging
        )

        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
        )

        # Train
        print("🎯 Starting cost-effective LoRA training...")
        print("💡 This may take a while depending on your hardware...")
        trainer.train()

        # Save LoRA adapters
        output_dir = self.models_dir / f"{output_name}_lora_{timestamp}"
        output_dir.mkdir(exist_ok=True)

        print(f"💾 Saving trained ADAM model to {output_dir}...")
        model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))

        # Save training info
        info_file = output_dir / "training_info.json"
        training_info = {
            "base_model": base_model,
            "model_name": model_name,
            "training_samples": len(train_texts),
            "validation_samples": len(val_texts),
            "epochs": epochs,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "batch_size": batch_size,
            "quantization": use_quantization,
            "trained_at": timestamp,
            "cost_estimate": f"${(len(train_texts) * epochs * 0.0001):.4f} (rough estimate)",
        }

        with open(info_file, "w") as f:
            json.dump(training_info, f, indent=2)

        print(f"✅ ADAM trained successfully!")
        print(f"📁 Model saved to: {output_dir}")
        print(f"💡 Cost estimate: {training_info['cost_estimate']}")
        print(f"🔧 To use this model, update Config.ATTACKER_MODEL in core.py")

        return {
            "status": "success",
            "model_path": str(output_dir),
            "training_info": training_info,
        }


async def main():
    trainer = ADAMTrainer()

    # Parse command line arguments
    model_choice = sys.argv[1] if len(sys.argv) > 1 else "phi-2"
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    output_name = sys.argv[3] if len(sys.argv) > 3 else "adam_jailbreak"

    print("🎯 ADAM Cost-Effective Training Launcher")
    print("=" * 50)
    print(f"Model: {model_choice}")
    print(f"Epochs: {epochs}")
    print(f"Output: {output_name}")
    print("=" * 50)

    result = await trainer.train_adam_model(
        model_name=model_choice, output_name=output_name, epochs=epochs
    )

    if result["status"] == "success":
        print("\n🎉 Training completed successfully!")
        print(f"Model path: {result['model_path']}")
        print("\nTo use this model in ADAM:")
        print("1. Open core.py")
        print(f"2. Change Config.ATTACKER_MODEL to: '{result['model_path']}'")
        print("3. Restart the server")
    else:
        print(f"\n❌ Training failed: {result.get('error', 'Unknown error')}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
