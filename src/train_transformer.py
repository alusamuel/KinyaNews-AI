"""Optional real Transformer fine-tuning for Colab/GPU; compare validation macro F1.

This is an experiment and is NOT the Vercel deployment model unless explicitly
converted into a size-compatible inference artifact. Results must be measured.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def train(model_name="xlm-roberta-base", epochs=2, max_length=256):
    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
    train_df = pd.read_csv("data/train.csv").dropna()
    val_df = pd.read_csv("data/val.csv").dropna()
    labels = sorted(train_df.label.astype(str).unique())
    id_map = {label: idx for idx, label in enumerate(labels)}
    def convert(frame):
        return Dataset.from_dict({"text": frame.text.astype(str).tolist(), "labels": [id_map[x] for x in frame.label.astype(str)]})
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)
    ds_train = convert(train_df).map(tokenize, batched=True)
    ds_val = convert(val_df).map(tokenize, batched=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(labels), id2label={i:l for i,l in enumerate(labels)}, label2id=id_map)
    args = TrainingArguments(output_dir="models/transformer", num_train_epochs=epochs, per_device_train_batch_size=8, per_device_eval_batch_size=16, learning_rate=2e-5, weight_decay=0.01, eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True, metric_for_best_model="macro_f1", greater_is_better=True, report_to="none", seed=42, fp16=torch.cuda.is_available(), save_total_limit=1)
    def metrics(p):
        preds=np.argmax(p.predictions,axis=-1)
        return {"accuracy": accuracy_score(p.label_ids,preds), "macro_f1":f1_score(p.label_ids,preds,average="macro",zero_division=0)}
    trainer=Trainer(model=model,args=args,train_dataset=ds_train,eval_dataset=ds_val,processing_class=tokenizer,data_collator=DataCollatorWithPadding(tokenizer),compute_metrics=metrics)
    trainer.train()
    final=trainer.evaluate()
    trainer.save_model("models/transformer/final")
    tokenizer.save_pretrained("models/transformer/final")
    Path("results").mkdir(exist_ok=True)
    Path("results/transformer_validation.json").write_text(json.dumps({"model":model_name,"epochs":epochs,"max_length":max_length,"metrics":final},indent=2))
    print(json.dumps(final,indent=2))

if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--model",default="xlm-roberta-base")
    p.add_argument("--epochs",type=int,default=2)
    p.add_argument("--max-length",type=int,default=256)
    a=p.parse_args()
    train(a.model,a.epochs,a.max_length)
