
### collate (manga batch)
import torch
def collate_fn(batch):
  return {
 'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
 'labels': torch.tensor([x['labels'] for x in batch])
 }

import numpy as np
from datasets import load_metric
metric = load_metric("accuracy")
def compute_metrics(p):
 return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

###
from transformers import ViTForImageClassification
labels =ds['train'].features['labels'].names
model =ViTForImageClassification.from_pretrained(
 model_name_or_path,
 num_labels=len(labels),
 id2label={str(i): c for i, c in enumerate(labels)},
 label2id={c: str(i) for i, c in enumerate(labels)}
)

### tain
from transformers import TrainingArguments
training_args =TrainingArguments(
 output_dir="./vit-base-beans-demo-v5",
 per_device_train_batch_size=16,
 evaluation_strategy="steps",
 num_train_epochs=4,
 fp16=True,
 save_steps=100,
 eval_steps=100,
 logging_steps=10,
 learning_rate=2e-4,
 save_total_limit=2,
 remove_unused_columns=False,
 push_to_hub=False,
 report_to='tensorboard',
 load_best_model_at_end=True,
)
### eval
from transformers import Trainer
trainer =Trainer(
 model=model,
 args=training_args,
 data_collator=collate_fn,
 compute_metrics=compute_metrics,
 train_dataset=prepared_ds["train"],
 eval_dataset=prepared_ds["validation"],
 tokenizer=feature_extractor,
)


train_results =trainer.train()
trainer.save_model()
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
trainer.save_state()
metrics =trainer.evaluate(prepared_ds['validation'])
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)
