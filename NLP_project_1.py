import numpy as np
import pandas as pd
import random
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments, DataCollatorForSeq2Seq)
from datasets import load_metric
import evaluate
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# read individual csv files
train_df = pd.read_csv('train.csv')
val_df = pd.read_csv('validation.csv')
test_df = pd.read_csv('test.csv')

# combine the dataframe
collated_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

# rename two column names
df = collated_df.rename(columns={"article": "source", "highlights": "target"})
df = df[['source', 'target']]

# splitting the collated df
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

class TextSummarizationDataset(Dataset):
    def __init__(self, df, tokenizer, max_input_len, max_output_len):
        self.df = df
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.iloc[idx]['source']
        summary = self.df.iloc[idx]['target']

        input_encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_input_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        output_encoding = self.tokenizer.encode_plus(
            summary,
            add_special_tokens=True,
            max_length=self.max_output_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )

        return{
            'input_ids': input_encoding['input_ids'].flatten(),
            'input_attention_mask': input_encoding['attention_mask'].flatten(),
            'labels': output_encoding['input_ids'].flatten(),
            'output_attention_mask': output_encoding['attention_mask'].flatten()
        }


# Model and training parameters
CHECKPOINT = "t5-small"
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
N_EPOCH = 1
BATCH_SIZE = 16
PREFETCH_FACTOR = 2
N_WORKER = 4
GRADIENT_ACCUMULATION_STEPS = 2
MAX_INPUT_LEN = 512
MAX_OUTPUT_LEN = 128

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)

# Create instances of the custom dataset
train_dataset = TextSummarizationDataset(train_df, tokenizer, MAX_INPUT_LEN, MAX_OUTPUT_LEN)
test_dataset = TextSummarizationDataset(test_df, tokenizer, MAX_INPUT_LEN, MAX_OUTPUT_LEN)

# Create DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, prefetch_factor=PREFETCH_FACTOR,
                              num_workers=N_WORKER)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, prefetch_factor=PREFETCH_FACTOR,
                             num_workers=N_WORKER)

# Create a batch of examples using DataCollatorForSeq2Seq
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=CHECKPOINT)

# Load the rouge metric
#rouge = load_metric('rouge')

# Define compute_metrics function
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_pred = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = evaluate.load('rouge').compute(predictions=decoded_pred, references=decoded_labels, use_stemmer=True)

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result['gen_len'] = np.mean(prediction_lens)

    # Access the mid F1 score for each rouge metric (rouge1, rouge2, and rougeL)
    return {k: round(v, 4) for k, v in result.items()}


model = AutoModelForSeq2SeqLM.from_pretrained(CHECKPOINT).to(device)

train_args = Seq2SeqTrainingArguments(
    output_dir='my_best_model',
    evaluation_strategy='epoch',
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    weight_decay=WEIGHT_DECAY,
    save_total_limit=3,
    num_train_epochs=N_EPOCH,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    predict_with_generate=True,
    fp16=True,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=train_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)
trainer.train()

# inference
source_texts = df['source'].tolist()
random_source_text = random.choice(source_texts)
print("Random source: ", random_source_text)

inputs = tokenizer(random_source_text, return_tensors='pt').input_ids.to(device)
outputs = model.generate(inputs, max_new_tokens=100, do_sample=False)

decoded_outputs = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Generated summary: ", decoded_outputs)
