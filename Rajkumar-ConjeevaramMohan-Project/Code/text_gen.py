from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForSeq2Seq
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSeq2SeqLM
from transformers import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
from datasets import load_metric
import re
import requests
from bs4 import BeautifulSoup as BS
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

class TextSummarizationDataset(Dataset):
    def __init__(self, df, tokenizer, max_input_len, max_output_len):
        self.df = df
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.iloc[idx]['article']
        summary = self.df.iloc[idx]['highlights']

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
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )

        return{
            'input_ids': input_encoding['input_ids'].squeeze(),
            # 'input_attention_mask': input_encoding['attention_mask'].squeeze(),
            'decoder_input_ids': output_encoding['input_ids'].squeeze(),
            # 'output_attention_mask': output_encoding['attention_mask'].squeeze()
        }

class TextSummarization:
    def __init__(self):
        self.num_epochs = 3
        MAX_INPUT_LEN = 512
        MAX_OUTPUT_LEN = 128
        self.CHECKPOINT = 't5-small'
        # self.CHECKPOINT = "BART"
        self.BATCH_SIZE = BATCH_SIZE = 10
        PREFETCH_FACTOR = 100
        N_WORKERS = 10
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        ts_df = pd.read_csv("data/test.csv")
        tr_df = pd.read_csv("data/train.csv")
        tr_df, vl_df, _, _ = train_test_split(tr_df, tr_df.highlights, test_size=0.3)
        self.tr_size = tr_df.shape[0]
        self.vl_size = vl_df.shape[0]

        # raw_datasets = load_dataset("glue", "mrpc")
        self.tokenizer = AutoTokenizer.from_pretrained(self.CHECKPOINT)
        tr_tsdataset = TextSummarizationDataset(tr_df, self.tokenizer, MAX_INPUT_LEN, MAX_OUTPUT_LEN)
        vl_tsdataset = TextSummarizationDataset(vl_df, self.tokenizer, MAX_INPUT_LEN, MAX_OUTPUT_LEN)
        ts_tsdataset = TextSummarizationDataset(ts_df, self.tokenizer, MAX_INPUT_LEN, MAX_OUTPUT_LEN)

        # Sample of data for program testing - Andy's snippet of code----------------------------
        sample_idx_train = [i for i in range(500)]
        sample_idx_eval = [i for i in range(250)]
        tr_tsdataset = torch.utils.data.Subset(tr_tsdataset, sample_idx_train)
        vl_tsdataset = torch.utils.data.Subset(vl_tsdataset, sample_idx_eval)
        ts_tsdataset = torch.utils.data.Subset(ts_tsdataset, sample_idx_eval)
        #-----------------------------------------------------------------------------------------

        # tokenized_datasets = tokenize_function(df)
        data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, model=self.CHECKPOINT)

        self.tr_dataloader = DataLoader(tr_tsdataset, shuffle=True, batch_size=BATCH_SIZE,
                                   collate_fn=data_collator, num_workers=N_WORKERS,
                                   prefetch_factor=PREFETCH_FACTOR)
        self.vl_dataloader = DataLoader(vl_tsdataset, batch_size=BATCH_SIZE,
                                   collate_fn=data_collator, num_workers=N_WORKERS,
                                   prefetch_factor=PREFETCH_FACTOR)
        self.ts_dataloader = DataLoader(ts_tsdataset, batch_size=BATCH_SIZE,
                                   collate_fn=data_collator, num_workers=N_WORKERS,
                                   prefetch_factor=PREFETCH_FACTOR)

    def compute_metrics(self, eval_pred):
        # Load the rouge metric
        predictions, labels = eval_pred
        print(labels)
        decoded_pred = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        result = self.rouge.compute(predictions=decoded_pred, references=decoded_labels, use_stemmer=True)

        prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in predictions]
        result['gen_len'] = np.mean(prediction_lens)

        # Access the mid F1 score for each rouge metric (rouge1, rouge2, and rougeL)
        return {k: round(v, 4) for k, v in result.items()}

    def build_and_fit(self):
        model = AutoModelForSeq2SeqLM.from_pretrained(self.CHECKPOINT).to(self.device)
        optimizer = AdamW(model.parameters(), lr=1e-3)

        num_training_steps = self.num_epochs * len(self.tr_dataloader)
        lr_scheduler = get_scheduler("linear", optimizer=optimizer,
                                     num_warmup_steps=0, num_training_steps=num_training_steps)
        print(num_training_steps)

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model.to(device)

        # progress_bar = tqdm(range(num_training_steps))
        criterion = torch.nn.CrossEntropyLoss().to(device)

        steps_train = self.tr_size//self.BATCH_SIZE
        steps_val = self.vl_size//self.BATCH_SIZE

        for epoch in range(self.num_epochs):
            model.train()
            loss_train = 0
            with tqdm(total=self.tr_size, desc=f"Epoch {epoch}") as progress_bar:
                for batch in self.tr_dataloader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    outputs = model(**batch)
                    tmp_rouge = self.compute_metrics((outputs, batch['decoder_input_ids']))
                    loss = criterion(torch.permute(outputs['logits'], [0, 2, 1]), batch['decoder_input_ids'])
                    loss.backward()
                    loss_train += loss.item()
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(self.BATCH_SIZE)
                    progress_bar.set_postfix_str("Train Loss: {:.5f}".format(loss_train/steps_train))

                # metric = load_metric("glue", "mrpc")
            model.eval()
            loss_val = 0
            with tqdm(total=self.vl_size, desc="Epoch {}".format(epoch)) as progress_bar:
                for batch in self.vl_dataloader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    with torch.no_grad():
                        outputs = model(**batch)

                    logits = outputs.logits
                    predictions = torch.argmax(logits, dim=-1)
                    loss = criterion(torch.permute(outputs['logits'], [0, 2, 1]), batch['decoder_input_ids'])
                    loss_val += loss.item()
                    progress_bar.update(self.BATCH_SIZE)
                    progress_bar.set_postfix_str("Test Loss: {:.5f}".format(loss_val / steps_val))

                train_loss_batch = round(loss_train / self.BATCH_SIZE, 4)
                val_loss_batch = round(loss_val / self.BATCH_SIZE, 4)
                print(f"Epoch {epoch + 1} | Train Loss {train_loss_batch}, - Validation Loss {val_loss_batch}")

    def test(self, trainer):
        # *****
        print("\nbegin testing...\n")
        # ***

        summaries = []
        # test_dataloader = DataLoader(self.test_dataset, batch_size=1, shuffle=False)
        # for i, batch in enumerate(self.test_dataset):
        #########################################################
        # For program testing
        test_dataloader = DataLoader(self.test_sample_dataset, batch_size=1, shuffle=False)
        # for i, batch in enumerate(self.test_dataloader):
        #########################################################

        for i, text in enumerate(test_dataloader):
            actual_summary = self.tokenizer.decode(text['labels'].detach().cpu().tolist()[0],
                                                   skip_special_tokens=True)

            input = text['input_ids'].to(self.device)
            output = self.model.generate(input, max_new_tokens=self.MAX_OUTPUT_LEN, do_sample=False)
            decoded_output = self.tokenizer.decode(output[0], skip_special_tokens=True)
            # summaries.append(decoded_output)

            summaries.append((actual_summary, decoded_output))

            if i % 10 == 0:
                print(f"{i}/{len(test_dataloader)} summarizations completed...")

        return summaries

if __name__ == "__main__":
    ts = TextSummarization()
    ts.build_and_fit()
    print("Checkpoint...")