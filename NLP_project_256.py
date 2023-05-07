#%%

import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import numpy as np
import pandas as pd
import random
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments, DataCollatorForSeq2Seq)
# from datasets import load_metric
import evaluate
import torch

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

class TextSummarization:
    def __init__(self, checkpoint="t5-small"):
        self.RANDOM_STATE = 42
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.DATA_DIR = "D:/GWU/DATS-6312/project/data" # change to match local path
        self.DATA_DIR = "data"
        self.rouge = evaluate.load('rouge')
        self.CHECKPOINT = checkpoint
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.CHECKPOINT).to(self.device)
        
        """--------------------------------------------------------------------------------
        HyperParameters for Model
        --------------------------------------------------------------------------------"""
        self.N_EPOCH = 60
        self.BATCH_SIZE = 30
        self.LR = 1e-3
        PREFETCH_FACTOR = 30
        self.N_WORKERS = 10

        self.GRADIENT_ACCUMULATION_STEPS = 1
        self.MAX_INPUT_LEN = 256
        self.MAX_OUTPUT_LEN = 128

        """--------------------------------------------------------------------------------
        Data Loading
        --------------------------------------------------------------------------------"""
        #*****
        print("\nload data...")
        #***
        
        # read individual csv files
        train_df = pd.read_csv(f'{self.DATA_DIR}/train.csv')
        val_df = pd.read_csv(f'{self.DATA_DIR}/validation.csv')
        test_df = pd.read_csv(f'{self.DATA_DIR}/test.csv')
        
        """--------------------------------------------------------------------------------
        Remove unwanted columns
        --------------------------------------------------------------------------------"""
        # rename two column names
        train_df = train_df.rename(columns={"article": "source", "highlights": "target"})
        train_df = train_df[['source', 'target']]

        # rename two column names
        val_df = val_df.rename(columns={"article": "source", "highlights": "target"})
        val_df = val_df[['source', 'target']]

        # rename two column names
        test_df = test_df.rename(columns={"article": "source", "highlights": "target"})
        test_df = test_df[['source', 'target']]
        
        #*****
        print("\ndata frames created...")
        #*****
        
        """--------------------------------------------------------------------------------
        Torch Dataset
        --------------------------------------------------------------------------------"""
        # Initialize the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.CHECKPOINT)

        # Create a batch of examples using DataCollatorForSeq2Seq
        self.data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, model=self.CHECKPOINT)

        # Create instances of the custom dataset
        self.train_dataset = TextSummarizationDataset(train_df, self.tokenizer,
                                                 self.MAX_INPUT_LEN, self.MAX_OUTPUT_LEN)
        self.val_dataset = TextSummarizationDataset(val_df, self.tokenizer,
                                               self.MAX_INPUT_LEN, self.MAX_OUTPUT_LEN)
        self.test_dataset = TextSummarizationDataset(test_df, self.tokenizer,
                                                self.MAX_INPUT_LEN, self.MAX_OUTPUT_LEN)
        
        #########################################################
        
        # Sample of data for program testing
        sample_idx_train = [i for i in range(1000)]
        sample_idx_eval = [i for i in range(250)]
        self.train_sample_dataset = torch.utils.data.Subset(self.train_dataset, sample_idx_train)
        self.val_sample_dataset = torch.utils.data.Subset(self.val_dataset, sample_idx_eval)
        self.test_sample_dataset = torch.utils.data.Subset(self.test_dataset, sample_idx_eval)
        
        #*****
        print("\ndata sets created...")
        #*****
        
        #########################################################
        
        """--------------------------------------------------------------------------------
        Evaluation Metric
        --------------------------------------------------------------------------------"""
        

    # Define compute_metrics function
    def compute_metrics(self, eval_pred):
        # Load the rouge metric
        predictions, labels = eval_pred
        decoded_pred = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
    
        result = self.rouge.compute(predictions=decoded_pred, references=decoded_labels, use_stemmer=True)
    
        prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in predictions]
        result['gen_len'] = np.mean(prediction_lens)
    
        # Access the mid F1 score for each rouge metric (rouge1, rouge2, and rougeL)
        return {k: round(v, 4) for k, v in result.items()}

    def build(self):
        
        # model = AutoModelForSeq2SeqLM.from_pretrained(self.CHECKPOINT).to(self.device)
        # Model and training parameters
        train_args = Seq2SeqTrainingArguments(
            output_dir='my_best_model',
            overwrite_output_dir=False,
            evaluation_strategy='epoch',
            learning_rate=self.LR,
            per_device_train_batch_size=self.BATCH_SIZE,
            per_device_eval_batch_size=self.BATCH_SIZE,
            # weight_decay=self.WEIGHT_DECAY,
            save_total_limit=3,
            num_train_epochs=self.N_EPOCH,
            gradient_accumulation_steps=self.GRADIENT_ACCUMULATION_STEPS,
            predict_with_generate=True,
            fp16=True,
            do_predict=True,
            # lr_scheduler_type='linear',
            half_precision_backend='auto',
            bf16_full_eval=False,
            dataloader_drop_last=True,
            dataloader_num_workers=self.N_WORKERS,
            # include_inputs_for_metrics=True,
        )

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=train_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            #########################################################
            # For program testing
            # train_dataset=self.train_sample_dataset,
            # eval_dataset=self.val_sample_dataset,
            #########################################################
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
        )

        #*****
        print("\nmodel and training sequence created...")
        #***
        
        return trainer
    
    def fit(self, trainer):
        #*****
        print("\nbegin training...")
        #***
        trainer.train()
        torch.save(self.model.state_dict(), "last_epoch.pt")

    def test(self, trainer):
        #*****
        print("\nbegin testing...\n")
        #***
        
        summaries = []
        # test_dataloader = DataLoader(self.test_dataset, batch_size=1, shuffle=False)
        # for i, batch in enumerate(self.test_dataset):
        #########################################################
        # For program testing
        test_dataloader = DataLoader(self.test_sample_dataset, batch_size=1, shuffle=False)
        #for i, batch in enumerate(self.test_dataloader):
        #########################################################
            
        for i,text in enumerate(test_dataloader):            
            actual_summary = self.tokenizer.decode(text['labels'].detach().cpu().tolist()[0], skip_special_tokens=True)
            # actual_summary = list(filter(lambda x: x!= "<pad>", actual_summary))
            # actual_summary = " ".join(actual_summary)
            # print(f"Actual summary of sample {i}:\n{actual_summary}")
        
            # pred_summary = trainer.predict(batch)
            # pred_summary = self.tokenizer.batch_decode(pred_summary)
            # pred_summary = list(filter(lambda x: x!= "<pad>", pred_summary))
            # pred_summary = " ".join(pred_summary)
            # print(f"Predicted summary of sample {i}:\n{pred_summary}")
            
            input = text['input_ids'].to(self.device)
            output = self.model.generate(input, max_new_tokens=self.MAX_OUTPUT_LEN, do_sample=False)
            decoded_output = self.tokenizer.decode(output[0], skip_special_tokens=True)
            # summaries.append(decoded_output)
            
            summaries.append((actual_summary, decoded_output))

            if i % 10 == 0:
                print(f"{i}/{len(test_dataloader)} summarizations completed...")
            
        return summaries


if __name__ == "__main__":
    # checkpoint = "my_best_model/checkpoint-6500"
    checkpoint = "t5-small"
    ts = TextSummarization(checkpoint=checkpoint)
    trainer = ts.build()
    ts.fit(trainer)
    summaries = ts.test(trainer)

    # View results
    print("\nExample of First 10 Summaries Produced Compared to Original Highlights from Data\n")
    for i in range(10):
        print(f"{i}:\n\tOriginal Summary: {summaries[i][0]}\n\tProduced Summary: {summaries[i][1]}")




# %%
