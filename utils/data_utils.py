import json
import torch
import pandas as pd

from typing import List
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, config, data_path, tokenizer):
        self.config = config

        self.max_length = config.max_length
        self.tokenizer = tokenizer
        self.use_token_type_ids = all([model_name not in self.config.model_type for model_name in ['roberta', 'distilbert']])

        self.data = pd.read_parquet(data_path)


    def make_data(self, sentence: str):
        output = self.tokenizer(
            sentence,
            max_length=self.max_length,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )

        return output

    def __getitem__(self, index):
        row = self.data.iloc[index]

        tokenized_premise = self.make_data(row['premise'])
        tokenized_hypothesis = self.make_data(row['hypothesis'])

        return {
            "premise" : {
                "input_ids" : tokenized_premise.input_ids.squeeze(),
                "token_type_ids": tokenized_premise.token_type_ids.squeeze() if self.use_token_type_ids else None,
                "attention_mask" : tokenized_premise.attention_mask.squeeze(),
            },
            "hypothesis" : {
                "input_ids" : tokenized_hypothesis.input_ids.squeeze(),
                "token_type_ids": tokenized_hypothesis.token_type_ids.squeeze() if self.use_token_type_ids else None,
                "attention_mask" : tokenized_hypothesis.attention_mask.squeeze(),
            },
            "labels": torch.LongTensor([row['label']])
        }


    def __len__(self):
        return len(self.data)