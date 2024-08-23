import torch
import torch.nn as nn

from transformers import AutoModel

class SiameseEncoder(nn.Module):
    def __init__(self, encoder_dim=768, dropout=0.1, pooling='eos'):
        super().__init__()
        # self.config = config
        self.mean = pooling
        self.encoder = AutoModel.from_pretrained('google-bert/bert-base-uncased', cache_dir='/data')
        self.layer = nn.Linear(encoder_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
        self.pooling = pooling

    def _pooling(self, batched_tensor, attention_mask, pooling='mean'):
        lengths = attention_mask.sum(dim=-1)
        if pooling == 'mean':
            batched_tensor = (batched_tensor * attention_mask.unsqueeze(-1).expand_as(batched_tensor)).sum(dim=1)
            pooled_output = batched_tensor / lengths.unsqueeze(1)

        elif pooling == 'bos':
            pooled_output = batched_tensor[:, 0, :]

        elif pooling == 'eos':
            indices = (lengths - 1).unsqueeze(1).unsqueeze(2).expand(-1, 1, batched_tensor.size(2))
            pooled_output = torch.gather(batched_tensor, 1, indices).squeeze(1)

        else:
            raise ValueError("Please, Select pooling type among ['mean', 'bos', 'eos'].")
        
        return pooled_output

    def forward(self, document1, document2):
        output_1 = self.encoder(**document1)
        output_2 = self.encoder(**document2)

        pooled_output_1 = self._pooling(output_1['last_hidden_state'], document1['attention_mask'], self.pooling)   # B, D
        pooled_output_2 = self._pooling(output_2['last_hidden_state'], document2['attention_mask'], self.pooling)   # B, D

        output = self.layer((pooled_output_1 + pooled_output_2)) # B, 1
        output = self.sigmoid(self.dropout(output))

        return output

