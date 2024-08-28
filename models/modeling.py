import torch
import torch.nn as nn

from transformers import AutoModel


class SiameseEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.mean = config.pooling
        self.encoder = AutoModel.from_pretrained(config.model_path, cache_dir=config.cache_dir)
        self.lm_head = nn.Linear(self.encoder.config.hidden_size * 3, self.config.num_labels)
        self.pooling = config.pooling

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

    def forward(self, premise, hypothesis, labels=None, return_logit=True):
        output_1 = self.encoder(**premise)
        output_2 = self.encoder(**hypothesis)

        pooled_output_1 = self._pooling(output_1['last_hidden_state'], premise['attention_mask'], self.config.pooling)   # B, D
        pooled_output_2 = self._pooling(output_2['last_hidden_state'], hypothesis['attention_mask'], self.config.pooling)   # B, D

        output = torch.concat([pooled_output_1, pooled_output_2, (pooled_output_1 - pooled_output_2).abs()], dim=-1)

        if return_logit:
            logit = self.lm_head(output)

            return {
                "pooled_outputs" : [pooled_output_1, pooled_output_2],
                "logit": logit
                }

        return {
            "pooled_outputs" : [pooled_output_1, pooled_output_2],
        }