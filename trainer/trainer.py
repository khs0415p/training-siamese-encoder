import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseTrainer
from utils import LOGGER
from utils.train_utils import get_dataloader, get_test_dataloader
from transformers import AutoTokenizer



class Trainer(BaseTrainer):
    def __init__(self, config, device) -> None:
        super().__init__(config, device)

        # dataloaders
        if config.mode == "train":
            self.dataloader, self.tokenizer = get_dataloader(config) # {'train': dataloader, 'valid': dataloader}
            self.config.vocab_size = len(self.tokenizer) 

        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.checkpoint, trust_remote_code=True, cache_dir=self.config.cache_dir)
            self.dataloader = get_test_dataloader(config, self.tokenizer)

        # acc history
        self.valid_acc_history = []

        # main process
        self.rank_zero = True if not self.ddp or (self.ddp and device == 0) else False

        # initialize trainer
        self._init_trainer()

        # classification objective
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)

        # regression objective
        self.cos_sim = nn.CosineSimilarity()
        self.mse_loss = nn.MSELoss()

        # triplet objective
        self.triplet_loss = nn.TripletMarginLoss(margin=self.config.margin)
    
    def _make_target(self, labels):
        targets = torch.ones_like(labels.squeeze(), dtype=torch.float, device=labels.device)
        targets[labels.squeeze() == 1] = -1

        return targets
    
    def _get_triplet_loss(self, premise, hypothesis, labels):
        positive_make = labels == 0
        negative_make = labels == 1

        positive_indices = positive_make.nonzero(as_tuple=True)[0]
        negative_indices = negative_make.nonzero(as_tuple=True)[0]

        if len(positive_indices) == 0 or len(negative_indices) == 0:
            LOGGER.info("Not enough positive or negative samples found.")
            return None
        
        num_samples = min(len(positive_indices), len(negative_indices))

        positive_indices = positive_indices[torch.randperm(len(positive_indices))][:num_samples]
        negative_indices = negative_indices[torch.randperm(len(negative_indices))][:num_samples]

        anchors = premise[positive_indices]
        positives = hypothesis[positive_indices]
        negatives = hypothesis[negative_indices]

        loss = self.triplet_loss(anchors, positives, negatives)
        return loss

    def _training_step(self, model_inputs):
        """
        Args:
            model_inputs: data of batch
        Return:
            (Tensor loss): loss
        """        
        labels = model_inputs['labels']

        output = self.model(**model_inputs, return_logit=True)
        (premise_output, hypothesis_output), logit = output['pooled_outputs'], output['logit']

        classification_loss = self.cross_entropy(logit.view(-1, self.config.num_labels), labels.view(-1))

        cosine_similarity = self.cos_sim(premise_output, hypothesis_output)
        targets = self._make_target(labels)
        cosine_loss = self.mse_loss(cosine_similarity, targets)

        triplet_loss = self._get_triplet_loss(premise_output, hypothesis_output, labels)

        if triplet_loss is None:
            loss = (classification_loss + cosine_loss) / 2
        else:
            loss = (classification_loss + cosine_loss + triplet_loss) / 3

        self._backward_step(loss)

        return loss.item(), classification_loss.item(), cosine_loss.item(), None if triplet_loss is None else triplet_loss.item()


    @torch.no_grad()
    def _validation_step(self, model_inputs):
        """
        Args:
            model_inputs: data of batch
        Return:
            (Tensor loss): loss
        """        
        labels = model_inputs['labels']

        output = self.model(**model_inputs, return_logit=True)
        (premise_output, hypothesis_output), logit = output['pooled_outputs'], output['logit']

        classification_loss = self.cross_entropy(logit.view(-1, self.config.num_labels), labels.view(-1))

        cosine_similarity = self.cos_sim(premise_output, hypothesis_output)
        targets = self._make_target(labels)
        cosine_loss = self.mse_loss(cosine_similarity, targets)

        triplet_loss = self._get_triplet_loss(premise_output, hypothesis_output, labels)

        if triplet_loss is None:
            loss = (classification_loss + cosine_loss) / 2
        else:
            loss = (classification_loss + cosine_loss + triplet_loss) / 3

        outputs = torch.argmax(logit.detach().cpu(), dim=-1)
        acc = torch.sum(outputs == labels.detach().cpu().squeeze()) / logit.size(0)
    
        return loss.item(), classification_loss.item(), cosine_loss.item(), None if triplet_loss is None else triplet_loss.item(), acc.item()

    @torch.no_grad()
    def _test_step(self, model_inputs):
        labels = model_inputs['labels']

        output = self.model(**model_inputs, return_logit=True)
        pred = torch.argmax(output['logit'].detach().cpu(), dim=-1)
        label = labels.detach().cpu().squeeze()

        return pred.tolist(), label.tolist()