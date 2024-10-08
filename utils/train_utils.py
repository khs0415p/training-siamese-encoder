import os
import torch

from typing import List, Dict
from collections import OrderedDict
from transformers import PreTrainedModel, AutoTokenizer
from utils import LOGGER, seed_worker
from utils.data_utils import CustomDataset
from torch.utils.data import DataLoader, random_split, distributed, WeightedRandomSampler, SequentialSampler


IGNORE_ID = -100


def get_pretrained_weights(model: PreTrainedModel, pretrained_model: PreTrainedModel) -> Dict[str, torch.Tensor]:
    # shared_layer_names =  set(model.state_dict().keys().intersection(old_model.state_dict().keys()))
    LOGGER.info("Change the weights of the model to the base model.")
    layers = []
    for k, v in model.state_dict().items():
        if 'embeddings' in k or 'predictions' in k or 'vocab_projector' in k:
            layers.append((k, v))
            continue
        
        try:
            pretrained_layer = pretrained_model.state_dict()[k]
        except:
            raise ValueError("model and pretrained-model are different.")
        layers.append((k, pretrained_layer))
        
    return OrderedDict(layers)


def collate_fn_warpper(padding_id, model_type):
    def collate_fn_inner(batch):
        return collate_fn(batch, padding_id, model_type)
    return collate_fn_inner


def collate_fn(batch: List[Dict[str, torch.Tensor]], padding_value: int = 0, model_type: str = '') -> Dict[str, torch.Tensor]:
    dataset_keys = ['premise', 'hypothesis']
    if any([model_name in model_type for model_name in ['roberta', 'distilbert']]):
        input_keys = ("input_ids", "attention_mask")
    else:
        input_keys = ("input_ids", "token_type_ids", "attention_mask")

    def pad_and_create_mask(input_ids: List[torch.Tensor], attention_mask: List[torch.Tensor], padding_value: int) -> Dict[str, torch.Tensor]:
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=padding_value)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=padding_value)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }

    outputs = {}
    for key in dataset_keys:
        collected_data = {item_key: [instance[key][item_key] for instance in batch] for item_key in input_keys}
        result = pad_and_create_mask(collected_data["input_ids"], collected_data["attention_mask"], padding_value)
        if "token_type_ids" in collected_data:
            result["token_type_ids"] = torch.nn.utils.rnn.pad_sequence(
                collected_data['token_type_ids'], batch_first=True, padding_value=padding_value
            )
        
        outputs[key] = result
    
    labels = [instance['labels'] for instance in batch]
    outputs['labels'] = torch.stack(labels, dim=0)
    return outputs


def build_dataloader(dataset, batch_size, num_workers, shuffle, model_type, ddp=False, pad_token_id=0):
    weights = (1 / dataset.data['label'].map(dataset.data['label'].value_counts(normalize=True))).values

    sampler = distributed.DistributedSampler(dataset, shuffle=shuffle) if ddp else WeightedRandomSampler(weights=weights, num_samples=len(dataset)) if shuffle else SequentialSampler(dataset)
    return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=sampler,
            pin_memory=True,
            shuffle=shuffle if sampler is None else False,
            collate_fn=collate_fn_warpper(pad_token_id, model_type),
            worker_init_fn=seed_worker
            )


def get_dataset(config, modes, tokenizer=None):
    if modes == 'test':
        return CustomDataset(config, config.test_data_path, tokenizer)

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path, cache_dir=config.cache_dir, trust_remote_code=True)
    datasets = [CustomDataset(config, config.train_data_path, tokenizer), CustomDataset(config, config.valid_data_path, tokenizer)]
    return {mode:ds for mode, ds in zip(modes, datasets)}, tokenizer


def get_dataloader(config):
    """
    Returns:
        (Dict[phase: DataLoader]): dataloader for training
    Examples:
        {'train': DataLoader, 'valid': DataLoader}
    """
    n_gpu = torch.cuda.device_count()
    n_cpu = os.cpu_count()
    num_workers = min([4 * n_gpu, config.batch_size // n_gpu, config.batch_size // n_cpu])  # number of workers
    modes = ['train', 'valid']

    dict_dataset, tokenizer = get_dataset(config, modes)

    dataloader = {mode: build_dataloader(dict_dataset[mode], config.batch_size, num_workers, mode == 'train', config.model_type, config.ddp, tokenizer.pad_token_id) for mode in modes}

    return dataloader, tokenizer

def get_test_dataloader(config, tokenizer):
    n_gpu = torch.cuda.device_count()
    n_cpu = os.cpu_count()
    num_workers = min([4 * n_gpu, config.batch_size // n_gpu, config.batch_size // n_cpu])  # number of workers

    dataset = get_dataset(config, 'test', tokenizer)
    return build_dataloader(dataset, config.batch_size, num_workers, True, config.model_type, config.ddp, tokenizer.pad_token_id)