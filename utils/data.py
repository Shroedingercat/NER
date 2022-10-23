import torch
from tqdm import tqdm
from typing import Dict, List
from torch.utils.data import Dataset
from transformers import AutoTokenizer


def read_data(file: str) -> (Dict[str, list], Dict[str, int]):
    lines = open(file, "r").readlines()
    data = {"sentences": [], "labels_per_sent": []}
    sentence, labels = [], []
    labels_to_id = {}
    count = 0
    for line in tqdm(lines):
        line = line.strip()
        if not line:
            if sentence and labels:
                assert len(sentence) == len(labels)
                data["sentences"].append(sentence)
                data["labels_per_sent"].append(labels)
                sentence, labels = [], []
            continue
        if line.startswith("-DOCSTART-"):
            continue
        else:
            values = line.split(" ")
            try:
                token, _, _, label = values
                sentence.append(token)

                if label != 'O':
                    labels.append(label.split('-')[-1])
                else:
                    labels.append(label)

            except Exception as e:
                print(f"Error has occur: {e}")
                continue

    for item in data["labels_per_sent"]:
        for label in item:
            if labels_to_id.get(label) is None:
                labels_to_id[label] = count
                count += 1
    return data, labels_to_id


def align_label_example(tokenized_input: Dict[str, torch.Tensor],
                        labels: List[str],
                        labels_to_ids: Dict[str, int]) -> (List[int], List[int]):
    word_ids = tokenized_input.word_ids()

    previous_word_idx = None
    label_ids = []
    tokens = []

    for word_idx in word_ids:
        if word_idx is None:
            label_ids.append(-100)
            tokens.append(0)
        elif word_idx != previous_word_idx:
            label_ids.append(labels_to_ids[labels[word_idx]])
            tokens.append(1)
        else:
            label_ids.append(-100)
            tokens.append(0)
        previous_word_idx = word_idx

    assert len(word_ids) == len(tokens)

    return label_ids, tokens


class NERDataset(Dataset):
    def __init__(self, data: Dict[str, list],
                 labels_to_id: Dict[str, int],
                 tokenizer: AutoTokenizer,
                 max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.labels_to_id = labels_to_id
        self.max_length = max_length

    def __getitem__(self, idx):
        text = self.data["sentences"][idx]
        labels = self.data["labels_per_sent"][idx]
        text_tokenized = self.tokenizer.encode_plus(text,
                                                    max_length=self.max_length, return_tensors="pt",
                                                    truncation=True, padding='max_length',
                                                    is_split_into_words=True)
        label_ids, tokens = align_label_example(text_tokenized,
                                        labels, self.labels_to_id)

        return {"labels": torch.Tensor(label_ids).long(),
                "input_ids": text_tokenized['input_ids'].squeeze(),
                "attention_mask": text_tokenized['attention_mask'].squeeze(),
                "tokens":  torch.Tensor(tokens).long()}

    def __len__(self):
        return len(self.data["sentences"])
