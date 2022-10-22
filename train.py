import os
import sys
import torch
import random
import numpy as np

from tqdm import tqdm
from typing import List, Dict
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from utils.data import NERDataset, read_data
from sklearn.metrics import classification_report, f1_score
from transformers import AutoModelForTokenClassification, AutoTokenizer, AdamW


def parse_arguments():
    parser = ArgumentParser(__doc__)
    parser.add_argument("--train_data_file", help="path to dataset"),
    parser.add_argument("--eval_data_file", help="path to dataset"),
    parser.add_argument("--output_dir", help="path to dataset"),
    parser.add_argument("--model_name_or_path", help="model")
    parser.add_argument("--model_type", help="model")
    parser.add_argument("--max_length", type=int)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def count_metrics(gold_labels_per_sentence: List[List[str]],
                  predict_labels_per_sentence: List[List[str]]):
    gold_labels = [l for l_per_sent in gold_labels_per_sentence for l in l_per_sent]
    predict_labels = [l for l_per_sent in predict_labels_per_sentence for l in l_per_sent]
    print(classification_report(gold_labels, predict_labels))


def transform_logits(predictions: torch.tensor,
                     id_to_labels: Dict[int, str]):
    return [[id_to_labels[ids] for ids in item] for item in predictions]


def transform_target(target_labels: torch.tensor,
                     id_to_labels: Dict[int, str]) -> List[List[str]]:
    return [[id_to_labels[ids] for ids in item] for item in target_labels]


def train(args):
    set_seed(args.seed)

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    if args.model_type == "bert":
        model = AutoModelForTokenClassification.from_pretrained(args.model_name_or_path,
                                                                num_labels=5)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    elif args.model_type == "roberta":
        model = AutoModelForTokenClassification.from_pretrained(args.model_name_or_path,
                                                                num_labels=5)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, add_prefix_space=True)
    else:
        raise NotImplemented

    model.to(args.device)

    train, labels_to_id = read_data(args.train_data_file)
    valid, _ = read_data(args.eval_data_file)
    id_to_labels = {labels_to_id[key]: key for key in labels_to_id}

    train_dataset = NERDataset(train, labels_to_id, tokenizer, args.max_length)
    valid_dataset = NERDataset(valid, labels_to_id, tokenizer, args.max_length)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True)

    losses = {"train_losses": [], "valid_losses": []}

    best_score = -float("inf")

    epochs = args.epochs
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    for epoch in range(epochs):

        print(f"Start eposh #{epoch}")
        model.train()
        for train_batch in tqdm(train_dataloader,
                                total=len(train_dataloader)):
            train_batch = {key: train_batch[key].to(args.device) for key in train_batch}
            out = model.forward(**train_batch)
            loss = out.loss
            logits = out.logits
            losses["train_losses"].append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        predict_labels, gold_labels = [], []
        model.eval()

        for valid_batch in valid_dataloader:
            with torch.no_grad():
                valid_batch = {key: valid_batch[key].to(args.device) for key in valid_batch}
                out = model.forward(**valid_batch)
                loss = out.loss
                logits = out.logits
                losses["valid_losses"].append(loss.item())
                logits = logits.argmax(dim=2).cpu()
                pred = [logits[i][valid_batch["labels"][i] != -100].tolist()
                        for i in range(valid_batch["labels"].shape[0])]
                valid_batch["labels"] = valid_batch["labels"].cpu()
                targets = [valid_batch["labels"][i][valid_batch["labels"][i] != -100].tolist()
                           for i in range(valid_batch["labels"].shape[0])]
                gold_labels += transform_target(targets,
                                                id_to_labels)
                predict_labels += transform_logits(pred,
                                                   id_to_labels)

        score = f1_score(np.array(predict_labels) == np.array(gold_labels),
                       [1]*len(predict_labels))

        if score > best_score:
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)

        print("Token level f1 score:", score)

        print(f"End eposh #{epoch}")


if __name__ == "__main__":
    args = parse_arguments()
    train(args)