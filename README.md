# NER
NER with transformer encoder model. 


Metric token level F1.


Start training
```bash
sh train.sh
```
Start inference
```bash
sh inference.sh
```


| architecture | F1    |
|-----------|-------|
| bert-base-cased | 0.920 | 
| roberta-base | 0.937 |
| microsoft/deberta-v3-base | 0.960 |