# federated-multi-modality-learning


## <a herf = "https://www.kaggle.com/datasets/rajnathpatel/ner-data"> NER dataset from Kaggle </a>
**code inherite from: https://towardsdatascience.com/named-entity-recognition-with-bert-in-pytorch-a454405e0b6a**

| Token | Description| Count (B) | Count (I) |
| ------| :-----------:| :----: | :---: |
|    geo | geographical entity| 37644 |7414 |
|    org | organization entity| 20143 | 16784|
|    per | person entity|16990 |17251 |
|    gpe | geopolitical entity|15870 | 198|
|    tim | time indicator entity| 20333| 6528|
|    art | artifact entity| 402| 297|
|    eve | event entity| 308| 253|
|    nat | natural phenomenon entity|201 |51 |
|    O | assigned if a word doesn’t belong to any entity.| 887908| |

## Model
>
bert-base-cased
In BERT uncased, the text has been lowercased before WordPiece tokenization step while in BERT cased, the text is same as the input text (no changes).
>

## TODO
- [ ] aggregate B/I 
- [ ] exactly match (strict)
- [ ] lenient match (loose)
- [ ] seqeval
- [ ] try bert large
