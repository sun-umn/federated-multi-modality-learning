# federated-multi-modality-learning


## <a herf="https://www.kaggle.com/datasets/rajnathpatel/ner-data"> NER dataset from Kaggle </a>
> code inherite from <a href="https://towardsdatascience.com/named-entity-recognition-with-bert-in-pytorch-a454405e0b6a"> this blog</a>

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

___
## <a href="https://huggingface.co/bert-large-cased">Model</a>


>
bert-base-cased
In BERT uncased, the text has been lowercased before WordPiece tokenization step while in BERT cased, the text is same as the input text (no changes).

<span style="color:green">bert-large-cased</span>

___

## Evaluation Metric
> take from <a href="https://github.com/chakki-works/seqeval">seqeval</a>
> 
> :warning: seqeval supports the two evaluation modes. You can specify the following mode to each metrics: **default**, **strict** :warning:

___
### main metrics
***precision***
$$
precision = \frac{TP}{TP + FP}
$$

***recall***
$$
recall = \frac{TP}{TP + FN}
$$

***f1-score***

$$
F_1 = \frac{2 precision\times recall}{precision + recall}
$$

___

### <a href="https://datascience.stackexchange.com/questions/36862/macro-or-micro-average-for-imbalanced-class-problems">aggregation schemes</a>

***micro average***

average samples (e.g. accuracy) to maximize the number of correct predictions the classifier makes

***macro average***

average the metric (e.g. balanced accuracy) <span style="color:green">suggests</span> 

***weighted average***

each classes’s contribution to the average is weighted by its size, lies in between micro and maroc average


___
*support count the number of sampling in each token*



## Experiment Setup

___
### Federated Learning with NVFlare
- algorithm: fedavg
- random splits into two clients
    > client 1: train size 15346, val size 1918
    > 
    > client 2: train size 15347, val size 1918
- learning rate $5\times10^{-3}$
- batch size 8
- epoches 20 (set larger as fedavg coverge slower than pooled training)
- aggregation weights: uniform (1:1)
___

### Single Client Learning
same as FL but with 10 epoches
___

## Preliminary Results
### site 1
**training**

|   token      | precision | recall | f1-score | support |
|--------------|------------|---------|----------|---------|
|         art   |    0.20   |   0.02  |    0.04  |     188 |
|         eve   |    0.39   |   0.08  |    0.13  |     111 |
|         geo   |    0.77   |   0.78  |    0.78  |   14495 |
|         gpe   |    0.89   |   0.85  |    0.87  |    5483 |
|         nat   |    0.40   |   0.02  |    0.04  |      94 |
|         org   |    0.64   |   0.60  |    0.62  |    8294 |
|         per   |    0.68   |   0.72  |    0.70  |    7136 |
|         tim   |    0.77   |   0.69  |    0.73  |    6793 |
|   micro avg   |    0.74   |   0.72  |    0.73  |   42594 |
|   macro avg   |    0.59   |   0.47  |    0.49  |   42594 |
|weighted avg   |    0.74   |   0.72  |    0.73  |   42594 |


### site 2
**training**

|   token      | precision | recall | f1-score | support |
|--------------|------------|---------|----------|---------|
|         art  |     0.48   |   0.14  |    0.22  |     186 |
|         eve  |     0.21   |   0.12  |    0.15  |     119 |
|         geo  |     0.80   |   0.82  |    0.81  |   14566 |
|         gpe  |     0.90   |   0.86  |    0.88  |    5444 |
|         nat  |     0.25   |   0.01  |    0.02  |     102 |
|         org  |     0.67   |   0.64  |    0.66  |    8239 |
|         per  |     0.70   |   0.74  |    0.72  |    6927 |
|         tim  |     0.79   |   0.73  |    0.76  |    6755 |
|   micro avg  |     0.77   |   0.76  |    0.76  |   42338 |
|   macro avg  |     0.60   |   0.51  |    0.53  |   42338 |
|weighted avg  |     0.77   |   0.76  |    0.76  |   42338 |

___

## TODO
- [x] aggregate B/I 
- [ ] exactly match (strict)
- [x] lenient match (loose)
- [x] seqeval
- [x] try bert large


## pending for GPU setup
> <a href="https://docs.google.com/document/d/1ugYeOUtJZtWraL7Zjk0D3axNl40wOb3-tsjb3HFaiLQ/edit">Link to the google doc</a>
- [ ] setup nvflare on ahc-ie gpu server
    - [ ] install pytorch container
    - [ ] setup nvflare environment
    - [ ] mount data to gpu server

- [ ] deploy nvflare model
    - [ ] package the pytorch container
    - [ ] streamline the data preprocessing pipeline
    - [ ] network test
    - [ ] run real-world FL
