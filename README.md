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
recall=\frac{TP}{TP + FN}
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
**test performance**

|   token      | precision | recall | f1-score | support |
|--------------|------------|---------|----------|---------|
|         art  |     0.00   |   0.00  |    0.00  |      17 |
|         eve  |     0.00   |   0.00  |    0.00  |      18 |
|         geo  |     0.68   |   0.78  |    0.73  |    1733 |
|         gpe  |     0.80   |   0.81  |    0.81  |     716 |
|         nat  |     0.00   |   0.00  |    0.00  |      17 |
|         org  |     0.61   |   0.51  |    0.56  |    1031 |
|         per  |     0.65   |   0.72  |    0.68  |     883 |
|         tim  |     0.72   |   0.66  |    0.69  |     866 |
|   micro avg  |     0.69   |   0.69  |    0.69  |    5281 |
|   macro avg  |     0.43   |   0.43  |    0.43  |    5281 |
|weighted avg  |     0.68   |   0.69  |    0.68  |    5281 |


### site 2
**test performance**

|   token      | precision | recall | f1-score | support |
|--------------|------------|---------|----------|---------|
|         art  |     0.00   |   0.00  |    0.00  |      31 |
|         eve  |     0.00   |   0.00  |    0.00  |      18 |
|         geo  |     0.81   |   0.69  |    0.75  |    1715 |
|         gpe  |     0.85   |   0.83  |    0.84  |     734 |
|         nat  |     0.00   |   0.00  |    0.00  |      15 |
|         org  |     0.53   |   0.64  |    0.58  |     997 |
|         per  |     0.67   |   0.70  |    0.68  |     951 |
|         tim  |     0.76   |   0.71  |    0.74  |     850 |
|   micro avg  |     0.71   |   0.70  |    0.71  |    5311 |
|   macro avg  |     0.45   |   0.45  |    0.45  |    5311 |
|weighted avg  |     0.72   |   0.70  |    0.71  |    5311 |





### FedAvg

coming soon...