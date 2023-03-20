
from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch.nn as nn

class BertModel(nn.Module):

    def __init__(self):
        super(BertModel, self).__init__()
        print("asdfasdfasdfa-==============================-0=-0=-0=-0=-0=00000000000000")
        num_labels=19
        self.model_name='bert-base-uncased'
        self.bert = AutoModelForTokenClassification.from_pretrained(self.model_name, num_labels=num_labels, \
                    output_attentions = False, \
                    output_hidden_states = False)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
    def forward(self, input_id, mask, label):
        output = self.bert(input_ids=input_id, attention_mask=mask, labels=label, return_dict=False)
        return output