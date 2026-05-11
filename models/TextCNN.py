import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification
from models.Model_base import MyModel


class Model(MyModel):
    def __init__(self, config):
        super(Model, self).__init__()
        num_labels = getattr(config, 'num_classes', 4)
        local_bert_path = "/root/autodl-tmp/FedUP_code/FedUP/models/bert-tiny-local"

        self.model = AutoModelForSequenceClassification.from_pretrained(
            local_bert_path,
            num_labels=num_labels,
            attn_implementation="eager"
        )


    def forward(self, *args, **kwargs):

        if len(args) == 1 and len(kwargs) == 0 and isinstance(args[0], dict):
            input_ids = args[0]['input_ids']
            attention_mask = args[0]['attention_mask']
        
        elif len(args) == 2 and len(kwargs) == 0:
            input_ids, attention_mask = args
        
        elif len(args) == 0 and 'input_ids' in kwargs and 'attention_mask' in kwargs:
            input_ids = kwargs['input_ids']
            attention_mask = kwargs['attention_mask']
        
        else:
            raise ValueError(
                "Unsupported input format. Please use one of:\n"
                "1. model({'input_ids': ..., 'attention_mask': ...})\n"
                "2. model(input_ids, attention_mask)\n"
                "3. model(input_ids=..., attention_mask=...)"
            )

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits
    
    def get_features(self, input_ids=None, attention_mask=None, x=None):

        if x is not None:
            input_ids = x['input_ids']
            attention_mask = x['attention_mask']
        elif input_ids is None or attention_mask is None:
            raise ValueError("Must provide input_ids and attention_mask")

        with torch.no_grad():
            encoder = self.model.bert
            encoder_outputs = encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            pooled_features = encoder.pooler(encoder_outputs.last_hidden_state) 


        return pooled_features
