# model.py

import torch.nn as nn
from transformers import AutoModel

class TripletLossModel(nn.Module):
    def __init__(self, model_name):
        super(TripletLossModel, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.loss_fn = nn.TripletMarginLoss(margin=1.0, p=2)

    def forward(self, anchor, positive, negative):
        anchor_emb = self.model(**anchor).last_hidden_state[:, 0, :]
        positive_emb = self.model(**positive).last_hidden_state[:, 0, :]
        negative_emb = self.model(**negative).last_hidden_state[:, 0, :]
        loss = self.loss_fn(anchor_emb, positive_emb, negative_emb)
        return loss
