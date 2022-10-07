from cgitb import text
from transformers import BertModel
import torch
from torch import nn


class SentenceBertClassifier(nn.Module):
    def __init__(self, device=None):
        super().__init__()

        # SentenceBERT
        model_name_or_path = "sonoisa/sentence-bert-base-ja-mean-tokens"
        self.sentence_BERT = BertModel.from_pretrained("sonoisa/sentence-bert-base-ja-mean-tokens")

        # classifier
        # input: BERT output 768*3 dim, output: 1 dim
        self.cls = nn.Linear(in_features=2304, out_features=1)

        # 重み初期化処理
        nn.init.normal_(self.cls.weight, std=0.02)
        nn.init.normal_(self.cls.bias, 0)

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, text_ids, former_ids, latter_ids):
        text_attn_mask = torch.where(text_ids > 0, 1, 0)
        former_attn_mask = torch.where(former_ids > 0, 1, 0)
        latter_attn_mask = torch.where(latter_ids > 0, 1, 0)

        text_output = self.sentence_BERT(text_ids)[0]
        text_embeddings = self._mean_pooling(text_output, text_attn_mask)

        former_output = self.sentence_BERT(text_ids)[0]
        former_text_embeddings = self._mean_pooling(former_output, former_attn_mask)
        latter_output = self.sentence_BERT(text_ids)[0]
        latter_text_embeddings = self._mean_pooling(latter_output, latter_attn_mask)
        context_embeddings = (former_text_embeddings + latter_text_embeddings) / 2 

        combined_embeddings = torch.abs(text_embeddings - context_embeddings) # |u-v| SentenceBERTの論文によると精度が良いらしい

        output = self.cls(torch.cat((text_embeddings,context_embeddings, combined_embeddings), dim=1))

        return output