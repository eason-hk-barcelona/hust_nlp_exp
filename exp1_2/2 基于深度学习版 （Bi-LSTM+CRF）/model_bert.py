import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel
from torchcrf import CRF

class BertCWS(nn.Module):
    def __init__(self, tag2id, dropout=0.1):
        super(BertCWS, self).__init__()
        self.tag2id = tag2id
        self.tagset_size = len(tag2id)
        
        # Load pre-trained BERT model
        self.bert = BertModel.from_pretrained('hfl/chinese-roberta-wwm-ext')
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Hidden to tag mapping
        self.hidden2tag = nn.Linear(self.bert.config.hidden_size, self.tagset_size)
        
        # CRF layer
        self.crf = CRF(self.tagset_size, batch_first=True)
        
    def _get_bert_features(self, input_ids, attention_mask):
        """Extract features from BERT model"""
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        sequence_output = self.dropout(sequence_output)
        emissions = self.hidden2tag(sequence_output)  # [batch_size, seq_len, tagset_size]
        return emissions
    
    def forward(self, input_ids, attention_mask, tags, valid_tags_mask=None):
        """Forward pass for training"""
        emissions = self._get_bert_features(input_ids, attention_mask)
        
        # 创建有效标签掩码，排除标签值为-100的位置
        if valid_tags_mask is None:
            valid_tags_mask = tags >= 0  # 排除-100值
        
        # 合并注意力掩码和有效标签掩码
        mask = attention_mask & valid_tags_mask
        
        # 只将第一个有效token位置设为1
        valid_lengths = attention_mask.sum(dim=1)
        batch_size = mask.size(0)
        for i in range(batch_size):
            # 将第一个有效位置设为1，可能不是位置0
            for j in range(mask.size(1)):
                if attention_mask[i, j]:
                    mask[i, j] = True
                    break
        
        # 处理CRF输入的标签，将-100替换为0，CRF会根据mask忽略这些位置
        crf_tags = tags.clone()
        crf_tags[crf_tags < 0] = 0
        
        loss = -self.crf(emissions, crf_tags, mask.byte(), reduction='mean')
        return loss
    
    def infer(self, input_ids, attention_mask):
        """Inference method, returns best tag sequence"""
        emissions = self._get_bert_features(input_ids, attention_mask)
        
        # 确保掩码的第一个时间步为1，与训练保持一致
        mask = attention_mask.clone()
        batch_size = mask.size(0)
        mask[:, 0] = torch.ones(batch_size, dtype=torch.bool, device=mask.device)
        
        return self.crf.decode(emissions, mask.byte())