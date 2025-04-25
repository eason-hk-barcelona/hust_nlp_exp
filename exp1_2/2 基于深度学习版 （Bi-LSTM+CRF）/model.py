import torch
import torch.nn as nn
from torchcrf import CRF
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class CWS(nn.Module):
    def __init__(self, vocab_size, tag2id, embedding_dim, hidden_dim, dropout=0.3, num_attention_heads=8):
        super(CWS, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag2id = tag2id
        self.tagset_size = len(tag2id)

        # 词嵌入层
        self.word_embeds = nn.Embedding(vocab_size + 1, embedding_dim)
        
        # Dropout正则化
        self.dropout = nn.Dropout(dropout)

        # 双向LSTM层
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=2,
                            bidirectional=True, batch_first=True, dropout=dropout)
        
        # 添加多头自注意力层
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_attention_heads, 
            dropout=dropout
        )
        
        # 添加层归一化
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # 输出层
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # CRF层
        self.crf = CRF(4, batch_first=True)

    def init_hidden(self, batch_size, device):
        """初始化LSTM隐藏状态"""
        return (torch.randn(4, batch_size, self.hidden_dim // 2, device=device),  # 4=num_layers*num_directions
                torch.randn(4, batch_size, self.hidden_dim // 2, device=device))

    def _get_lstm_features(self, sentence, length):
        batch_size, seq_len = sentence.size(0), sentence.size(1)

        # 1. 词嵌入处理
        embeds = self.word_embeds(sentence.view(-1)).reshape(batch_size, seq_len, -1)
        embeds = self.dropout(embeds)  
        embeds = pack_padded_sequence(embeds, length, batch_first=True)

        # 2. LSTM编码
        self.hidden = self.init_hidden(batch_size, sentence.device)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        
        # 3. 应用自注意力机制
        # 转换格式为(seq_len, batch, hidden_dim)以适应MultiheadAttention
        lstm_out_t = lstm_out.transpose(0, 1)
        
        # 创建注意力掩码(1表示保留，0表示掩码)
        attn_mask = torch.zeros(batch_size, seq_len, device=sentence.device)
        for i in range(batch_size):
            attn_mask[i, :length[i]] = 1
        attn_mask = attn_mask.bool()
        
        # 应用多头自注意力
        attn_output, _ = self.attention(
            query=lstm_out_t,
            key=lstm_out_t,
            value=lstm_out_t,
            key_padding_mask=~attn_mask  # ~操作反转mask，MultiheadAttention中True表示需要掩码的位置
        )
        
        # 转回(batch, seq_len, hidden_dim)格式
        attn_output = attn_output.transpose(0, 1)
        
        # 残差连接和层归一化
        lstm_out = self.layer_norm(lstm_out + attn_output)
        
        # 4. 应用Dropout
        lstm_out = self.dropout(lstm_out)
        
        # 5. 输出映射到标签空间
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def forward(self, sentence, tags, mask, length):
        """前向传播计算损失，用于训练"""
        emissions = self._get_lstm_features(sentence, length)
        loss = -self.crf(emissions, tags, mask, reduction='mean')
        return loss

    def infer(self, sentence, mask, length):
        """推理方法，返回最优标签序列"""
        emissions = self._get_lstm_features(sentence, length)
        return self.crf.decode(emissions, mask)