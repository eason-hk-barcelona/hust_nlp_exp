import torch
import pickle
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast

class BertSentence(Dataset):
    def __init__(self, x, y, id2word, tag2id, max_length=512):
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
        self.x = x
        self.y = y
        self.id2word = id2word
        self.tag2id = tag2id
        self.max_length = max_length

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        # 还原原始句子
        text = "".join([self.id2word[i] for i in self.x[idx]])
        tags = self.y[idx]

        # 分词
        encoding = self.tokenizer(
            list(text),  # 按字分
            is_split_into_words=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        word_ids = encoding.word_ids(batch_index=0)

        # 标签对齐 - 改进版
        labels = torch.ones(len(word_ids), dtype=torch.long) * -100  # 使用-100作为忽略索引
        tag_idx = 0
        for i, word_id in enumerate(word_ids):
            # 跳过特殊标记和None值
            if word_id is None or word_id >= len(tags):
                continue
            
            # 只为每个原始字符的第一个token分配标签
            if i > 0 and word_id == word_ids[i-1]:
                continue
                
            if tag_idx < len(tags):
                labels[i] = tags[tag_idx]
                tag_idx += 1
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
    
    @staticmethod
    def collate_fn(batch):
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


if __name__ == '__main__':
    # Test
    with open('data/datasave.pkl', 'rb') as inp:
        word2id = pickle.load(inp)
        id2word = pickle.load(inp)
        tag2id = pickle.load(inp)
        id2tag = pickle.load(inp)
        x_train = pickle.load(inp)
        y_train = pickle.load(inp)
        x_test = pickle.load(inp)
        y_test = pickle.load(inp)
    
    # Convert character indices to actual characters for BERT tokenizer
    for i in range(5):
        text = "".join([id2word[j] for j in x_train[i]])
        tags = [id2tag[j] for j in y_train[i]]
        print(f"Text: {text}")
        print(f"Tags: {tags}")
        
    train_dataset = BertSentence(x_train[:100], y_train[:100], id2word, tag2id, max_length=512)
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=BertSentence.collate_fn)
    
    for batch in train_dataloader:
        print(f"Input shape: {batch['input_ids'].shape}")
        print(f"Attention mask shape: {batch['attention_mask'].shape}")
        print(f"Labels shape: {batch['labels'].shape}")
        break 