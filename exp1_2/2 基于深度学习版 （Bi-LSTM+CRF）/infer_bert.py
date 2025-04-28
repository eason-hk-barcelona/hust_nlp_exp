import torch
import pickle
import argparse
from transformers import BertTokenizerFast


def batch_inference(model, tokenizer, texts, id2tag, batch_size=16):
    results = []
    
    # 批量处理文本
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        # 批量编码
        encodings = tokenizer(
            [list(text) for text in batch_texts],
            is_split_into_words=True,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encodings['input_ids']
        attention_mask = encodings['attention_mask']
        
        with torch.no_grad():
            # 批量预测
            predictions = model.infer(input_ids, attention_mask)
        
        # 处理每个文本的预测结果
        for b, text in enumerate(batch_texts):
            word_ids = encodings.word_ids(batch_index=b)
            pred_tags = predictions[b]
            
            segmented_text = ""
            previous_word_id = None
            
            for i, word_id in enumerate(word_ids):
                if word_id is None or i == 0:  # 跳过[CLS]
                    continue
                if i >= len(pred_tags):
                    break
                    
                # 只处理每个字的第一个token
                if word_id == previous_word_id:
                    continue
                    
                previous_word_id = word_id
                char = text[word_id]
                segmented_text += char
                if id2tag[pred_tags[i]] in ['E', 'S']:
                    segmented_text += ' '
            
            results.append(segmented_text)
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='save/best_bert_model.pkl')
    args = parser.parse_args()

    # Load the saved model
    print(f"Loading model: {args.model_path}")
    model = torch.load(args.model_path, map_location=torch.device('cpu'))
    model.eval()

    # Load the BERT tokenizer
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')

    # Open output file for writing results
    output = open('cws_result_bert.txt', 'w', encoding='utf-8')

    # Load the tag mappings
    with open('data/datasave.pkl', 'rb') as inp:
        word2id = pickle.load(inp)
        id2word = pickle.load(inp)
        tag2id = pickle.load(inp)
        id2tag = pickle.load(inp)
        x_train = pickle.load(inp)
        y_train = pickle.load(inp)
        x_test = pickle.load(inp)
        y_test = pickle.load(inp)

    # Process each line in the test file
    all_texts = []
    with open('data/test_final.txt', 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                all_texts.append(line)

    # 批量推理
    batch_size = 32  # 根据GPU内存调整
    segmented_texts = batch_inference(model, tokenizer, all_texts, id2tag, batch_size)

    # 写入结果
    with open('cws_result_bert.txt', 'w', encoding='utf-8') as output:
        for text in segmented_texts:
            output.write(text + '\n')

    print("Segmentation results saved to cws_result_bert.txt")