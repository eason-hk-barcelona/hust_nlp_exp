import pickle
import logging
import argparse
import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from model_bert import BertCWS
from dataloader_bert import BertSentence

def get_param():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=2e-5)  # Lower learning rate for BERT
    parser.add_argument('--max_epoch', type=int, default=5)  # Fewer epochs for BERT fine-tuning
    parser.add_argument('--batch_size', type=int, default=64)  # Smaller batch size for BERT
    parser.add_argument('--max_length', type=int, default=512)  # Max sequence length for BERT
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.01)  # Weight decay for AdamW
    parser.add_argument('--early_stopping', type=int, default=2)  # Early stopping criteria
    parser.add_argument('--warmup_steps', type=int, default=500)  # Warmup steps for lr scheduler
    parser.add_argument('--cuda', action='store_true', default=False)
    return parser.parse_args()


def set_logger():
    os.makedirs('save', exist_ok=True)
    log_file = os.path.join('save', 'bert_log.txt')
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.DEBUG,
        datefmt='%Y-%m%d %H:%M:%S',
        filename=log_file,
        filemode='w',
    )

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def entity_split(x, y, id2tag, entities, cur):
    start, end = -1, -1
    for j in range(len(x)):
        if id2tag[y[j]] == 'B':
            start = cur + j
        elif id2tag[y[j]] == 'M' and start != -1:
            continue
        elif id2tag[y[j]] == 'E' and start != -1:
            end = cur + j
            entities.add((start, end))
            start, end = -1, -1
        elif id2tag[y[j]] == 'S':
            entities.add((cur + j, cur + j))
            start, end = -1, -1
        else:
            start, end = -1, -1


def evaluate_model(model, test_data, id2tag, use_cuda=False):
    entity_predict = set()
    entity_label = set()
    with torch.no_grad():
        model.eval()
        cur = 0
        for batch in test_data:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']
            
            if use_cuda:
                input_ids = input_ids.cuda()
                attention_mask = attention_mask.cuda()
                labels = labels.cuda()
            
            # 获取模型预测
            predict = model.infer(input_ids, attention_mask)
            
            # 处理每个序列
            batch_size = input_ids.size(0)
            for i in range(batch_size):
                # 获取实际字符序列长度（排除特殊标记和填充）
                sequence_tokens = []
                sequence_pred_tags = []
                sequence_gold_tags = []
                
                # 提取原始字符及其对应标签
                word_ids = test_data.dataset.tokenizer.convert_ids_to_tokens(input_ids[i])
                
                for j in range(1, len(word_ids)):  # 跳过[CLS]
                    if attention_mask[i][j] == 0:  # 跳过填充标记
                        break
                    if word_ids[j] in ['[SEP]', '[PAD]']:
                        continue
                        
                    # 检查是否是特殊子词标记（以##开头）
                    if j > 0 and word_ids[j].startswith('##'):
                        continue
                        
                    sequence_tokens.append(input_ids[i][j].item())
                    sequence_pred_tags.append(predict[i][j])
                    
                    # 只收集有效标签（非-100）
                    if labels[i][j] >= 0:
                        sequence_gold_tags.append(labels[i][j].item())
                
                # 构建实体集合
                if len(sequence_tokens) > 0:
                    entity_split(sequence_tokens, sequence_pred_tags, id2tag, entity_predict, cur)
                    entity_split(sequence_tokens, sequence_gold_tags, id2tag, entity_label, cur)
                    cur += len(sequence_tokens)

        # Calculate F-score
        right_predict = [i for i in entity_predict if i in entity_label]
        if len(right_predict) != 0:
            precision = float(len(right_predict)) / len(entity_predict)
            recall = float(len(right_predict)) / len(entity_label)
            fscore = (2 * precision * recall) / (precision + recall)
            logging.info(f"precision: {precision:.4f}")
            logging.info(f"recall: {recall:.4f}")
            logging.info(f"fscore: {fscore:.4f}")
        else:
            precision = 0
            recall = 0
            fscore = 0
            logging.info("precision: 0")
            logging.info("recall: 0")
            logging.info("fscore: 0")
        model.train()
        return fscore


def main(args):
    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    logging.info(f"Using device: {device}")

    with open('data/datasave.pkl', 'rb') as inp:
        word2id = pickle.load(inp)
        id2word = pickle.load(inp)
        tag2id = pickle.load(inp)
        id2tag = pickle.load(inp)
        x_train = pickle.load(inp)
        y_train = pickle.load(inp)
        x_test = pickle.load(inp)
        y_test = pickle.load(inp)

    # Initialize BERT model
    model = BertCWS(tag2id=tag2id, dropout=args.dropout)
    if use_cuda:
        model = model.cuda()
    
    # Log model parameters
    for name, param in model.named_parameters():
        logging.debug('%s: %s, require_grad=%s' % (name, str(param.shape), str(param.requires_grad)))

    # Create datasets and dataloaders
    train_dataset = BertSentence(x_train, y_train, id2word, tag2id, max_length=args.max_length)
    test_dataset = BertSentence(x_test, y_test, id2word, tag2id, max_length=args.max_length)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=BertSentence.collate_fn,
        num_workers=12,
        pin_memory=True,
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=BertSentence.collate_fn,
        num_workers=8,
    )

    # Set up optimizer and learning rate scheduler
    # No weight decay for bias and LayerNorm parameters
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
    
    # Total number of training steps
    total_steps = len(train_dataloader) * args.max_epoch
    
    # Create learning rate scheduler
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
        num_cycles=0.5  # 半周期余弦
    )

    # Track best model performance
    best_fscore = 0
    no_improvement = 0

    for epoch in range(args.max_epoch):
        model.train()
        losses = []
        logging.info(f"Starting epoch {epoch+1}/{args.max_epoch}")
        
        for step, batch in enumerate(train_dataloader):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']
            
            if use_cuda:
                input_ids = input_ids.cuda()
                attention_mask = attention_mask.cuda()
                labels = labels.cuda()

            # Create valid tags mask, excluding values of -100
            valid_tags_mask = (labels >= 0)

            # Forward pass
            loss = model(input_ids, attention_mask, labels, valid_tags_mask)
            losses.append(loss.item())

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()

            if step % 50 == 0:
                avg_loss = sum(losses[-50:]) / min(len(losses), 50)
                logging.debug(f'Epoch {epoch+1} - Step {step} - Loss: {avg_loss:.4f}')

        # Evaluate model after each epoch
        logging.info(f"Evaluating model after epoch {epoch+1}...")
        fscore = evaluate_model(model, test_dataloader, id2tag, use_cuda)
        
        # Save model if it's the best so far
        if fscore > best_fscore:
            best_fscore = fscore
            torch.save(model, "./save/best_bert_model.pkl")
            logging.info(f"New best model saved with fscore: {fscore:.4f}")
            no_improvement = 0
        else:
            no_improvement += 1
            logging.info(f"No improvement for {no_improvement} epochs. Best fscore: {best_fscore:.4f}")
        
        # Early stopping
        if no_improvement >= args.early_stopping:
            logging.info(f"Early stopping triggered after {epoch+1} epochs!")
            break

        # Save current epoch's model
        torch.save(model, f"./save/bert_model_epoch{epoch}.pkl")
        logging.info(f"Model saved as bert_model_epoch{epoch}.pkl")


if __name__ == '__main__':
    set_logger()
    main(get_param()) 