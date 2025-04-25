import pickle
import logging
import argparse
import os
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from model import CWS
from dataloader import Sentence
from torch.optim.lr_scheduler import ReduceLROnPlateau

def get_param():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_dim', type=int, default=200)  # 增加嵌入维度
    parser.add_argument('--lr', type=float, default=0.001)  # 降低初始学习率
    parser.add_argument('--max_epoch', type=int, default=20)  # 增加训练轮数
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--hidden_dim', type=int, default=400)  # 增加隐藏层大小
    parser.add_argument('--dropout', type=float, default=0.3)  # 添加dropout参数
    parser.add_argument('--weight_decay', type=float, default=1e-5)  # 添加L2正则化
    parser.add_argument('--early_stopping', type=int, default=5)  # 早停轮数
    parser.add_argument('--cuda', action='store_true', default=False)
    return parser.parse_args()


def set_logger():
    log_file = os.path.join('save', 'log.txt')
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
        for sentence, label, mask, length in test_data:
            if use_cuda:
                sentence = sentence.cuda()
                label = label.cuda()
                mask = mask.cuda()
            predict = model.infer(sentence, mask, length)

            for i in range(len(length)):
                entity_split(sentence[i, :length[i]], predict[i], id2tag, entity_predict, cur)
                entity_split(sentence[i, :length[i]], label[i, :length[i]], id2tag, entity_label, cur)
                cur += length[i]

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

    with open('data/datasave.pkl', 'rb') as inp:
        word2id = pickle.load(inp)
        id2word = pickle.load(inp)
        tag2id = pickle.load(inp)
        id2tag = pickle.load(inp)
        x_train = pickle.load(inp)
        y_train = pickle.load(inp)
        x_test = pickle.load(inp)
        y_test = pickle.load(inp)

    model = CWS(len(word2id), tag2id, args.embedding_dim, args.hidden_dim, dropout=args.dropout)
    if use_cuda:
        model = model.cuda()
    for name, param in model.named_parameters():
        logging.debug('%s: %s, require_grad=%s' % (name, str(param.shape), str(param.requires_grad)))

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=2, verbose=True)

    train_data = DataLoader(
        dataset=Sentence(x_train, y_train),
        shuffle=True,
        batch_size=args.batch_size,
        collate_fn=Sentence.collate_fn,
        drop_last=False,
        num_workers=12,           
        pin_memory=True,          # 启用固定内存
        persistent_workers=True,  # 保持worker进程活跃
        prefetch_factor=3         # 预取批次数
    )

    test_data = DataLoader(
        dataset=Sentence(x_test[:1000], y_test[:1000]),
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=Sentence.collate_fn,
        drop_last=False,
        num_workers=6
    )

    # 在训练循环中添加早停机制
    best_fscore = 0
    no_improvement = 0

    for epoch in range(args.max_epoch):
        step = 0
        log = []
        for sentence, label, mask, length in train_data:
            if use_cuda:
                sentence = sentence.cuda()
                label = label.cuda()
                mask = mask.cuda()

            # forward
            loss = model(sentence, label, mask, length)
            log.append(loss.item())

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            if step % 100 == 0:
                logging.debug('epoch %d-step %d loss: %f' % (epoch, step, sum(log)/len(log)))
                log = []

        # 评估模型
        logging.info(f"Evaluating model after epoch {epoch+1}...")
        fscore = evaluate_model(model, test_data, id2tag, use_cuda)
        
        # 学习率调度
        scheduler.step(-fscore)  # 使用负值，因为ReduceLROnPlateau默认是最小化
        
        # 保存最佳模型
        if fscore > best_fscore:
            best_fscore = fscore
            torch.save(model, f"./save/best_model.pkl")
            logging.info(f"New best model saved with fscore: {fscore:.4f}")
            no_improvement = 0
        else:
            no_improvement += 1
            logging.info(f"No improvement for {no_improvement} epochs. Best fscore: {best_fscore:.4f}")
        
        # 早停
        if no_improvement >= args.early_stopping:
            logging.info(f"Early stopping triggered after {epoch+1} epochs!")
            break

        # 保存当前epoch的模型
        path_name = f"./save/model_epoch{epoch}.pkl"
        torch.save(model, path_name)
        logging.info(f"Model has been saved in {path_name}")


if __name__ == '__main__':
    set_logger()
    main(get_param())
