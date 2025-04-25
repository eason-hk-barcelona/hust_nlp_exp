class Tokenizer(object):
    def __init__(self, words, max_len):
        self.words = words
        self.max_len = max_len

    def fmm_split(self, text):
        '''
        正向最大匹配分词算法
        :param text: 待分词字符串
        :return: 分词结果，以list形式存放，每个元素为分出的词
        '''
        result = []
        while text: 

            max_len = min(self.max_len, len(text))
            word = text[:max_len]

            while word not in self.words and len(word) > 1:
                word = word[:-1]
            
            result.append(word)
            text = text[len(word):]
        
        return result

    def rmm_split(self, text):
        '''
        逆向最大匹配分词算法
        :param text: 待分词字符串
        :return: 分词结果，以list形式存放，每个元素为分出的词
        '''
        result = []
        while text: 

            max_len = min(self.max_len, len(text))
            word = text[-max_len:]

            while word not in self.words and len(word) > 1:
                word = word[1:]
            
            result.insert(0, word)
            text = text[:-len(word)]
        
        return result

    def bimm_split(self, text):
        '''
        双向最大匹配分词算法
        :param text: 待分词字符串
        :return: 分词结果，以list形式存放，每个元素为分出的词
        '''
        fmm_result = self.fmm_split(text)
        rmm_result = self.rmm_split(text)

        if fmm_result == rmm_result:
            return fmm_result
        
        if len(fmm_result) != len(rmm_result):
            return fmm_result if len(fmm_result) < len(rmm_result) else rmm_result
        
        fmm_single_char_count = sum(1 for word in fmm_result if len(word) == 1)
        rmm_single_char_count = sum(1 for word in rmm_result if len(word) == 1)

        return fmm_result if fmm_single_char_count < rmm_single_char_count else rmm_result


def load_dict(path):
    tmp = set()
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            word = line.strip().split(' ')[0]
            tmp.add(word)
    return tmp


if __name__ == '__main__':
    words = load_dict('dict.txt')
    max_len = max(map(len, [word for word in words]))

    # test
    tokenizer = Tokenizer(words, max_len)
    texts = [
        '研究生命的起源',
        '无线电法国别研究',
        '人要是行，干一行行一行，一行行行行行，行行行干哪行都行。'
    ]
    for text in texts:
        # 前向最大匹配
        print('前向最大匹配:', '/'.join(tokenizer.fmm_split(text)))
        # 后向最大匹配
        print('后向最大匹配:', '/'.join(tokenizer.rmm_split(text)))
        # 双向最大匹配
        print('双向最大匹配:', '/'.join(tokenizer.bimm_split(text)))
        print('')
