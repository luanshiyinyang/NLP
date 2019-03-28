# -*-coding:utf-8-*-


def get_content(path):
    """
    获取文件内容
    :param path:
    :return:
    """
    with open(path, 'r', encoding='gbk', errors='ignore') as f:
        content = ''
        for l in f:
            l = l.strip()
            content += l
        return content


def get_TF(words, topK=10):
    """
    计算TF值
    :param words:
    :param topK:
    :return:
    """
    tf_dic = {}
    for w in words:
        tf_dic[w] = tf_dic.get(w, 0) + 1
    return sorted(tf_dic.items(), key=lambda x: x[1], reverse=True)[:topK]


def stop_words(path):
    """
    获取停用词
    :param path:
    :return:
    """
    with open(path, encoding='utf-8') as f:
        return [l.strip() for l in f]


stop_words('./data/stop_words.utf8')


def main():
    """
    主模块
    :return:
    """
    # 文件操作模块，允许通配符等
    import glob
    import jieba
    import random
    # 这里只使用13年数据
    files = glob.glob('./data/news/C000013/*.txt')
    corpus = [get_content(x) for x in files]
    # 随机文字进行高频词提取
    sample_inx = random.randint(0, len(files))
    split_words = [x for x in jieba.cut(corpus[sample_inx]) if x not in stop_words('./data/stop_words.utf8')]
    print('样本之一：' + corpus[sample_inx])
    print('样本分词效果：' + '/ '.join(split_words))
    print('样本的topK（10）词：' + str(get_TF(split_words)))


if __name__ == '__main__':
    main()
