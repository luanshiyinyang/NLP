# -*-coding:utf-8-*-
import jieba
sentence = "中文分词是文本处理不可或缺的一步！"

seg_list = jieba.cut(sentence, cut_all=True)
print('全模式', '/'.join(seg_list))

seg_list = jieba.cut(sentence, cut_all=False)
print('精确模式', '/'.join(seg_list))

seg_list = jieba.cut(sentence)
print('默认精确模式', '/'.join(seg_list))

seg_list = jieba.cut_for_search(sentence)
print('搜索引擎模式', '/'.join(sentence))
