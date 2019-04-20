# -*- coding:utf-8 -*-
import pkuseg


def test1():
    '''
    使用默认模型和默认词典分词
    :return:
    '''
    # 创建分析对象
    seg = pkuseg.pkuseg()
    rst = seg.cut("这是一个小小的测试而已哦")
    print(rst)


def test2():
    '''
    使用自定义词典,词典中存在的一定会单独成词
    :return:
    '''
    myDict = "dict.txt"
    seg = pkuseg.pkuseg(user_dict=myDict)
    rst = seg.cut("我是一个好好学生")
    print(rst)


def test3():
    '''
    使用其他模型，不使用字典
    :return:
    '''
    # 必须下载这个模型且ctb8这个目录放在当前目录下
    seg = pkuseg.pkuseg(model_name='models/ctb8', user_dict=None)
    rst = seg.cut('我是一个好学生')
    print(rst)


def test4():
    '''
    对文件进行分词
    :return:
    '''
    # 对input.txt的文件分词输出到output.txt中
    # 使用默认模型，使用词典，开20个进程
    pkuseg.test('input.txt', 'output.txt', nthread=5)


def test5():
    '''
    训练新模型
    :return:
    '''
    # 训练文件为'msr_training.utf8'
    # 测试文件为'msr_test_gold.utf8'
    # 训练好的模型存到'./models'目录下，开20个进程训练模型
    # 训练模式下会保存最后一轮模型作为最终模型
    # 目前仅支持utf-8编码，训练集和测试集要求所有单词以单个或多个空格分开
    pkuseg.train('msr_training.utf8', 'msr_test_gold.utf8', 'models/', nthread=20)


if __name__ == '__main__':
    # test1()
    # test2()
    # test3()
    test4()
    # test5()