# -*-coding:utf-8-*-


def cut_words():
    import fool
    text = "一个傻子在北京"
    print(fool.cut(text))


def user_dict():
    import fool
    fool.load_userdict('myDict.txt')
    text = ["我在北京天安门看你难受香菇", "我在北京晒太阳你在非洲看雪"]
    print(fool.cut(text))


def cixinbiaozhu():
    import fool

    text = ["一个傻子在北京"]
    print(fool.pos_cut(text))


def shitishibie():
    import fool

    text = ["一个傻子在北京", "你好啊"]
    words, ners = fool.analysis(text)
    print(ners)


if __name__ == '__main__':
    # cut_words()
    # user_dict()
    # cixinbiaozhu()
    shitishibie()
