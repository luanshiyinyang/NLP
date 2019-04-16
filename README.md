# FoolNTLK的使用
- 简介
	- NLTK（自然语言处理工具包）可以说是五花八门，但是，用户wu.zheng开源的这个使用双向LSTM构建的中文处理工具包FoolNLTK，不仅可以实现分词、词性标注和命名实体识别，同时还能使用用户自定义字典加强分词的效果。
	- 该工具包[Github官方地址](https://github.com/rockyzhengwu/FoolNLTK)，不过，最近的一次维护已经是一年前了。
	- 该工具包有Java版本，Pyhton版本后端使用TensorFlow。
- 工具包特点
	- 可能不是最快的开源中文分词，但很可能是最准的开源中文分词
	- 基于BiLSTM模型训练而成
	- 包含分词，词性标注，实体识别,　都有比较高的准确率
	- 用户自定义词典
	- 可训练自己的模型
	- 批量处理
- 参考论文
	- 上述所说的BiLSTM可以参考[这篇论文](http://www.aclweb.org/anthology/N16-1030)。
- 具体使用
	- 安装
		- `pip install foolnltk`
		- 安装默认下载模型。
	- 分词
		- 代码如下
			- ```python
				import fool
				
				
				def cut_words():
				    text = "一个傻子在北京"
				    print(fool.cut(text))
				```
		- 也可以使用命令行对文件进行分词
			- `python -m fool [filename]`
			- 可指定-b参数,每次切割的行数,能加快分词速度
	- 用户自定义词典(类似jieba）
		- 词典每一行格式如下，词的权重越高，词的长度越长就越越可能出现,　权重值请大于1
			- 词语名称 权重值（建议整数且大于1）
		- 加载词典并使用
			- 代码如下
				- ```python
					def user_dict():
					    import fool
					    fool.load_userdict('myDict.txt')
					    text = ["我在北京天安门看你难受香菇", "我在北京晒太阳你在非洲看雪"]
					    print(fool.cut(text))
					
					```
		- 删除字典
			- fool.delete_userdict()
	- 词性标注（标注规则见官方文档）
		- 代码如下
			- ```python
				def cixinbiaozhu():
				    import fool
				
				    text = ["一个傻子在北京"]
				    print(fool.pos_cut(text))
				```
	- 实体识别
		- 代码如下
			- ```python
				def shitishibie():
				    import fool
				
				    text = ["一个傻子在北京", "你好啊"]
				    words, ners = fool.analysis(text)
				    print(ners)
				```
- 补充说明
	- 我的环境是Linux下Python3环境，Windows环境未知。
	- 找不到模型文件的, 可以看下sys.prefix,一般默认为/usr/local/。
	- 具体代码和配置好的Linux下的venv环境可以查看我的GitHub。