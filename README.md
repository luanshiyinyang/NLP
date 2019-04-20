# pkuseg使用
- 简介
	- 最近社区推了一些文章介绍了北大开源的一个新的中文分词工具包pkuseg。它在多个分词数据集上都有非常高的分词准确率。其中广泛使用的结巴分词（jieba）误差率高达18.55%和20.42%，而北大的pkuseg只有3.25%与4.32%。
	- 在中文处理领域，特别是数据分析挖掘这个领域，数据预处理重要性不言而喻，那么分词的重要性也是不言而喻的。
	- 简单使用pkuseg这个包，这是一个python语言写成的包，具体可以查看官方[GitHub地址](https://github.com/lancopku/PKUSeg-python)。
- 特点
	- 多领域分词。不同于以往的通用中文分词工具，此工具包同时致力于为不同领域的数据提供个性化的预训练模型。根据待分词文本的领域特点，用户可以自由地选择不同的模型。 目前支持了新闻领域，网络文本领域和混合领域的分词预训练模型，同时也拟在近期推出更多的细领域预训练模型，比如医药、旅游、专利、小说等等。
	- 更高的分词准确率。相比于其他的分词工具包，当使用相同的训练数据和测试数据，pkuseg可以取得更高的分词准确率。
	- 支持用户自训练模型。支持用户使用全新的标注数据进行训练。
- 安装
	- 目前只支持python3，这点无伤大雅，越来越多的项目向3迁移了。
	- 最新版本
		- 2019-1-23
	- pip安装
		- pip install pkuseg
		- 由于比较大，pip源下载可能比较慢，可以使用镜像源。
			- pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pkuseg
	- 源码安装
		- 访问上面的官方地址，下载源码放到项目目录下。
- 使用
	- 使用默认模型和默认字典
		- 代码
		- 结果
			- ![](https://img-blog.csdnimg.cn/20190128154954455.png)
		- 出现错误 
			- ValueError: numpy.ufunc size changed, may indicate binary incompatibility. Expected 216 from C header, got 192 from PyObject
			- 这是因为numpy版本过低，更新即可pip install --upgrade numpy。
	- 使用自定义字典
		- 代码
		- 结果
			- ![](https://img-blog.csdnimg.cn/20190128155221357.png)
	- 使用其他模型
		- 需要下载该模型
		- 代码
		- 结果
			- ![](https://img-blog.csdnimg.cn/2019012815550973.png)
	- 对文件分词(使用默认模型，使用词典)
		- 因为包括文件处理，这个过程比较慢，建议开多个线程。
		- 代码
		- 结果
			- ![](https://img-blog.csdnimg.cn/2019012815563987.png)
	- 训练新模型
		- 代码
	- 参数说明
		- 详见官方文档
		- pkuseg.pkuseg(model_name="default", user_dict="default")
			- model_name		
				- 模型路径。默认是"default"表示我们预训练好的模型(仅对pip下载的用户)。用户可以填自己下载或训练的模型所在的路径如model_name='./models'。
			- user_dict		
				- 设置用户词典。默认使用我们提供的词典。用户可以填自己的用户词典的路径，词典格式为一行一个词。填None表示不使用词典。
		- pkuseg.test(readFile, outputFile, model_name="default", user_dict="default", nthread=10)
			- readFile		
				- 输入文件路径
			- outputFile		
				- 输出文件路径
			- model_name		
				- 模型路径。同pkuseg.pkuseg
			- user_dict		
				- 设置用户词典。同pkuseg.pkuseg
			- nthread			
				- 测试时开的进程数
		- pkuseg.train(trainFile, testFile, savedir, nthread=10)
			- trainFile		
				- 训练文件路径
			- testFile		
				- 测试文件路径
			- savedir			
				- 训练模型的保存路径
			- nthread			
				- 训练时开的进程数
- 关于预训练模型
	- 官方提供了几种模型（文件均可以在我的GitHub找到）
	- MSRA: 在MSRA（新闻语料）上训练的模型。
	- CTB8: 在CTB8（新闻文本及网络文本的混合型语料）上训练的模型。
	- WEIBO: 在微博（网络文本语料）上训练的模型。
	- MixedModel: 混合数据集训练的通用模型。随pip包附带的是此模型。
	- 其中，MSRA数据由第二届国际汉语分词评测比赛提供，CTB8数据由LDC提供，WEIBO数据由NLPCC分词比赛提供。
- 补充说明
	- 详细代码和预训练模型文件均可以在我的GitHub找到
	- 具体了解请查看官方[参考文档](https://github.com/lancopku/PKUSeg-python)