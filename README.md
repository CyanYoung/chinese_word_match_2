## Chinese Word Match 2018-10

#### 1.preprocess

prepare() 将按类文件保存的训练数据汇总、去重，去除停用词，统一替换

地区、时间等特殊词，word_replace() 替换同音、同义词，同类语句合并为文档

#### 2.extract

rank_fit() 使用 textrank、freq_fit() 使用 tfidf，提取各类的关键词和权重

通过 pos_set 限定词性、not_key 过滤无效词，修改 textrank 代码、保留单字

#### 3.match

predict() 去除停用词，统一替换地区、时间等特殊词，切分后

分别通过 textrank、tfidf 与各类的关键词匹配、选取最大的平均权重