## Chinese Word Match 2018-10

#### 1.preprocess

prepare() 将按类文件保存的训练数据汇总、清洗、去重，同类语句合并为文档

#### 2.extract

rank_fit() 使用 textrank、freq_fit() 使用 tfidf，提取各类的关键词和权重

通过 pos_set 限定词性、not_key 过滤无效词，修改 textrank 代码、保留单字

#### 3.match

predict() 实时交互，输入单句、清洗后进行预测，输出所有类别的得分