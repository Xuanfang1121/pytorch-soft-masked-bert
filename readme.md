### Soft Maksed Bert


#### 依赖
```
torch==1.8.1
pandas==1.1.5
transformers==4.10.1
numpy==1.19.2
scikit_learn==1.0.2
```

#### 实验结果
在sighan 数据集上的实验结果为
```
sighan数据集
gamma = 0.8
acc: 0.9778
句子绝对相等 acc: 0.5472
detector预测错误位置处的字与真实句子相应位置的字是否相等 abs acc：0.5636

gamma = 0.7
acc: 0.9780
句子绝对相等 acc: 0.5454
detector预测错误位置处的字与真实句子相应位置的字是否相等 abs acc：0.5727
sent abs acc:0.5445454545454546, precision:0.5734767025089605, recall:0.2952029520295203 f1 score:0.389768574908648
```
