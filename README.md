# kddcup2015

KDD Cup 2015. https://www.kddcup2015.com/



## TODO

0. 提取特征

1. 把课程材料更新时间加到feature中

2. 建立用户-课程语义模型

3. 融合log_train和log_test，以便获取比较完整的整体信息



## data peek

### possible source-event pair

```python
from data_util import load_train
train = load_train('./data/train/log_train.csv')
se_pair = train[['source', 'event']]
uniq_se_pair = set([tuple(p) for p in se_pair.values])
print(uniq_se_pair)
%history
```

    browser, access
    browser, page_close
    browser, problem
    browser, video

    server, access  # 不清楚是什么，但是应该是与browser, access不同的操作
    server, discussion
    server, navigate
    server, problem  # 可能是提交作业进行评分
    server, wiki


### enrollment log

In the training set:

120542 enrollment TOTAL

59569 enrollment, log in ONE hour


## features
