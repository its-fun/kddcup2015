# kddcup2015

KDD Cup 2015. https://www.kddcup2015.com/



## TODO

0. 提取特征

1. 把课程材料更新时间加到feature中：

  + 最后一次课程材料更新距今几天

  + 对课程材料的操作，据该材料的发布有几天 -> 最近一周内的平均值、两周的平均值、三周的......


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



## dataset

### instance building

主要是选择一个时间点T，T以后10天的数据用来提取标签，T以及T以前的所有数据用来提取特征。

第一个T是最大的时间，第二个是10天前，接下来每个都是上一个的7天前，直到数据不够为止



## features

### 数量

+ 用户在该课程的操作数量，前一周、前两周、前三周、前四周的，以及更以前的时间

+ 用户当前同时选的课程数

+ 用户在这门课程期间放弃了几门其他课程

+ 有多少人选这门课


### 比例

+ 用户在该课程的操作数量占他在所有课程的操作数量的比例，前一周、前两周、前三周、前四周的，
以及更以前的时间

+ 用户在该课程的操作数量占所有用户在该课程的操作数量的比例，前一周、前两周、前三周、前四周的，
以及更以前的时间

+ 用户的总放弃率，前一周、前两周、前三周、前四周、更以前的时间

+ 课程的总放弃率，前一周、前两周、前三周、前四周、更以前的时间

+ 用户在这门课程期间放弃的其他课程的数量与这期间所有选过的课的总数的比例


### 时间相关

+ 这门课程的材料首次发布距今的日期差

+ 用户上次操作据今的日期差

+ 用户对课程材料的首次操作时间与课程材料发布时间的日期差的平均值，前一周、前两周、前三周、前四周、更以前的时间


### 其他特征

+ 用户的兴趣偏好（需要另外建模）与课程的特征
