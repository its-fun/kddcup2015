# kddcup2015

KDD Cup 2015. https://www.kddcup2015.com/



## project structure

```
|
|- data/  # 由于数据量很大，因此应该自己创建这个文件夹，并且将数据文件按照这个结构放好
   |- cache/  # pickle dump出来的对象，用于加快文件读取速度
   |- test/  # 测试数据
      |- enrollment_test.csv
      |- log_test.csv
   |- train/  # 训练数据
      |- enrollment_train.csv
      |- log_train.csv
      |- truth_train.csv
   |- object.csv
|- path_config.py  # 与数据路径相关的配置
|- modeling_config.py  # 与模型训练相关的配置
|- dataset.py  # 生成数据集的相关方法
|- feature_extraction.py  # 提取特征的相关方法
|- util.py  # IO等辅助性的工具
|- modeling.py  # 建模的方法
```

Run `python3 dataset.py` to load the dataset



## TODO

0. cache的数据需要进行gzip压缩 (DONE)

0. 根据特征选择的结果对特征进行重新加工

1. 观察决策树，进行特征选择、加工

2. 谨慎进行特征normalize，最好手工来做

2. 手工加入规则，比如关于课程最近更新时间、用户最近操作时间

3. 逆向工程找到这39门课分别是什么

3. 观察预测错了的instance

4. 尝试深度学习model

5. AdaBoost, RF, Bagging, Blending



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


**NOTE**:
首先是已经停课的先筛选出来，预测成dropout。
还在进行中的课程才用这里的模型来预测是否dropout。



## features


1. count by source-event pair (0 ~ 44)

```
                    0, 1, 2, 3, more
browser-access      v  v
browser-page_close  v  v
browser-problem     v  v
browser-video          v
server-access
server-discussion   v  v
server-navigate     v  v
server-problem      v  v  v
server-wiki         v  v  v  v
```

2. #courses by user, weekly (45 ~ 48)

3. course population (49)

4. course dropout count (50)

5. ratio of user ops on all courses (51 ~ 95)

```
                    0, 1, 2, 3, more
browser-access      v  v  v  v
browser-page_close  v  v  v  v
browser-problem     v  v  v  v
browser-video       v  v  v  v
server-access       v  v  v  v
server-discussion   v  v  v  v
server-navigate     v  v  v  v  v
server-problem      v  v  v  v
server-wiki         v  v  v  v
```

6. ratio of courses ops of all users (96 ~ 140)

```
                    0, 1, 2, 3, more
browser-access
browser-page_close
browser-problem
browser-video
server-access
server-discussion
server-navigate
server-problem
server-wiki
```

7. dropout ratio of courses (141)

8. days from course first update (142)

9. days from course last update (143)

10. days from user last op (144)

11. days from user first op (145)

12. days from course first update to user first op (146)


### 数量

+ 用户在该课程的操作数量，前一周、前两周、前三周、前四周的，以及更以前的时间 (DONE)

+ 用户有行为的课程数量，前一周、前两周、前三周、前四周的 (DONE)

+ 用户在这门课程期间放弃了几门其他课程

+ 有多少人选这门课 (DONE)

+ 有多少人放弃了这门课 (DONE)

+ 用户在该课程中连续10天或以上没有操作的次数num_dropout (DONE)

+ 用户在其他所有课程中连续10天或以上没有操作的次数的平均值avg_dropout，方差std_dropout (DONE)


### 比例

+ 用户在该课程的操作数量占他在所有课程的操作数量的比例，前一周、前两周、前三周、前四周的，
以及更以前的时间 (DONE)

+ 用户在该课程的操作数量占所有用户在该课程的操作数量的比例，前一周、前两周、前三周、前四周的，
以及更以前的时间 (DONE)

+ 用户的放弃率

+ 课程的放弃率 (DONE)

+ 用户在这门课程期间放弃的其他课程的数量与这期间所有选过的课的总数的比例

+ 用户在该课程连续10天没有操作的次数与所有课程中连续10天没有操作的次数的平均值的比例 (DONE)

+ 连续10天没有操作的课程占用户所选的其他课程的比例 (DONE)

+ 在Poisson distribution中，出现大于num_dropout的概率


### 时间相关

+ 这门课程的材料首次发布距今几天了 (DONE)

+ 这门课程的材料最后一次发布距今几天了 (DONE)

+ 用户上次操作此课程据今几天 (DONE)

+ 用户初次操作此课程据今几天 (DONE)

+ 用户初次访问课程材料距离开课时间有多少天 (DONE)

+ 用户对课程材料的首次操作时间与课程材料发布时间的日期差的平均值，前一周、前两周、前三周、前四周、更以前的时间


### 其他特征

+ 用户的兴趣偏好（需要另外建模）与课程的特征
