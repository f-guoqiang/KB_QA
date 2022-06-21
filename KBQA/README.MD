## 基于知识图谱的医疗诊断知识问答系统

### 效果展示

对话上下文并没有使用Redis之类的数据库实时存储到内存里，而是使用json文件的存、读进行的。

能够回答哪些疾病相关知识，可以参考[这个疾病实体字典里的疾病](./bilstm_crf/checkpoint/diseases.json)；另外目前并没有实现实体链指的功能（因为没有相关的开源数据集），因此现在只支持“疾病”实体的完整、准确输入的解析。

演示视频地址:(待补充)

### 如何运行demo代码

0. **下载意图识别模型文件**

由于该模型是使用bert训练的，比较大，因此我上传到百度网盘了，有需要的可以自己去下载，然后将其放到目录`KBQA-for-Diagnosis\nlu\bert_intent_recognition\checkpoint`下。

方法一 **下载地址百度云**
https://github.com/fgq1994/KB_QA/blob/a03e9a27f95535582fad2174552260ea19d03bb3/KBQA/README.MD

方法二 **下载roberta预训练模型**
[从哈工大官方github下载，点这里](https://github.com/ymcui/Chinese-BERT-wwm#%E4%B8%AD%E6%96%87%E6%A8%A1%E5%9E%8B%E4%B8%8B%E8%BD%BD)，下载RBT3,Chinese 版本（貌似是这个，反正是一个三层的简单版）；[或者直接点这里用讯飞云下载](http://pan.iflytek.com/#/link/275E5B46185C982D4AF5AC295E1651B6)

下载好后解压放在项目路径下，然后在bert意图识别文件夹的 `app.py`代码里改一下预训练模型的路径。

1. **启动neo4j**
- 使用windows_desktop
- 命令行模式

2. **启动意图识别和槽位填充服务**
- bisltm app.py
- bert app.py
3. **启动微信小程序前端**



### 后期计划

