# bert_fine_tuning
基于tensorflow hub的bert的fine_tuning进行二分类

后续会不断完善，过段时间放出自己的jupyter版本

预备工作
```
pip install tensorflow-hub
pip install bert-tensorflow

# 加载本地下载的bert文件， 参考如下链接
https://github.com/tensorflow/hub/blob/master/docs/common_issues.md
mkdir bert_module   # 创建保存bert的文件夹
curl -L "https://tfhub.dev/google/bert_chinese_L-12_H-768_A-12/1?tf-hub-format=compressed" | tar -zxvC ./       # 下载解压并保存bert

bert_path = "https://tfhub.dev/google/bert_chinese_L-12_H-768_A-12/1"

about explation of output:
https://tfhub.dev/google/bert_chinese_L-12_H-768_A-12/1
The pooled_output is a [batch_size, hidden_size] Tensor.
The sequence_output is a [batch_size, sequence_length, hidden_size] Tensor.
```
