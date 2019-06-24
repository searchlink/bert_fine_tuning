# -*- coding: utf-8 -*-
# @Time    : 2019/6/10 14:55
# @Author  : skydm
# @Email   : wzwei1636@163.com
# @File    : bert_fine_tuning.py
# @Software: PyCharm

'''
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

'''
import os
os.environ["CUDA_VISIBLE_DEVICES"] ="5"
import pickle
import bert
from bert import run_classifier
from bert import tokenization
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.models import *
import tensorflow.python.keras.backend as K
from tensorflow.python.keras.utils import plot_model

sess = tf.Session()
fp = "/tmp/pycharm_project_717/demo-chinese-text-binary-classification-with-bert/dianping_train_test.pickle"
bert_path="/tmp/pycharm_project_717/keras-bert/bert_module"
max_seq_length = 256

params = {
        "DATA_COLUMN": "comment",
        "LABEL_COLUMN": "sentiment",
        "LEARNING_RATE": 2e-5,
        "NUM_TRAIN_EPOCHS": 5,
        "bert_model_hub": "/tmp/pycharm_project_717/keras-bert/bert_module"
    }

def load_data(fp):
    with open(fp, 'rb') as f:
        train, test = pickle.load(f)
        # shuffle data
        train = train.sample(len(train))
        return train, test

train, test = load_data(fp)
print(train.shape, '\n', train.head(), '\n', test.shape,  '\n', test.head())

label_list = train[params["LABEL_COLUMN"]].unique().tolist()

def create_tokenizer_from_hub_module(bert_model_hub):
    """Get the vocab file and casing info from the Hub module."""
    with tf.Graph().as_default():
        bert_module = hub.Module(bert_model_hub)
        # 实例化图的signature
        tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
        with tf.Session() as sess:
            vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"], tokenization_info["do_lower_case"]])

    return bert.tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)

def convert_text_to_examples(dataset, label_list, max_seq_length, tokenizer, DATA_COLUMN, LABEL_COLUMN):
    # # Convert data to InputExample format
    print(dataset[DATA_COLUMN].head())
    input_example = dataset.apply(lambda row: bert.run_classifier.InputExample(guid=None,
                                                                               text_a=row[DATA_COLUMN],
                                                                               text_b=None,
                                                                               label=row[LABEL_COLUMN]),
                                  axis=1)
    """Convert a set of `InputExample`s to a list of `InputFeatures`."""
    # 经过了padding和masking，进行了标记
    features = bert.run_classifier.convert_examples_to_features(input_example, label_list, max_seq_length, tokenizer)
    return features

class BertLayer(Layer):
    '''BertLayer which support next output_representation param:

        pooled_output: the first CLS token after adding projection layer () with shape [batch_size, 768].
        sequence_output: all tokens output with shape [batch_size, max_length, 768].
        mean_pooling: mean pooling of all tokens output [batch_size, max_length, 768].


        You can simple fine-tune last n layers in BERT with n_fine_tune_layers parameter.
        For view trainable parameters call model.trainable_weights after creating model.

    '''
    def __init__(self, n_fine_tune_layers=3, bert_path=bert_path, output_type='pooled_output', trainable=True, **kwargs):
        super(BertLayer, self).__init__(**kwargs)
        self.n_fine_tune_layers = n_fine_tune_layers
        self.bert_path = bert_path
        self.output_type = output_type
        self.is_trainable = trainable
        self.output_size = 768
        self.supports_masking = True

    def build(self, input_shape):
        # 加载bert模型
        self.bert = hub.Module(self.bert_path, trainable=self.is_trainable, name="{}_module".format(self.name))

        # 获取所有变量
        variables = self.bert.variables

        if self.is_trainable:
            trainable_vars = [var for var in variables if '/cls/' not in var.name]  # 'bert_module/bert/embeddings/LayerNorm/beta:0'

            if self.output_type == "sequence_output" or self.output_type == "mean_pooling":
                # 移除/pooler/层
                trainable_vars = [var for var in trainable_vars if '/pooler/' not in var.name]

            # 选择调参的层数
            trainable_layers = []
            for i in range(self.n_fine_tune_layers):
                trainable_layers.append("encoder/layer_{}".format(str(11 - i)))

            # Update trainable vars to contain only the specified layers
            final_trainable_vars = []
            for var in trainable_vars:
                for layer in trainable_layers:
                    if layer in var.name:
                        final_trainable_vars.append(var)

            # 添加trainable weights
            for var in final_trainable_vars:
                self._trainable_weights.append(var)  # self._trainable_weights is list

            # 添加non-trainable weights
            for var in self.bert.variables:
                if var not in self._trainable_weights:
                    self._non_trainable_weights.append(var)

        else:
            for var in variables:
                self._non_trainable_weights.append(var)

        self.built = True

    def call(self, inputs):
        '''
            pooled_output: pooled output of the entire sequence with shape [batch_size, hidden_size].
            sequence_output: representations of every token in the input sequence with shape [batch_size, max_sequence_length, hidden_size].
            类似LSTM的输出, 通过return_sequence = True/False来输出一个还是多个向量

            https://github.com/google-research/bert/blob/master/run_classifier_with_tfhub.py
            https://tfhub.dev/google/bert_chinese_L-12_H-768_A-12/1
        '''
        # 需要进行格式转化
        #  input_ids, input_mask, and segment_ids are int32 Tensors of shape [batch_size, max_sequence_length]
        inputs = [K.cast(x, 'int32') for x in inputs]
        input_ids, input_mask, segment_ids = inputs

        bert_inputs = dict(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids)

        # Instantiates a module signature in the graph 构造图， 包含变量的输出结果
        #　This modules outputs a representations for every token in the input sequence and a pooled representation of the entire input.
        result = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)

        if self.output_type == "pooled_output":
            pooled = result["pooled_output"]    # [batch_size, 768]

        elif self.output_type == "mean_pooling":
            result_tmp = result["sequence_output"]  # (batch_size, seq_len, 768)

            # 进行mask操作
            mul_mask = lambda x, m : x * K.expand_dims(m, axis=-1)  # # (batch_size, seq_len, 768)
            masked_reduce_mean = lambda x, m: tf.reduce_sum(mul_mask(x, m), axis=1) / (K.sum(m, axis=1, keepdims=True) + 1e-10)
            input_mask = K.cast(input_mask, "float32")  # (batch_size, seq_len)
            pooled = masked_reduce_mean(result_tmp, input_mask)  # # (batch_size, 768)

        elif self.output_type == "sequence_output":
            pooled = result["sequence_output"]

        return pooled

    def compute_mask(self, inputs, mask=None):
        '''
        # do not pass the mask to the next layer
        在调用call方法时, 会调用compute_mask方法
        如果还要继续输出mask, 供之后的层使用, 需要调用compute_mask方法
        The mask is one rank lower than the input, so if you have (batch_size, seq_len, channels), the mask will usually be (batch_size, seq_len)
        '''
        if self.output_type == "sequence_output":
            mask = K.cast(inputs[1], "bool")
            return mask
        else:
            return None

    def compute_output_shape(self, input_shape):
        if self.output_type == "sequence_output":
            return input_shape[0][0], input_shape[0][1], self.output_size
        else:
            return input_shape[0][0], self.output_size


def build_model(max_seq_length):
    input_ids = Input(shape=(max_seq_length,), name="input_ids")      # token embedding
    input_mask = Input(shape=(max_seq_length,), name="input_mask")     # masking
    segment_ids = Input(shape=(max_seq_length,), name="segment_ids")    # segment embedding
    in_bert = [input_ids, input_mask, segment_ids]

    bert_output = BertLayer(n_fine_tune_layers=3, bert_path=bert_path, output_type='mean_pooling', trainable=True)(in_bert)
    dense = Dense(256, activation="relu")(bert_output)
    pred = Dense(1, activation="sigmoid")(dense)

    model = Model(inputs=in_bert, outputs=pred)
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.summary()
    plot_model(model, "./bert_structure.png", show_shapes=True)

    return model

def initialize_vars(sess):
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    K.set_session(sess)

# Instantiate tokenizer
tokenizer = create_tokenizer_from_hub_module(bert_path)

# Convert data to InputExample format
train_features = convert_text_to_examples(train, label_list, max_seq_length, tokenizer, params["DATA_COLUMN"], params["LABEL_COLUMN"])
test_features = convert_text_to_examples(test, label_list, max_seq_length, tokenizer, params["DATA_COLUMN"], params["LABEL_COLUMN"])

# 构建模型输入
train_input_ids = []
train_input_masks = []
train_segment_ids = []
train_labels = []

for feature in train_features:
    train_input_ids.append(feature.input_ids)
    train_input_masks.append(feature.input_mask)
    train_segment_ids.append(feature.segment_ids)
    train_labels.append(feature.label_id)

test_input_ids = []
test_input_masks = []
test_segment_ids = []
test_labels = []

for feature in test_features:
    test_input_ids.append(feature.input_ids)
    test_input_masks.append(feature.input_mask)
    test_segment_ids.append(feature.segment_ids)
    test_labels.append(feature.label_id)


# 构建模型
# bert = hub.Module(bert_path, trainable=True, name="bert_module")
model = build_model(max_seq_length)

# Instantiate variables
initialize_vars(sess)

model.fit(x=[train_input_ids, train_input_masks, train_segment_ids],
          y=train_labels,
          validation_data=([test_input_ids, test_input_masks, test_segment_ids], test_labels),
          epochs=1,
          batch_size=32)


