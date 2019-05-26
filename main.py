# encoding=utf8
import itertools
from collections import OrderedDict
import os
import tensorflow as tf
import numpy as np
from model import Model
from loader import load_sentences, update_tag_scheme
from loader import char_mapping, tag_mapping
from loader import augment_with_pretrained, prepare_dataset
from utils import get_logger, make_path, clean, create_model,create_testmodel
from utils import print_config, save_config, load_config, test_ner
from data_utils import load_word2vec, create_input, input_from_line, BatchManager
root_path=os.getcwd()+os.sep
flags = tf.app.flags
#定义参数
flags.DEFINE_boolean("clean",       True,      "clean train folder")
flags.DEFINE_boolean("train",       False,      "Whether train the model")
# configurations for the model
flags.DEFINE_integer("seg_dim",     20,         "Embedding size for segmentation, 0 if not used")#词语切分标记的维度
flags.DEFINE_integer("char_dim",    100,        "Embedding size for characters")#表示每个字符的向量的维度
flags.DEFINE_integer("lstm_dim",    100,        "Num of hidden units in LSTM, or num of filters in IDCNN")#lstm细胞的个数
flags.DEFINE_string("tag_schema",   "iobes",    "tagging schema iobes or iob")#命名实体标记的类型

# configurations for training
flags.DEFINE_float("clip",          5,          "Gradient clip")#梯度截断为5，大于5的全部取5
flags.DEFINE_float("dropout",       0.5,        "Dropout rate")
flags.DEFINE_float("batch_size",    60,         "batch size")
flags.DEFINE_float("lr",            0.001,      "Initial learning rate")
flags.DEFINE_string("optimizer",    "adam",     "Optimizer for training")
flags.DEFINE_boolean("pre_emb",     False,       "Wither use pre-trained embedding")#是否使用预训练的词嵌入
flags.DEFINE_boolean("zeros",       True,      "Wither replace digits with zero")
flags.DEFINE_boolean("lower",       False,       "Wither lower case")

flags.DEFINE_integer("max_epoch",   100,        "maximum training epochs")#最大训练次数
#保存模型的参数
flags.DEFINE_integer("steps_check", 100,        "steps per checkpoint")
flags.DEFINE_string("ckpt_path",    "./ckpt",      "Path to save model")
flags.DEFINE_string("summary_path", "summary",      "Path to store summaries")
flags.DEFINE_string("log_file",     "train.log",    "File for log")
flags.DEFINE_string("map_file",     "./config/maps.pkl",     "file for maps")
flags.DEFINE_string("vocab_file",   "vocab.json",   "File for vocab")
flags.DEFINE_string("config_file",  "./config/config_file",  "File for config")
flags.DEFINE_string("tag_to_id_path",  "./config/tag_to_id.txt",  "File for tag_to_id.txt")
flags.DEFINE_string("id_to_tag_path",  "./config/id_to_tag.txt",  "File for id_to_tag.txt")

flags.DEFINE_string("script",       "conlleval",    "evaluation script")
flags.DEFINE_string("result_path",  "result",       "Path for results")
#训练数据的路径
flags.DEFINE_string("emb_file",     os.path.join(root_path+"data", "vec.txt"),  "Path for pre_trained embedding")
flags.DEFINE_string("train_file",   os.path.join(root_path+"data", "example.dev"),  "Path for train data")  # example.train
flags.DEFINE_string("dev_file",     os.path.join(root_path+"data", "example.dev"),    "Path for dev data")   # 注意，这里的数据集我都做了修改，为了在我的机器上运行
flags.DEFINE_string("test_file",    os.path.join(root_path+"data", "example.dev"),   "Path for test data")   # example.test
#使用的模型
flags.DEFINE_string("model_type", "idcnn", "Model type, can be idcnn or bilstm")
# flags.DEFINE_string("model_type", "bilstm", "Model type, can be idcnn or bilstm")

FLAGS = tf.app.flags.FLAGS#可以从对应的命令行参数取出参数
#如果参数不合理就退出
assert FLAGS.clip < 5.1, "gradient clip should't be too much"
assert 0 <= FLAGS.dropout < 1, "dropout rate between 0 and 1"
assert FLAGS.lr > 0, "learning rate must larger than zero"
assert FLAGS.optimizer in ["adam", "sgd", "adagrad"]


# 模型的参数配置
def config_model(char_to_id, tag_to_id):
    config = OrderedDict()
    config["model_type"] = FLAGS.model_type
    config["num_chars"] = len(char_to_id)
    config["char_dim"] = FLAGS.char_dim
    config["num_tags"] = len(tag_to_id)
    config["seg_dim"] = FLAGS.seg_dim
    config["lstm_dim"] = FLAGS.lstm_dim
    config["batch_size"] = FLAGS.batch_size

    config["emb_file"] = FLAGS.emb_file
    config["clip"] = FLAGS.clip
    config["dropout_keep"] = 1.0 - FLAGS.dropout
    config["optimizer"] = FLAGS.optimizer
    config["lr"] = FLAGS.lr
    config["tag_schema"] = FLAGS.tag_schema
    config["pre_emb"] = FLAGS.pre_emb
    config["zeros"] = FLAGS.zeros
    config["lower"] = FLAGS.lower
    return config

def evaluate(sess,model,name,data,id_tag):
    ner_results = model.evaluate(sess,data,id_tag)#这一批验证数据的句子和每个字真实的tag和预测的tag
    eval_lines = test_ner(ner_results,FLAGS.result_path)#将验证结果写入文件中（ner_results），然后计算得到F1等批判模型好坏的指标
    f1 = float(eval_lines[1].strip().split()[-1])#截取出f1指标
    return f1

import pickle
def evaluate_line():
    config = load_config(FLAGS.config_file)
    logger = get_logger(FLAGS.log_file)
    # limit GPU memory
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with open(FLAGS.map_file, "rb") as f:
        char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)
    with tf.Session(config=tf_config) as sess:
        model = create_testmodel(sess, Model, FLAGS.ckpt_path, load_word2vec, config, id_to_char)
        while True:
                line = input("请输入测试句子:")
                result = model.evaluate_line(sess, input_from_line(line, char_to_id), id_to_tag)
                print(result)

def train():
    #加载训练用的数据
    train_sentences = load_sentences(FLAGS.train_file, FLAGS.lower, FLAGS.zeros)
    #加载验证集和测试集合
    dev_sentences = load_sentences(FLAGS.dev_file, FLAGS.lower, FLAGS.zeros)
    test_sentences = load_sentences(FLAGS.test_file, FLAGS.lower, FLAGS.zeros)
    # Use selected tagging scheme (IOB / IOBES)使用选定的标记方案I：中间，O：其他，B：开始 | E：结束，S：单个
    update_tag_scheme(train_sentences, FLAGS.tag_schema)
    update_tag_scheme(test_sentences, FLAGS.tag_schema)
    update_tag_scheme(dev_sentences, FLAGS.tag_schema)
    _c, char_to_id, id_to_char = char_mapping(train_sentences, FLAGS.lower)#统计每个字的频率以及为每个字分配一个id
    _t, tag_to_id, id_to_tag = tag_mapping(train_sentences,FLAGS.id_to_tag_path,FLAGS.tag_to_id_path)#统计每个命名实体的频率以及为每个命名实体分配一个id
    #将字典写入pkl文件中
    with open(FLAGS.map_file, "wb") as f:
            pickle.dump([char_to_id, id_to_char, tag_to_id, id_to_tag], f)
    #准备数据，获取包含索引的列表集合，得到用于输入网络进行训练的数据
    train_data = prepare_dataset(                                    # train_data[0][0]:一句话；train_data[0][1]：单个字的编号；train_data[0][2]：切词之后，切词特征：词的大小是一个字的话是0，词的大小是2以上的话：1,2.。。，2,3； train_data[0][3]：每个字的标签
        train_sentences, char_to_id, tag_to_id, FLAGS.lower
    )
    dev_data = prepare_dataset(
        dev_sentences, char_to_id, tag_to_id, FLAGS.lower
    )
    test_data = prepare_dataset(
        test_sentences, char_to_id, tag_to_id, FLAGS.lower
    )
    train_manager = BatchManager(train_data, FLAGS.batch_size)        # 将数据拆分成以60句话为一个batch，得到一个可迭代对象
    dev_manager = BatchManager(dev_data, 100)
    test_manager = BatchManager(test_data, 100)
    config = config_model(char_to_id, tag_to_id)#补全参数配置
    #限制GPU的使用
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        model = create_model(sess, Model,load_word2vec, config, id_to_char)
        saver = tf.train.Saver()  # 用于保存模型
        with tf.device("/cpu:0"):
            for i in range(100):
                for batch in train_manager.iter_batch(shuffle=True):
                    step, batch_loss = model.run_step(sess, True, batch)          # 按批次训练模型 这个是训练的开始，可以从这里倒着找整个网络怎么训练
                #每训练5次做一次验证并计算模型的f1
                if (i+1) % 1 == 0:
                    f1 = evaluate(sess,model,"dev",dev_manager,id_to_tag)
                    print("验证集的F1系数：",f1)
                #每训练20次保存一次模型
                if (i+10) % 1 == 0:
                    saver.save(sess,save_path=FLAGS.ckpt_path)
if __name__ == "__main__":
    train()#训练
    # evaluate_line()#测试