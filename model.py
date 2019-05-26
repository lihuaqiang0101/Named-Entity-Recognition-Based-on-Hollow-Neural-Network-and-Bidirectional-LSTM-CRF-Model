# encoding = utf-8
import numpy as np
import tensorflow as tf
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode
from tensorflow.contrib.layers.python.layers import initializers

from utils import result_to_json
from data_utils import create_input, iobes_iob,iob_iobes


class Model(object):
    #初始化模型参数
    def __init__(self, config):

        self.config = config

        self.lr = config["lr"]
        self.char_dim = config["char_dim"]
        self.lstm_dim = config["lstm_dim"]
        self.seg_dim = config["seg_dim"]

        self.num_tags = config["num_tags"]
        self.num_chars = config["num_chars"]#样本中总字数
        self.num_segs = 4

        self.global_step = tf.Variable(0, trainable=False)#trainable=False表示参数不会被更新
        self.best_dev_f1 = tf.Variable(0.0, trainable=False)
        self.best_test_f1 = tf.Variable(0.0, trainable=False)
        self.initializer = initializers.xavier_initializer()

        # add placeholders for the model
        #每个字用什么形状来表示
        self.char_inputs = tf.placeholder(dtype=tf.int32,
                                          shape=[None, None],#多少个字，每个字用多少维来表示
                                          name="ChatInputs")
        #0，1，2，3这四个值，所以shape=[4，用多少维来表示这四个数字]
        self.seg_inputs = tf.placeholder(dtype=tf.int32,
                                         shape=[None, None],
                                         name="SegInputs")
        #每个字多对应的标签，[标签的类别个数, 每个标签的维度数]
        self.targets = tf.placeholder(dtype=tf.int32,
                                      shape=[None, None],
                                      name="Targets")
        # dropout失活率
        self.dropout = tf.placeholder(dtype=tf.float32,
                                      name="Dropout")

        used = tf.sign(tf.abs(self.char_inputs))#将输入字符进行归到-1、0、1只用这三个数来表示
        length = tf.reduce_sum(used, reduction_indices=1)#统计每句话有多少个字
        self.lengths = tf.cast(length, tf.int32)#将字数转换为in32类型
        self.batch_size = tf.shape(self.char_inputs)[0]#输入字符的第一维是批次
        self.num_steps = tf.shape(self.char_inputs)[-1]#输入字符的最后一维作为细胞的步长，即表示这个字的向量的维度


        #Add model type by crownpku bilstm or idcnn
        self.model_type = config['model_type']
        #parameters for idcnn
        #膨胀系数
        self.layers = [
            {
                'dilation': 1
            },
            {
                'dilation': 1
            },
            {
                'dilation': 2
            },
        ]
        self.filter_width = 3#卷积核的宽度为3表示一次性只看这个字的id、seg、tag
        self.num_filter = self.lstm_dim#卷积核的个数
        self.embedding_dim = self.char_dim + self.seg_dim#词嵌入的维度数是每个字符的维度加上这个字的seg的维度
        self.repeat_times = 4#一共重复卷三次
        self.cnn_output_width = 0#卷积网络最终输出的维度数，相当于最终的通道数

        #词嵌入层
        embedding = self.embedding_layer(self.char_inputs, self.seg_inputs, config)

        if self.model_type == 'bilstm':
            # apply dropout before feed to lstm layer
            model_inputs = tf.nn.dropout(embedding, self.dropout)

            # bi-directional lstm layer
            model_outputs = self.biLSTM_layer(model_inputs, self.lstm_dim, self.lengths)

            # logits for tags
            self.logits = self.project_layer_bilstm(model_outputs)#三维[批次,最长句子的字数(步长),命名实体的个数]

        elif self.model_type == 'idcnn':
            #为了防止模型对一种或另一种表现形式的依赖性过强在输入网络之前对词嵌入进行dropout
            model_inputs = tf.nn.dropout(embedding, self.dropout)

            #将词嵌入输入进网络得到输出，是一个二维的
            model_outputs = self.IDCNN_layer(model_inputs)

            # logits for tags
            self.logits = self.project_layer_idcnn(model_outputs)#三维[批次,最长句子的字数(步长),命名实体的个数]

        else:
            raise KeyError

        # loss of the model
        self.loss = self.loss_layer(self.logits, self.lengths)#self.lengths是这一批句子中字数最多的句子的字数
        #将训练的损失收集起来
        tf.summary.scalar('loss',self.loss)
        with tf.variable_scope("optimizer"):
            optimizer = self.config["optimizer"]
            if optimizer == "sgd":
                self.opt = tf.train.GradientDescentOptimizer(self.lr)
            elif optimizer == "adam":
                self.opt = tf.train.AdamOptimizer(self.lr)
            elif optimizer == "adgrad":
                self.opt = tf.train.AdagradOptimizer(self.lr)
            else:
                raise KeyError

            # apply grad clip to avoid gradient explosion
            grads_vars = self.opt.compute_gradients(self.loss)
            capped_grads_vars = [[tf.clip_by_value(g, -self.config["clip"], self.config["clip"]), v]
                                 for g, v in grads_vars]  #config["clip"]梯度截断【-5,5】范围内
            self.train_op = self.opt.apply_gradients(capped_grads_vars, self.global_step)
            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

    def embedding_layer(self, char_inputs, seg_inputs, config):
        """
        :param char_inputs: one-hot encoding of sentence
        :param seg_inputs: segmentation feature
        :param config: wither use segmentation feature
        :return: [1, num_steps, embedding size],
        """
        #高:3 血:22 糖:23 和:24 高:3 血:22 压:25 char_inputs=[3,22,23,24,3,22,25]
        #高血糖 和 高血压 seg_inputs 高血糖=[1,2,3] 和=[0] 高血压=[1,2,3]  seg_inputs=[1,2,3,0,1,2,3]
        embedding = []
        with tf.variable_scope("char_embedding"), tf.device('/cpu:0'):
            #字符的词嵌入参数，最终学习到的就是每个字应该用什么向量来表示
            self.char_lookup = tf.get_variable(
                    name="char_embedding",
                    shape=[self.num_chars, self.char_dim],
                    initializer=self.initializer)
            #输入char_inputs='常' 对应的字典的索引/编号/value为：8
            #self.char_lookup=[2677*100]的向量，char_inputs字对应在字典的索引/编号/key=[1]

            #char_inputs是一个批次的句子数x这一批句子的最大字数，而每个元素是这个字所对应的id，因此
            #embedding_lookup筛选出的就是这一批句子每个字所对应的向量，然后将这个数据加入词嵌入列表
            #而这个数据是一个三维的[批次，这一批句子字数最多的句子的字数，每个字所表示的向量的维数]
            embedding.append(tf.nn.embedding_lookup(self.char_lookup, char_inputs))
            #self.embedding1.append(tf.nn.embedding_lookup(self.char_lookup, char_inputs))
            if config["seg_dim"]:
                with tf.variable_scope("seg_embedding"), tf.device('/cpu:0'):
                    self.seg_lookup = tf.get_variable(
                        name="seg_embedding",
                        #shape=[4*20]
                        shape=[self.num_segs, self.seg_dim],
                        initializer=self.initializer)
                    #添加进embedding的是一个三维的数据，[批次，这一批句子字数最多的句子的字数，每个seg所表示的向量的维数]
                    embedding.append(tf.nn.embedding_lookup(self.seg_lookup, seg_inputs))
            # 对这两个在最后一个维度上进行连接最后得到[批次，这一批句子字数最多的句子的字数，每个字所表示的向量的维数+每个seg所表示的向量的维数]
            embed = tf.concat(embedding, axis=-1)
        return embed

    def biLSTM_layer(self, model_inputs, lstm_dim, lengths, name=None):
        """
        :param lstm_inputs: [batch_size, num_steps, emb_size]
        :return: [batch_size, num_steps, 2*lstm_dim]
        """
        with tf.variable_scope("char_BiLSTM" if not name else name):
            lstm_cell = {}
            #创建前向和后向的LSTM层
            for direction in ["forward", "backward"]:
                with tf.variable_scope(direction):
                    lstm_cell[direction] = tf.contrib.rnn.LSTMCell(lstm_dim)
            outputs, final_states = tf.nn.bidirectional_dynamic_rnn(
                lstm_cell["forward"],
                lstm_cell["backward"],
                model_inputs,
                dtype=tf.float32,
                sequence_length=lengths)
        return tf.concat(outputs, axis=2)

    #IDCNN layer
    def IDCNN_layer(self, model_inputs,
                    name=None):
        """
        :param idcnn_inputs: [batch_size, num_steps, emb_size]
        :return: [batch_size, num_steps, cnn_output_width]
        """
        #tf.expand_dims会向tensor中插入一个维度，插入位置就是参数代表的位置（维度从0开始）。
        #将三维的升一个维度变成四维
        model_inputs = tf.expand_dims(model_inputs, 1)
        reuse = False#用于控制是否重用变量名
        if self.dropout == 1.0:
            reuse = True
        with tf.variable_scope("idcnn" if not name else name):

            #在进行膨胀卷积之前先进行一次普通卷积

            #卷积核，[高度，宽度，每个字所表示的维度（输入的通道数），卷积核的个数（输出的通道数）]
            #shape=[1*3*120*100]
            filter_weights = tf.get_variable(
                "idcnn_filter",
                shape=[1, self.filter_width, self.embedding_dim,
                       self.num_filter],
                initializer=self.initializer)

            """
            shape of input = [batch, in_height, in_width, in_channels]
            shape of filter = [filter_height, filter_width, in_channels, out_channels]
            """
            layerInput = tf.nn.conv2d(model_inputs,
                                      filter_weights,
                                      strides=[1, 1, 1, 1],
                                      padding="SAME",
                                      name="init_layer",use_cudnn_on_gpu=True)#原来False
            finalOutFromLayers = []#用于添加每次最后的膨胀卷积之后的输出
            totalWidthForLastDim = 0#累加每次最后的膨胀卷积之后的输出的维度，因为每卷一次维度会有所提升
            for j in range(self.repeat_times):#相当于卷积网络的层数
                for i in range(len(self.layers)):#相当于普通卷积的步长
                    #膨胀系数1,1,2
                    dilation = self.layers[i]['dilation']
                    isLast = True if i == (len(self.layers) - 1) else False#当所有膨胀系数的卷积都完成以后才算一次膨胀膨胀卷积
                    with tf.variable_scope("atrous-conv-layer-%d" % i,
                                           reuse=True
                                           if (reuse or j > 0) else False):
                        #w 卷积核的高度，卷积核的宽度，图像通道数，卷积核个数
                        w = tf.get_variable(
                            "filterW",
                            shape=[1, self.filter_width, self.num_filter,
                                   self.num_filter],
                            initializer=tf.contrib.layers.xavier_initializer())
                        if j==1 and i==1:
                            self.w_test_1=w
                        if j==2 and i==1:
                            self.w_test_2=w
                        b = tf.get_variable("filterB", shape=[self.num_filter])
#tf.nn.atrous_conv2d(value,filters,rate,padding,name=None）
    #除去name参数用以指定该操作的name，与方法有关的一共四个参数：
    #value：
    #指需要做卷积的输入图像，要求是一个4维Tensor，具有[batch, height, width, channels]这样的shape，具体含义是[训练时一个batch的图片数量, 图片高度, 图片宽度, 图像通道数]
    #filters：
    #相当于CNN中的卷积核，要求是一个4维Tensor，具有[filter_height, filter_width, channels, out_channels]这样的shape，具体含义是[卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]，同理这里第三维channels，就是参数value的第四维
    #rate：
    #要求是一个int型的正数，正常的卷积操作应该会有stride（即卷积核的滑动步长），但是空洞卷积是没有stride参数的，
    #这一点尤其要注意。取而代之，它使用了新的rate参数，那么rate参数有什么用呢？它定义为我们在输入
    #图像上卷积时的采样间隔，你可以理解为卷积核当中穿插了（rate-1）数量的“0”，
    #把原来的卷积核插出了很多“洞洞”，这样做卷积时就相当于对原图像的采样间隔变大了。
    #具体怎么插得，可以看后面更加详细的描述。此时我们很容易得出rate=1时，就没有0插入，
    #此时这个函数就变成了普通卷积。
    #padding：
    #string类型的量，只能是”SAME”,”VALID”其中之一，这个值决定了不同边缘填充方式。
    #ok，完了，到这就没有参数了，或许有的小伙伴会问那“stride”参数呢。其实这个函数已经默认了stride=1，也就是滑动步长无法改变，固定为1。
    #结果返回一个Tensor，填充方式为“VALID”时，返回[batch,height-2*(filter_width-1),width-2*(filter_height-1),out_channels]的Tensor，填充方式为“SAME”时，返回[batch, height, width, out_channels]的Tensor，这个结果怎么得出来的？先不急，我们通过一段程序形象的演示一下空洞卷积。
                        conv = tf.nn.atrous_conv2d(layerInput,
                                                   w,
                                                   rate=dilation,
                                                   padding="SAME")
                        self.conv_test=conv
                        conv = tf.nn.bias_add(conv, b)
                        conv = tf.nn.relu(conv)
                        if isLast:
                            finalOutFromLayers.append(conv)
                            totalWidthForLastDim += self.num_filter
                        layerInput = conv
            finalOut = tf.concat(axis=3, values=finalOutFromLayers)#将每次膨胀卷积在输出通道上连接起来
            keepProb = 1.0 if reuse else 0.5
            finalOut = tf.nn.dropout(finalOut, keepProb)
            #Removes dimensions of size 1 from the shape of a tensor.
                #从tensor中删除所有大小是1的维度

                #Given a tensor input, this operation returns a tensor of the same type with all dimensions of size 1 removed. If you don’t want to remove all size 1 dimensions, you can remove specific size 1 dimensions by specifying squeeze_dims.

                #给定张量输入，此操作返回相同类型的张量，并删除所有尺寸为1的尺寸。 如果不想删除所有尺寸1尺寸，可以通过指定squeeze_dims来删除特定尺寸1尺寸。
            finalOut = tf.squeeze(finalOut, [1])
            finalOut = tf.reshape(finalOut, [-1, totalWidthForLastDim])#将前三个合在一起，totalWidthForLastDim=重复卷积的次数x每次卷积的卷积核个数
            self.cnn_output_width = totalWidthForLastDim
            return finalOut

    def project_layer_bilstm(self, lstm_outputs, name=None):
        """
        hidden layer between lstm layer and logits
        :param lstm_outputs: [batch_size, num_steps, emb_size]
        :return: [batch_size, num_steps, num_tags]
        """
        with tf.variable_scope("project"  if not name else name):
            with tf.variable_scope("hidden"):
                W = tf.get_variable("W", shape=[self.lstm_dim*2, self.lstm_dim],
                                    dtype=tf.float32, initializer=self.initializer)

                b = tf.get_variable("b", shape=[self.lstm_dim], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())
                output = tf.reshape(lstm_outputs, shape=[-1, self.lstm_dim*2])
                hidden = tf.tanh(tf.nn.xw_plus_b(output, W, b))

            # project to score of tags
            with tf.variable_scope("logits"):
                W = tf.get_variable("W", shape=[self.lstm_dim, self.num_tags],
                                    dtype=tf.float32, initializer=self.initializer)

                b = tf.get_variable("b", shape=[self.num_tags], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())

                pred = tf.nn.xw_plus_b(hidden, W, b)

            return tf.reshape(pred, [-1, self.num_steps, self.num_tags])

    #Project layer for idcnn by crownpku
    #Delete the hidden layer, and change bias initializer
    def project_layer_idcnn(self, idcnn_outputs, name=None):
        """
        :param lstm_outputs: [batch_size, num_steps, emb_size]
        :return: [batch_size, num_steps, num_tags]
        """
        with tf.variable_scope("project"  if not name else name):

            # project to score of tags
            with tf.variable_scope("logits"):
                #全连接
                W = tf.get_variable("W", shape=[self.cnn_output_width, self.num_tags],
                                    dtype=tf.float32, initializer=self.initializer)

                b = tf.get_variable("b",  initializer=tf.constant(0.001, shape=[self.num_tags]))

                pred = tf.nn.xw_plus_b(idcnn_outputs, W, b)

            return tf.reshape(pred, [-1, self.num_steps, self.num_tags])

    def loss_layer(self, project_logits, lengths, name=None):#project_logits[60,100,num_tag]
        """
        calculate crf loss
        :param project_logits: [1, num_steps, num_tags]
        :return: scalar loss
        """
        with tf.variable_scope("crf_loss"  if not name else name):
            small = -1000.0
            # pad logits for crf loss
            start_logits = tf.concat(#small * tf.ones(shape=[self.batch_size, 1, self.num_tags])这么多全部都是-1000.0的张量
                [small * tf.ones(shape=[self.batch_size, 1, self.num_tags]), tf.zeros(shape=[self.batch_size, 1, 1])], axis=-1)
            #start_logits=[self.batch_size, 1, self.num_tags+1][60,1,]
            pad_logits = tf.cast(small * tf.ones([self.batch_size, self.num_steps, 1]), tf.float32)#[60,100,1]
            logits = tf.concat([project_logits, pad_logits], axis=-1)#[60,100,num_tag+1]
            logits = tf.concat([start_logits, logits], axis=1)#[60,101,num_tag+1]
            targets = tf.concat(
                [tf.cast(self.num_tags*tf.ones([self.batch_size, 1]), tf.int32), self.targets], axis=-1)#[self.batch_size,num_tag+1]

            self.trans = tf.get_variable(
                "transitions",
                shape=[self.num_tags + 1, self.num_tags + 1],
                initializer=self.initializer)
            #crf_log_likelihood在一个条件随机场里面计算标签序列的log-likelihood
            #inputs: 一个形状为[batch_size, max_seq_len, num_tags] 的tensor,
            #一般使用BILSTM处理之后输出转换为他要求的形状作为CRF层的输入.
            #tag_indices: 一个形状为[batch_size, max_seq_len] 的矩阵,其实就是真实标签.
            #sequence_lengths: 一个形状为 [batch_size] 的向量,表示每个序列的长度.
            #transition_params: 形状为[num_tags, num_tags] 的转移矩阵
            #log_likelihood: 标量,log-likelihood
            #transition_params: 形状为[num_tags, num_tags] 的转移矩阵
            log_likelihood, self.trans = crf_log_likelihood(
                inputs=logits,
                tag_indices=targets,
                transition_params=self.trans,
                sequence_lengths=lengths+1)
            return tf.reduce_mean(-log_likelihood)

    def create_feed_dict(self, is_train, batch):
        """
        :param is_train: Flag, True for train batch
        :param batch: list train/evaluate data
        :return: structured data to feed
        """
        _, chars, segs, tags = batch
        feed_dict = {
            self.char_inputs: np.asarray(chars),
            self.seg_inputs: np.asarray(segs),
            self.dropout: 1.0,
        }
        if is_train:
            feed_dict[self.targets] = np.asarray(tags)
            feed_dict[self.dropout] = self.config["dropout_keep"]
        return feed_dict

    def run_step(self, sess, is_train, batch):
        """
        :param sess: session to run the batch
        :param is_train: a flag indicate if it is a train batch
        :param batch: a dict containing batch data
        :return: batch result, loss of the batch or logits
        """
        feed_dict = self.create_feed_dict(is_train, batch)
        if is_train:
            global_step, loss, _ = sess.run([self.global_step, self.loss, self.train_op],feed_dict)
            return global_step, loss
        else:
            lengths, logits = sess.run([self.lengths, self.logits], feed_dict)
            return lengths, logits

    def decode(self, logits, lengths, matrix):
        """
        :param logits: [batch_size, num_steps, num_tags]float32, logits
        :param lengths: [batch_size]int32, real length of each sequence
        :param matrix: transaction matrix for inference
        :return:
        """
        # 使用维特比算法计算最终的标签
        paths = []
        small = -1000.0
        start = np.asarray([[small]*self.num_tags +[0]])
        for score, length in zip(logits, lengths):
            score = score[:length]
            pad = small * np.ones([length, 1])
            logits = np.concatenate([score, pad], axis=1)#进行拼接
            logits = np.concatenate([start, logits], axis=0)
            path, _ = viterbi_decode(logits, matrix)#使用维特比算法进行解码输出,是一个向量，第一个元素是命名实体的类别个数，其后是这个句子中每个字的命名实体的预测结果
            paths.append(path[1:])
        return paths

    #模型的验证函数
    def evaluate(self, sess, data_manager, id_to_tag):
        """
        :param sess: session  to run the model
        :param data: list of data
        :param id_to_tag: index to tag name
        :return: evaluate result
        """
        results = []
        trans = self.trans.eval()#转移矩阵的验证
        for batch in data_manager.iter_batch():
            strings = batch[0]
            tags = batch[-1]
            lengths, scores = self.run_step(sess, False, batch)#将验证集进行前向计算得到字数和输出
            batch_paths = self.decode(scores, lengths, trans)#得到这一批句子中每个字的预测的命名实体的id
            for i in range(len(strings)):#遍历每个句子
                result = []
                string = strings[i][:lengths[i]]#这句话的前i个字，实际上就是这批句子的每句话
                gold = iobes_iob([id_to_tag[int(x)] for x in tags[i][:lengths[i]]])#将真实的tag转化为iobes的形式
                pred = iobes_iob([id_to_tag[int(x)] for x in batch_paths[i][:lengths[i]]])#将预测的tag转化为iobes的形式
                for char, gold, pred in zip(string, gold, pred):
                    result.append(" ".join([char, gold, pred]))#将句子等打包作为输出
                results.append(result)
        return results

    def evaluate_line(self, sess, inputs, id_to_tag):
        trans = self.trans.eval(session=sess)
        lengths, scores = self.run_step(sess, False, inputs)
        batch_paths = self.decode(scores, lengths, trans)
        tags = [id_to_tag[idx] for idx in batch_paths[0]]
        return result_to_json(inputs[0][0], tags)