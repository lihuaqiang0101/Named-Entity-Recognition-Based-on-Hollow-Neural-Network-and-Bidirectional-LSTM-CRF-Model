# Named-Entity-Recognition-Based-on-Hollow-Neural-Network-and-Bidirectional-LSTM-CRF-Model
基于膨胀神经网络及双向LSTM+CRF模型进行命名实体识别

模型的结构简要图：

![image](https://github.com/lihuaqiang0101/Named-Entity-Recognition-Based-on-Hollow-Neural-Network-and-Bidirectional-LSTM-CRF-Model/blob/master/images/%E6%89%B9%E6%B3%A8%202019-05-26%20172209.jpg)


网络的主要架构:


![image](https://github.com/lihuaqiang0101/Named-Entity-Recognition-Based-on-Hollow-Neural-Network-and-Bidirectional-LSTM-CRF-Model/blob/master/images/20190524175823413.jpg)

这是了双向LSTM的字嵌入,Li代表I和它的左上下文，ri代表I和它的右上下文。将这两个向量连接在一起，就可以在其上下文Ci中表示单词I。
其中圆代表观察变量，菱形是确定性函数，双圆是随机变量。
