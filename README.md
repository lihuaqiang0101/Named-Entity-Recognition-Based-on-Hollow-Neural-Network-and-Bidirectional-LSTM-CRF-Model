# Named-Entity-Recognition-Based-on-Hollow-Neural-Network-and-Bidirectional-LSTM-CRF-Model
基于膨胀神经网络及双向LSTM+CRF模型进行命名实体识别

模型的结构简要图：

![image](https://github.com/lihuaqiang0101/Named-Entity-Recognition-Based-on-Hollow-Neural-Network-and-Bidirectional-LSTM-CRF-Model/blob/master/images/%E6%89%B9%E6%B3%A8%202019-05-26%20172209.jpg)


网络的主要架构:


![image](https://github.com/lihuaqiang0101/Named-Entity-Recognition-Based-on-Hollow-Neural-Network-and-Bidirectional-LSTM-CRF-Model/blob/master/images/20190524175823413.png)


这是了双向LSTM的字嵌入,Li代表I和它的左上下文，ri代表I和它的右上下文。将这两个向量连接在一起，就可以在其上下文Ci中表示单词I。
其中圆代表观察变量，菱形是确定性函数，双圆是随机变量。

测试结果：
请输入测试句子:全身皮肤及粘膜无黄染，未触及肿大浅表淋巴结。
{'string': '全身皮肤及粘膜无黄染,未触及肿大浅表淋巴结。', 'entities': [{'word': '全身', 'start': 0, 'end': 2, 'type': 'REG'}, {'word': '皮肤', 'start': 2, 'end': 4, 'type': 'ORG'}, {'word': '粘膜', 'start': 5, 'end': 7, 'type': 'ORG'}, {'word': '黄染', 'start': 8, 'end': 10, 'type': 'SGN'}, {'word': '肿大', 'start': 14, 'end': 16, 'type': 'SYM'}, {'word': '浅表淋巴结', 'start': 16, 'end': 21, 'type': 'ORG'}]}
