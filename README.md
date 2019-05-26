# Named-Entity-Recognition-Based-on-Hollow-Neural-Network-and-Bidirectional-LSTM-CRF-Model
基于膨胀神经网络及双向LSTM+CRF模型进行命名实体识别

模型的结构简要图：

![image](https://github.com/lihuaqiang0101/Named-Entity-Recognition-Based-on-Hollow-Neural-Network-and-Bidirectional-LSTM-CRF-Model/blob/master/images/%E6%89%B9%E6%B3%A8%202019-05-26%20172209.jpg)


网络的主要架构:


![image](https://github.com/lihuaqiang0101/Named-Entity-Recognition-Based-on-Hollow-Neural-Network-and-Bidirectional-LSTM-CRF-Model/blob/master/images/20190524175823413.png)


这是了双向LSTM的字嵌入,Li代表I和它的左上下文，ri代表I和它的右上下文。将这两个向量连接在一起，就可以在其上下文Ci中表示单词I。
其中圆代表观察变量，菱形是确定性函数，双圆是随机变量。

使用膨胀卷积神经网络的优势：

![image](https://github.com/lihuaqiang0101/Named-Entity-Recognition-Based-on-Hollow-Neural-Network-and-Bidirectional-LSTM-CRF-Model/blob/master/images/2540794-96ee339fb182240a.png)

它在不降低图像分辨率的基础上聚合图像中不同尺寸的上下文信息，并且扩大了感受野的范围。
在NER问题中：视野更大的好处是能够更多的提取上下文的特征更广了，因为从b的9个点相对于a的9个点的视野范围就更大了，比如原来的卷积核大小是3，只能卷3个字，看这3个字有没有相关性，是不是一个命名实体，但如果有些命名实体的6个字的，那么就识别不准，将它放大以后一下子就可以看6个字了，再放大一倍一下子又可以看9个字了。但是这里并不冲突，因为最后会全部综合起来，它的最大的优点就在于此，也就是可以解决更长层的依赖。

测试结果：
请输入测试句子:全身皮肤及粘膜无黄染，未触及肿大浅表淋巴结。
{'string': '全身皮肤及粘膜无黄染,未触及肿大浅表淋巴结。', 'entities': [{'word': '全身', 'start': 0, 'end': 2, 'type': 'REG'}, {'word': '皮肤', 'start': 2, 'end': 4, 'type': 'ORG'}, {'word': '粘膜', 'start': 5, 'end': 7, 'type': 'ORG'}, {'word': '黄染', 'start': 8, 'end': 10, 'type': 'SGN'}, {'word': '肿大', 'start': 14, 'end': 16, 'type': 'SYM'}, {'word': '浅表淋巴结', 'start': 16, 'end': 21, 'type': 'ORG'}]}
