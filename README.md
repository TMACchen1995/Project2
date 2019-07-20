# Project2
 Fine-grained sentiment analysis

整体是借鉴的当时的参赛大佬的，当时他的f1_sroce是0.70201. 他的github: https://github.com/pengshuang/AI-Comp .
他用的是对每个类分别建模，用20个模型， 每个模型训练20epoch，确实取得了不错的效果但是即使是在google colab上运行，一个epoch还是要5分多钟，训练完20个模型花费的时间实在太长。 所以就想在此基础上改成80分类。此外用fasttext的pre-trained的词嵌入（原作者没有用pre-trained）。 看看是否对最终结果有所提升。

由于硬件限制，代码都放在googel colab上运行。

依次运行Preprocess.py -> JoinAttLayer.py -> classifier_rcnn.py -> train_model_v2.py 
