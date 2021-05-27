## Attention Is All You Need总结



## Transformer(self-attention) 优势

- computational complexity低
- 可以parallelize计算
- 方便学到 long-range dependencies



## 结构

<img src="https://github.com/lxy5513/typora-image-depositories/blob/main/image-20210526172819174.png" />

### Encoder

- 目的

  将输入序列的符号表达$(x_1,...,x_n)$映射为连续的表达$z=(z_1,...,z_n)$. 并且每个word带有对其他word的相关信息。

- 流程

  Encoder是有6个同样的层组成的栈结构，每一层有两个子层。第一个是multi-head self-attention 机制，第二个是简单的position-wised 全连接前馈神经网络。我们对每一个子层使用LayerNorm和residual connection。为了方便这些残差连接，每个sub-layers和embedding layers都产生$d_{model}$ = 512维度的输出。

  > BN对同一mini-batch中对不同特征进行标准化（**纵向规范化**：每一个特征都有自己的分布），受限于batch size，难以处理动态神经网络中的变长序列的mini-bach。
  >
  > RNNs不同时间步共享权重参数，使得RNNs可以处理不同长度的序列，RNNs使用 Layer Normalization 对不同时间步进行标准化（**横向标准化**：每一个时间步都有自己的分布），从而可以处理单一样本、变长序列，而且 训练和测试处理方式一致。

#### 第一层

**第一步：** embedding:  将word转换为词向量。

- Input embedding

  将一个句子的每个单词都转化为维度$d_{model}$=512的向量。

- Positional Encoding embedding

  给每个输入的词向量加入位置信息(维度$d_{model}$=512)

  > 对编码的需求
  >
  > - 需要体现同一单词在不同位置的区别。
  >
  > - 需要体现一定的先后次序，并且在一定范围内的编码差异不应该依赖于文本的长度，具有一定的不变性。
  >
  > - 需要有值域的范围限制。
  >
  >   <img src="https://github.com/lxy5513/typora-image-depositories/blob/main/image-20210526181605047.png" alt="image-20210526181605047" style="zoom:33%;" />

**第二步：** self-attention 每个向量创建三个向量（Q，K，V），其维度是64 （512/64=8）。

<img src="https://github.com/lxy5513/typora-image-depositories/blob/main/image-20210526192035872.png" alt="image-20210526192035872" style="zoom:33%;" />

- 对于每个单词，将自己和其他单词的$q_i，k_i$分别相乘，并scale（除以$\sqrt{d_k}$), 然后使用softmax归一化， 最后乘以这个单词对应的V向量。

  ​										$$Attention(Q,K,V) = softmax(\frac{QK^{T}}{\sqrt{d_k}})V$$

**第三步:** multi-head attention

​		将第二部独立重复8次，得到8个head, 然后将其concatenate , 得到512维度的向量。

> 好处：
>
> 1. 拓展了模型关注不同位置的能力
> 2. 赋予attention layer层多个"表示子空间表示".



#### 第二层

pointwise全连接前馈神经网络

<img src="https://miro.medium.com/max/1280/0*-fdpoPbN-BHAMRnr.gif" alt="0*-fdpoPbN-BHAMRnr" style="zoom:67%;" />

### Decoder

**目的**

- 将编码后的向量经一系列操作后转换成文本序列。

**结构**

![0*u8nSpT8Z8ITwzNLV](https://miro.medium.com/max/960/0*u8nSpT8Z8ITwzNLV.gif)

The decoder is autoregressive自动回归, it begins with a 【start】 token, and it takes in a list of 该句子之前的输出 as inputs, as well as the encoder outputs that contain the attention information from the input. The decoder 停止 decoding when it generates a token as an 【output】.

Decoder同样是6个相同层组成的栈结构，每层有三个子层.
第一个子层上加入mask, masked multi-head attention. 表明我们只能attend到前面已经翻译过的词语，因为翻译过程中我们当前还并不知道下一个输出词语。这种masking，加上output embeddings的位置偏移，确保了对位置i的预测只能依赖于小于i位置的已知输出

​																							<img src="https://miro.medium.com/max/1090/0*0pqSkWgSPZYr_Sjx.png" alt="0*0pqSkWgSPZYr_Sjx" style="zoom:70%;" />

第二个子层称为==Encoder-Decoder Attention==，Q来自第一个子层输出，K和V则来自encoder的输出.

第三个子层是pointwise全连接前馈神经网络。



###  Final  Output 

在final point wise feedforward之后通过一个linear layer，作为一个**分类器**。这个分类数量是你所有可能词的总数，然后经过**softmax layer**, 得到每个词的最终概率，我们去概率最大的作为输出。



### 说明：

本文参考一下文章：



