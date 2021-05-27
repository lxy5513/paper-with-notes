# ViT 解析与代码实现

AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE

1. [Introduction](https://amaarora.github.io/2021/01/18/ViT.html#introduction)
2. [Key Contributions](https://amaarora.github.io/2021/01/18/ViT.html#key-contributions)
3. [The Vision Transformer](https://amaarora.github.io/2021/01/18/ViT.html#the-vision-transformer)
4. [Patch Embeddings](https://amaarora.github.io/2021/01/18/ViT.html#patch-embeddings)
5. [`[cls\]` token & Position Embeddings](https://amaarora.github.io/2021/01/18/ViT.html#cls-token--position-embeddings)
6. [The Transformer Encoder](https://amaarora.github.io/2021/01/18/ViT.html#the-transformer-encoder)
7. [The Vision Transformer in PyTorch](https://amaarora.github.io/2021/01/18/ViT.html#the-vision-transformer-in-pytorch)



## Introduction

ViT是第一篇论文，使用transformer结构来实现image classification, 在使用大数据集做预训练之后，能达到state-of-the-art效果的.



## Key Contributions

- an existing architecture (Transformers), to the field of Computer Vision
- It is the **training method** and the **dataset used to pretrain the network**
- get excellent results compared to SOTA (State of the Art) on ImageNet.



## The Vision Transformer

We will be using a top down approach to understand the **Visual Transformer** architecture. We will first start by looking at the overall architecture and then dig deeper into each of the ==five steps== in the overall architecture.

As an overall method, from the paper:

> 我们将一张图片分成固定大小的patches, 线性的embeding, 并且加上position embeddings, 将这一系列vectors输入到标准的Transformer encoder上， 为了实现分类任务，我们在序列上添加一个额外的可学习的“classification token“，



![fig-1 The Model Overview](https://amaarora.github.io/images/ViT.png)fig-1 The Model Overview



The overall architecture can be described easily in five simple steps below:

1. Split an input image into patches.
2. Get linear embeddings (representation) from each patch referred to as **Patch Embeddings**.
3. Add position embeddings and **==a `[cls]` token==** to each of the Patch Embeddings.
4. Pass through a **Transformer Encoder** and get the output values for each of the `[cls]` tokens.
5. Pass **the representations of `[cls]` **tokens through a**`MLP Head`**to 得到分类预测.

> Transformer Encoder 内部的MLP和 用于分类任务的 MLP是不相同的。





![fig-2 Simplified Model Overview](https://amaarora.github.io/images/vit-01.png)fig-2 Simplified Model Overview

由上图,  想象下我们将3通道的， 224*224的RGB图片通过5个步骤进行分类。

The **first step** is to create patches all over the image of patch size `16 x 16`. Thus we create `14 x 14` or `196` such patches. We can have these patches in **a straight line** as in `fig-2` （其中第一个patch是图片的左上方， 最后一个patch位于图片的右下方）。.As can be seen from the figure, the patch size is `3 x 16 x 16` where `3` represents the number of channels (RGB).

In the **second step**, we pass these patches through a **linear projection layer** to get `1 x 768` long vector representation for each of the image patches （如图片上的紫色部分）. In the paper, 作者将这些patch representation 叫做 **Patch Embeddings**. Can you guess what’s the size of this patch embedding matrix? It’s `196 x 768`. Because we had a total of `196` patches and each patch has been represented as a `1 x 768` long vector. Therefore, the total size of the patch embedding matrix is `196 x 768`.

> You might wonder why is the vector length `768`? Well, `3 x 16 x 16 = 768`. So, we are not really losing any information as step of this process of getting these **patch embeddings**.

In the **third step**, we take this patch embedding matrix of size `196 x 768` and similar to [BERT](https://arxiv.org/abs/1810.04805), the authors prepend a `[cls]` token to this sequence of embedded patches and then add **Position Embeddings**. As can be seen from `fig-2`, the size of the **Patch Embeddings** becomes `197 x 768` after adding the `[cls]` token and also the size of the **Position Embeddings** is `197 x 768`.

> Why do we add this class token and position embeddings? 
>
> the `[class]` tokens 作为一个特别的tokens， 其经过 the `Transformer Encoder` 的输出，可以看作整个图片patch的表达，对于positional embedding， 这么做的目的是来维持patches的位置信息，因为不像CNNs中有顺序的patches, Transformer model自己并不知道关于patches的顺序信息。所以我们手动的插入其patches的绝对或者相对位置信息。

In the **fourth step**, we pass these preprocessed patch embeddings with positional information and 插入 `[cls]` token to the **Transformer Encoder** and get the ==learned representations of the `[cls]` token==. Thus, the output frpm the Transformer Encoder would be of size `1 x 768` which is then fed to the `MLP Head` (which is nothing but a Linear Layer) as part of the final **fifth step** to get class predictions.

Having looked at the overall architecture, we will now look at the individual steps in detail in the following sections.



## Patch Embeddings

In this section we will be looking at **steps one and two** in detail. That is the process of getting patch embeddings from an input image.



![fig-3 Patch Embeddings](https://amaarora.github.io/images/vit-02.png)`fig-3 Patch Embeddings`

我们通过2D的卷积操作来实现patch embedding.

So far in the blog post I have mentioned that the way we get patch embeddings from an input image is to first split an image into fixed-size patches and then linearly embed each one of them using a **linear projection layer** as shown in `fig-2`.

But, it is actually possible to combine both steps into a single step using **2D Convolution** operation. It is also better from an implementation perspective to do it this way as our GPUs are optimized to perform the convolution operation and it takes away the need to first split an image into patches. Let’s see why this works?

If we set the the number of `out_channels` to `768`, and both `kernel_size` & `stride` to `16`, then as shown in `fig-3`, once we perform the convolution operation (where the 2-D Convolution has kernel size `3 x 16 x 16`), we can get the **Patch Embeddings** matrix of size `196 x 768` like below:

```python
# input image `B, C, H, W`
x = torch.randn(1, 3, 224, 224)
# 2D conv
conv = nn.Conv2d(3, 768, 16, 16) #  out_channels` to 768, and both kernel_size & stride to 16
conv(x).reshape(-1, 196).transpose(0,1).shape

>> torch.Size([196, 768])
```



## `[cls]` token & Position Embeddings

In this section, let’s look at the **third step** in more detail. In this step, we prepend `[cls]` tokens and add **Positional Embeddings** to the **Patch Embeddings**.

From the paper:

> 相似于BERT‘s `[class] token`, 我们在embedded patches中嵌入一个可学习的embedding， 在经过Transformer Encoder后，用**ZLO**作为the image representation, 不管是在pre-training还是fine-tuning中， a classification head is attached to **ZL0**.

> Position embeddings are also added to the patch embeddings to retain positional information. We use standard ==learnable 1D position embeddings== and the resulting sequence of embedding vectors serves as input to the encoder.

This process can be easily visualized as below:



![fig-4 `CLS` token and Position Embeddings](https://amaarora.github.io/images/vit-03.png)fig-4 `CLS` token and Position Embeddings



As can be seen from `fig-4`, the `[cls]` token is a vector of size `1 x 768`. We **prepend** it to the **Patch Embeddings**, thus, the updated size of **Patch Embeddings** becomes `197 x 768`.

Next, we add **Positional Embeddings** of size `197 x 768` to the **Patch Embeddings** with `[cls]` token to get **combined embeddings** which are then fed to the `Transformer Encoder`. This is a pretty standard step that comes from the original Transformer paper - [Attention is all you need](https://arxiv.org/abs/1706.03762).

> Note that the Positional Embeddings and `cls` token vector is nothing fancy but rather just a trainable `nn.Parameter` matrix/vector. 所以说positional embeddings 和 `cls` token都是可训练的。



## The Transformer Encoder

In this section, we will be looking into the **Transformer Encoder** from `fig-1` in detail. As shown in `fig-1`, the Transformer Encoder consists of alternating layers of **Multi-Head Attention** and **MLP** blocks. Also, as shown in `fig-1`, **Layer Norm** is used before every block and residual connections after every block.

A single layer/block of the **Transformer Encoder** can be visualized as below:



![fig-5 Transformer Encoder Block](https://amaarora.github.io/images/vit-07.png)																												fig-5 Transformer Encoder Block



The first layer of the **Transformer Encoder** accepts **combined embeddings** of shape`197 x 768` as input. For all subsequent layers, the inputs are the outputs `Out` matrix of shape `197 x 768` from the previous layer of the **Transformer Encoder**.  the **Transformer Encoder** of the ViT-Base architecture 总共有12个layer。

在一个layer内部，the inputs首先通过一个Layer Norm， 然后再fed to **Multi-Head Attention** block中。

在**Multi-Head Attention**中， the inputs首先用**Linear layer**转化成`197 x 2304 (768*3)`形状。接下来reshape this **qkv** matrix into `197 x 3 x 768` where each of the three matrices of shape `197 x 768` represent the **q**, **k** and **v** matrices. These **q**, **k** and **v** matrices are further reshaped to `12 x 197 x 64` to represent the 12 attention heads. Once we have the **q**, **k** and **v** matrices, we finally perform the attention operation inside the **Multi-Head Attention** block which is given by the equation:

​																											$$Attention(qkv) = softmax(\frac {qk^T}{\sqrt{d_k}} ) \times v$$

eq-1 Attention



Once we get the outputs from the **Multi-Head Attention** block, these are added to the inputs (skip connection) to get the final outouts that again get passed to **Layer Norm** before being fed to the **MLP** Block.

The **MLP**, is a Multi-Layer Perceptron block consists of **two linear layers and a GELU non-linearity激活函数**. The outputs from the **MLP** block are again added to the inputs (skip connection) to get the final output from one layer of the **Transformer Encoder**.



let’s now zoom out and look at the complete **Transformer Encoder**.



![fig-6 Transformer Encoder](https://amaarora.github.io/images/vit-06.png)																													`fig-6 Transformer Encoder`



As can be seen from the image above, a single **Transformer Encoder** consists of 12 layers. The outputs from the first layer are fed to the second layer, outputs from the second fed to the third until we get the final outputs from the 12th layer of the **Transformer Encoder** which are then fed to the **MLP Head** to get class predictions. The above image is another way to summarize `fig-1`.



---



## The Vision Transformer in PyTorch(代码实现)

Having understood the Vision Transformer Architecture in great detail, let’s now look at the code-implementation and understand how to implement this architecture in PyTorch. We will be referencing the code from [timm](https://github.com/rwightman/pytorch-image-models) to explain the implementation. The code below has been directly copied from [here](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py).

We will build Vision Transformer using a bottom-up approach. We will take what we have learnt so far and start implementing the overall architecture piece-by-piece. First things first, how do get **Patch Embeddings**?

```python
class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    		将(N, 3, 224, 224)的图片通过卷积转化成（N，196，768）shape
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)  # 16*16
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x
```

As we know, we use a **2-D Convolution** where `stride`, `kernel_size` are set to `patch_size`. Thus, that is exactly what the class above does. We set `self.proj` to be a `nn.Conv2d` which goes from 3-channels to `768` and to get `196 x 768` patch embedding matrix.

```python
patch_embed = PatchEmbed()
x = torch.randn(1, 3, 224, 224)
patch_embed(x).shape 
>> torch.Size([1, 196, 768])
```

Okay, so that’s that. It is also pretty easy to implement the **MLP** Block inside the **Transformer Encoder** below:

```python
class Mlp(nn.Module):
		"""位于multi-head attention之后"""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()  # 激活函数
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
```

Basically, it consists of two layers and a `GELU` activation layer. There isn’t a lot happening in this class and is pretty easy to implement. 

Next, we implement `Attention` as below:

```python
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights, 一般来说是除以根号64
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape 
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # @代表矩阵相乘
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
 
>>> attention = Attention(768)
>>> embedded_patches = torch.zeros((1, 197, 768))
>>> z = attention(embedded_pathches)
>>> z.shape  # (1, 197, 768)
```

As described inside the **Multi-Head Attention** block, we use a Linear layer to get the **qkv** matrix. Also, we apply the attention operation inside the `forward` method above like so:    $Attention(qkv) = softmax(\frac {qk^T}{\sqrt{d_k}} )$

```python
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
```

The above code implements `eq-1`. Since we have already implemented the **Attention** Layer and **MLP** block, 

以下是快速的实现单层的 the **Transformer Encoder**. As we already know from `fig-5`,  a single `Block` consists of Layer Norm, Attention and MLP block.

```python
class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        # Stochastic Depth（随机深度网络），随机丢掉一些层.
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))  # 残差连接
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
```

As can be seen in the `forward` method above, this `Block` accepts inputs `x`, passes them through `self.norm1` which is `LayerNorm` followed by the attention operation. Next, we normalize the output after the attention operation again before passing through `self.mlp` followed by `Dropout` to get the outputs `Out` matrix from this single block as in `fig-5`.



整体的代码实现:

```python
class VisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_classes = num_classes  # 分类数量
        self.num_features = self.embed_dim = embed_dim  # paper中为768(16x16x3)
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches  # paper中为196(224*224/16/16)
        # nn.Parameter 类型转换函数，将一个不可训练的类型Tensor转换成可以训练的类型parameter并将这个parameter绑定到这个module里面
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # 扩充cls token
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0]  # 返回cls token代表的representation

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x
```

I leave it as an exercise to the reader to understand the implementation of the Vision Transformer. It merely brings all the pieces together and performs the steps as described in `fig-2`.





## 说明：

本文借鉴参考以下文章：

http://jalammar.github.io/illustrated-transformer/

https://amaarora.github.io/2021/01/18/ViT.html

https://arxiv.org/abs/2010.11929





