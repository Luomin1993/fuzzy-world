## 基于视觉与语义融合实现推理的方法探索

-------------------------------------------------------------------
>在当前的研究中计算机视觉和自然语言的结合越来越得到青睐,我们也尝试提出一种基于视觉与语义融合实现推理的方法,并且构建了工具来设定任务环境度量这种方法的有效性,也设定baseline model并讨论了实验结果。

----------------------------------------------------------
### 语义+视觉的理论基础
在大脑认知中,语义信息$S_i$和视觉信息$V_i$都有一一对应的表征,这样一来就可以构成一个个**pair**形式的**key-value**对$< S_i,V_i>$,我们希望能够找到一个公共表征$\hat{C}_i$可以很好地涵盖$< S_i,V_i>$的信息(相当于将语义信息$S_i$和视觉信息$V_i$映射到一个公共空间里进行度量).

这个想法是容易实现的,我们只需要分别训练${C}_s = f_1(S)$和${C}_v = f_2(V)$这两个映射且约束$\min {||C_s-C_v||}$即可,也有很多其他的方法可以实现,但是我们的终极目的是希望利用公共表征$\hat{C}_i$来实现逻辑推理.

<img src="/App/FuzzyWorld/DatGraph/space.jpg" width="300" hegiht="313" align=center />

在上图中表达的是如何将浅层的语义信息$S_i$和视觉信息$V_i$映射到高阶的场景逻辑信息,我们可以将多个零碎的$< S_i,V_i>$拼凑为完整的**Scene State Graph**$G_s(O,P)$.但是逻辑推理需要多个时间步的状态$\{G^{(t)}_s\}$来获取,比如在$G^{(t)}_s$中有改变$open(light)$,在$G^{(t+1)}_s$中观察得温度上升$up(T)$,就可以获得一条一阶逻辑$open(light) \Rightarrow up(T)$.

--------------------------------------------------------
### XWorld
这个是百度算法团队今年五月开源的一套c++库,可以在其中完成 **Robotics Navigation** 的实验.在这个工具上Baidu实现了以下网络的Agent来完成依照指令寻找物体的任务:

<img src="/App/FuzzyWorld/DatGraph/baidu.jpg" width="420" hegiht="313" align=center />

其 **Policy Gradient** 如下,我们来好好分析一下,首先语义信息$l$和视觉信息$o$都融合进了网络,其中$A^{[t]}=r^{[t]}+\gamma v_{\theta}({o}^{[t+1]},{h}^{[t]},{l}) - v_{\theta}({o}^{[t]},{h}^{[t-1]},{l})$含有网络参数$\theta$但是实际上并未参与梯度求导:

$$-\mathbb{E}_{{s}^{[t]},a^{[t]},{l}} =[(  \underbrace{ \nabla_{\theta}\log\pi_{\theta}(a^{[t]}|{o}^{[t]},{h}^{[t-1]},{l}) }_{Policy项拟合策略}  +   \underbrace{  \eta\nabla_{\theta}v_{\theta}({o}^{[t]},{h}^{[t-1]},{l})}_{Value项评估价值}   )   \underbrace{A^{[t]}}_{Award项最大化} +  \underbrace{  \kappa\nabla_{\theta}\mathcal{E}(\pi_{\theta})}_{Norm项防爆炸}  ]$$

------------------------------------------------------
### 一个类XWorld的小工具
由于在XWorld中难以实现需要逻辑推导完成的任务,因此基于Pyglet开发了一个可以定义3D环境和任务的小工具**FuzzyWorld**,目前实现了以下功能:

- 通过程序自动生成的3D场景和场景的逻辑,因此可以自动构建任务集.
- 自动给出求解步骤,因此可以自动构建训练集.
- 自动生成场景语义图.

<img src="/App/FuzzyWorld/DatGraph/fw.png" width="420" hegiht="313" align=center />


-----------------------------------------------
### 语义图
根据大脑的功能性原理,即,大脑负责认知和推理的是不同的模块(那么负责学习认知和推理的模块应该也不一样),我们于是将语义图模块拆分为$G_S$和$G_I$分别负责认知和推理.

- **在环境状态认知语义图$G_S$中**：谓词连接了实体或概念，所以$G_S$是多个三元组组成的，比如$is(moving,cow)$或者$biggerThan(cow,desk)$。
- **在逻辑推断语义图$G_I$中**：是有因果联系的事件构成的，比如$is(openning,light) \Rightarrow is(hot,room)$就是一个一阶逻辑的事件关联。

<img src="/App/FuzzyWorld/DatGraph/GV.jpg" width="520" hegiht="313" align=center />

-------------------------------------------------
### 将语义图融合进Agent网络

分别负责认知和推理的$G_S$和$G_I$需要表征为向量才能融入网络,如果采用类似**Node2vec**的**Embedding**方法恐将损失准确性,所以目前采用的还是类似**one-hot**的无歧义表征$G_S \in \mathbb{R}^{ d_o \times d_o \times d_p}$.格式如下:
$$G_S=
\left[
 \begin{matrix}
   [1,0,0,1...] & [1,0,0,0...] & [1,0,1,1...] \\
   [1,0,0,1...] & [1,0,0,1...] & [1,0,1,1...] \\
   [0,0,0,1...] & [1,0,1,0...] & [0,0,1,1...] 
  \end{matrix}
  \right]
$$
$G_I \in \mathbb{R}^{ d_E \times d_E}$表征的是事件之间的关联性,可由矩阵表达.下面就可以构造一个融合语义图的Agent网络:

<img src="/App/FuzzyWorld/DatGraph/model.png" width="420" hegiht="313" align=center />

--------------------------------------------------
### 实验及其结果分析
训练方式上,传统的强化学习训练应当是将Agent代入环境一步步完成任务且学习$\pi(a|s)$和$Q(a,s)$(连续任务:让Agent一直做下去),但是为了方便神经网络的训练采用的方式是"离散任务",也就是先在环境上跑一边正确的任务示范,收集数据$\{ V^{(t)},S^{(t)},G_s^{(t)},A^{(t)},V^{(t)}\}$然后训练Agent网络.

对比Baidu和我们的demoNet在任务完成上的结果:

<img src="/App/FuzzyWorld/model_baidu.png" width="620" hegiht="313" align=center />
<img src="/App/FuzzyWorld/model_mine.png" width="620" hegiht="313" align=center />

总体来说结论就是,用于决策的policy network $\pi(a^{(t)}|G_S,G_I,l^{(t)})$和用于评估策略的value network $Q(A,G_S,G_I,l^{(t)})$收敛得不好,且任务成功率亦不高.后期我会继续改进,大概就以下几个方面:

- 数据集方面仍然需要优化,当前的状态网络和逻辑网络是整个参与模型训练,细腻度不够.
- 网络架构方面,仍然没有太好地利用好两个语义图参与推理(是否不应进行如此的模糊拟合而应该加入更多Symbolic Reasoning的东西?)
- 损失函数方面,当前只是简单的似然最大化.
