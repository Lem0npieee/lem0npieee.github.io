## 1 历史背景

### Evolution Big Bang

视觉功能的出现 → 积极主动捕食/逃避捕食 → 物种需急速演化 → 物种数量爆炸

### 机器获得视觉的历史

- Camera Obscura（18th century）
- → 动物视觉处理机制的研究（1959）
- → Block World（1963）→ Summer Vision Porject（1966）
- → David Mars 计算机视觉算法、图形化结构（圆柱、点线）、...（70-80s）
	- Challenge：样本数据太少
- → Image Segmentation（1997）
- → Face Detection（2001）、机器学习（SVM，boosting，...）、SIFT Feature（1999）、Saptial Pyramid Matching（抽取特征向量符）（2006）、人体部位识别（2009）
	- 互联网发展、图片质量↑ → 拥有更多更好的数据
- → Object Recognition
	- PASCAL dataset（20 object categories）（2006-2012）
		- Challenge：训练样本不够 → 过拟合问题
	- ImageNet（2009）
	- Large Scale Visual Recognition Challenge（2014）
		- 1000 object classes、1431167 images
		- 错误率↓，甚至超过人类
		- ⭐2012：卷积神经网络（AlexNet）

###  卷积神经网络的历史

- 1998 用于识别手写字迹
  |    
  | 计算速度提升、GPU 、数据集
  ↓
- 2012 用于图像识别

## 2 图像分类
#### 2.1 数据驱动方法

Image Classification：A **core task** in Computer Vision
- Image → matrices of number
- label
Challenges：
- Semantic Gap（from images to number/pixel）
- Viewpoint Variation（should have robustness）
- Illumination, Deformation, Occlusion, Background（颜色、形状、光影...）

Solution：Data-Driven Approach
1. 收集数据和标签
2. 用机器学习进行分类（train）
3. 使用新图片来评估算法（predict）

Algorithm1：最邻近算法（Nearest Neighbor Classifier）
指标：曼哈顿距离（L1 distance）$d_1(I_1,I_2)=\sum_p |I_1^p-I_2^p|$  
函数：`__init__(), train(), predict()` 
缺陷：时间差异大（train: O(1) VS predict: O(N)）
#### 2.2 K-最近邻算法

→ Algorithm2：K-nearest Neighbor
Take majority vote from **K closest points**
缺陷：噪声更大
改进：欧式距离（L2 distance） $d_2(I_1,I_2)=\sqrt{(\sum_p (I_1^p-I_2^p)^2)}$ 
超参数：K值，距离形式
How do I set?
- Idea 1: Choose the best on the data ×
- Idea 2: Choose the best on the **test** data ×
- Idea 3: Choose the best on the **validation** data and use it on the **test** data √
	- Key: Split data into train, val, and test. 
- Idea 4: Split data into folds, and try each fold as validation in turns 
	- ![[Pasted image 20250719211333.png|400]]
	- Useful for small datasets, but not too frequently
缺点：
1. Slow while testing
2. Distance are not informative
3. 数据需要在空间中密集分布 → 指数级增长的训练

### 2.3 线性分类（Linear Classification）

Linear Classifier 1 + Linear Classifier 2 + ... + Linear Classifier n = Neural Network

e.g. 图片描述系统：CNN for image + RNN for language

Algorithm：Parametric Approach
- Image → $f(x,W)$ → score of labels
- $f(x,W) = Wx+b$
![[Pasted image 20250720103851.png|400]]
几何理解：用线性图像（线性决策边界）把空间分成若干区域
缺陷：非线性情形

## 3 损失函数及其优化
### 3.1 损失函数介绍

—— How to evaluate the parametric $W$ ?
—— **Loss function**

Given a dataset $\{(x_i,y_i)\}_{i=1}^N$ （$x$ is image, $y$ is label）
Loss: $L=\frac{1}{N}\sum_i L_i(f(x_i,W),y_i)$ 

E.g1. in  Multiclass SVM, $L_i=\sum_{j≠y_i} max(0,s_j-s_{y_i}+1)$ （合页函数）
![[Pasted image 20250720110449.png|300]] 
问题：如果最小的损失函数对应多个$W$怎么选择？过拟合问题？
Improved Loss (Regularization): $L=\frac{1}{N}\sum_i L_i(f(x_i,W),y_i) + \lambda R(W)$ 
→ **Model should be SIMPLE**
In common use: 
- L1/L2 regularization
- Elastic net (L1+L2)
- Max norm
- Dropout
- ... (Anything to penalize the complexity of the model)

E.g2. Softmax Classifier (Multinomial Logistic Regression)
赋予了得分额外的含义（概率）
$P(Y=k|X=x_i)=\frac{e^{s_k}}{\sum_je^{s_j}}$ (Softmax function)
where $s=f(x_i;W)$ ；
$L_i=-log P(Y=y_i|X=x_i)=-log(\frac{e^{s_{y_i}}}{\sum_je^{s_j}})$ 
**Goal: 真实类别的概率接近于1**
![[Pasted image 20250720132142.png|400]]
### 3.2 优化

Strategy 1: random search (poor)
Strategy 2: follow the **slope** 
- 1-dimension: $\frac{df(x)}{dx}=\lim_{h→0} \frac{f(x+h)-f(x)}{h}$   
- n-deimension: gradient
	- numeric gradient（有限差分逼近）
		- ![[Pasted image 20250720135032.png|400]]
		- 缺陷：慢
	- analytic gradient（微积分）
		- $dW=f(W)$ and calculate

#### Geadient Descent
```python
while True:
	weights_grad = evaluate_gradient(loss_fun, data, weights)
	weights -= step_size * weights_grid
```
注意: step_size 又叫学习率，是一个重要的超参数
缺陷：还是慢

**Improvement**：Stochastic Gradient Descent (SGD)
- 每次迭代，只选取一小部分作为训练样本 (minibatch) 进行 Gradient Descent
- 通常取$2^n$个

#### Image Features
计算图片形象有关的数值 → 合并特征向量 → 图像特征表述
图像像素输入线性分类器 → 图像特征输入线性分类器
原因：可以把复杂（非线性）的数据集转化为可线性分类的

E.g1 Color Histogram
![[Pasted image 20250721092941.png|400]]
E.g2 Histogram of Oriented Gradients (HoG)
![[Pasted image 20250721093330.png|400]]
E.g3 Bag of Words
先编写一个视觉词典，然后用图像“单词”编码图像
![[Pasted image 20250721093748.png|400]]
### 3.3 反向传播

目标：最小化损失函数
- 一维形式
- 矩阵形式
（此处内容基本与《人工智能》课程相一致，笔记省略）

整理了一下作业1中的其他updated rules:
##### AdaGrad

^9a033a

- $g=∇_{\theta_{k-1}}L(\theta)$  计算参数$\theta$ 在$\theta_{k-1}$点上的梯度
- $r_k=r_{k-1}+g⊙g$  累积梯度的平方和
- $\eta=\frac{\eta}{\sqrt{r_k+\epsilon}}$  用累积梯度平方和调整学习率
- $\theta_k=\theta_{k-1}-\eta g$  更新参数
优点：根据历史梯度自适应调整学习率
缺点：随着累积平方梯度的增加，学习率会持续衰减，后期学习率容易过小，模型训练难以收敛
##### RMSProp
- $g=∇_{\theta_{k-1}}L(\theta)$  计算参数$\theta$ 在$\theta_{k-1}$点上的梯度
- $r_k=\beta r_{k-1}+(1-\beta)g⊙g$  累积梯度的平方和，但添加了衰减参数$\beta$（通常为0.9）
- $\eta=\frac{\eta}{\sqrt{r_k+\epsilon}}$  用累积梯度平方和调整学习率
- $\theta_k=\theta_{k-1}-\eta g$  更新参数
优点：相比于Adagrad，学习率的调整更加平滑
缺点：超参数$\beta$的选择影响很大；不能保证所有问题最优
##### Adam
- $g=∇_{\theta_{k-1}}L(\theta)$  计算参数$\theta$ 在$\theta_{k-1}$点上的梯度
- $m_k=\beta_1 m_{k-1}+(1-\beta_1)g, v_k=\beta_2 v_{k-1}+(1-\beta_2)g⊙g$  计算梯度的一阶矩和二阶矩
- $\hat{m_k}=\frac{m_k}{1-\beta_1^k}, \hat{v_k}=\frac{v_k}{1-\beta_2^k}$  偏差校正
- $\theta_k=\theta_{k-1}-\frac{\eta}{\sqrt{\hat{v_k}+\epsilon}}$  更新参数
优点：通过计算一阶和二阶矩估计来为每个参数自适应地调整学习率，通过偏差校正可以加速初期的学习速率，性能良好
缺点：超参数的选择
## 4 卷积神经网络
### 4.1 历史介绍

感知器算法（1957）→ 多层传感器网络（1960）→ 反向传播算法（1986）→ 用反向传播和梯度识别的方法训练卷积神经网络（文档识别）（1998）→ 神经网络狂潮（声音识别、图像识别AlexNet、...）（2012始）
### 4.2 卷积和池化

Fully Connected Layer: 32×32×3 image → stretch to 3072×1 → Wx+b（W：10×3072）→ activation（10×1）
Convolusion Layer: 32×32×3 image → W$*$x+b（W：5×5×3 filter）→ activation map（28×28×1）-if we have 6 filters→ activation map（28×28×6）
![[Pasted image 20250723110922.png|100]] **Output Size:** $(N-F)/stride+1$ 
##### Padding
Padding: zero pad the boader of the image
- In general, common to see stride 1
- the size of filter is $F×F$ → zero-padding with $P=(F-1)/2$
- OUtput Size: $(N+2P-F)/stride+1$
![[Pasted image 20250723111533.png|400]]
作用：如果不填充，卷积层会迅速缩小，容易损失信息

卷积核的抽象含义：关注图像的各个局部区域被激发的程度
卷积核 → 感受野（recepitive field）
每个卷积核有不同的功用（边缘检测、颜色变化、纹理、...）
##### Pooling
- function: makes the representations smaller and more manageable
- It effects the number of parameters
- 只做平面变化，不改变深度
![[Pasted image 20250723115342.png|400]]

E.g 1 Max Pooling（常用）
![[Pasted image 20250723115638.png|400]]
注意：在池化层中，设置步长时尽量避免对某些区域重叠q处理
Common setting: $F=2, S=2; F=3, S=3$
抽象含义：选取最强/最显著的刺激来激活

E.g 2 Average Pooling

### 4.3 全连接层

![[Pasted image 20250723142612.png|400]]
[ConvNetJS CIFAR-10 demo](https://cs.stanford.edu/people/karpathy/convnetjs/demo/cifar10.html)

## 5 训练神经网络
### 5.1 激活函数
##### sigmoid函数 (DO NOT USE)
![[Pasted image 20250723145112.png|200]]
- $y=\frac{1}{1+e^{-x}}$
- 元素压缩到(0,1)范围内
- 抽象含义：神经元饱和放电率
- 问题：饱和神经元使梯度缺失；数值始终为正 → 梯度更新低效

##### tanh(x)函数
![[Pasted image 20250723145942.png|200]]
- $y=tanh(x)$ 
- 元素压缩到(-1,1)范围内
- 问题：还是有梯度缺失
##### ReLU函数（常用）
![[Pasted image 20250723150213.png|200]]
- $y=max(0,x)$
- 优势：正数部分不会饱和；计算速度快
- 问题：不是以0为中心，负数部分还是有梯度缺失 (died ReLU)
##### Leaky ReLU
![[Pasted image 20250723151102.png|200]]
- $y=max(0.01x,x)$ 
- 优势：速度快；不会die
- 扩展：PReLU $y=max(\alpha x,x)$
	- $\alpha$ 可当作反向传播学习的参数
##### ELU
![[Pasted image 20250723151406.png|200]]
- $f(n) \begin{cases} x, &x\gt0\\ \alpha(exp(x)-1), &x≤0 \end{cases}$ 
- 优势：同ReLU，负饱和机制提升鲁棒性
- 问题：含有指数运算
##### Maxout ”Neuron“
- $y=max(w_1^Tx+b_1,w_2^Tx+b_2)$ 
- 并非传统的”点积→非线性“的结构，、而是取两线性函数最大值
- 优势：不会饱和；不会die
- 问题：需要多个权重矩阵$w_i$ 
### 5.2 数据预处理
##### Data Preprocessing

zero-centered data: `X -= np.mean(X, axis=0)` 
- 两种处理选择：用总图像的均值处理；分通道处理
normalized data: `X /= np.std(X, axis=0)` 
other advanced process: PCA, Whitening（ignored）
##### Weight Initialization

What if all W=0 init is used? 
权重为0 → 每个神经元执行相同的操作 → 输出相同数值 → 算出相同梯度 → 用相同方式更新 → 完全相同的神经元

*权重太小，网络崩溃；权重太大，网络饱和*

Idea 1：Small random numbers
`W=0.01*np.random.randn(D,H)` 
只适用于结构简单的神经网络

Idea 2：random numbers
`W=1.0*np.random.randn(D,H)`
通过tanh后会饱和，权重趋于±1，梯度趋于0

Idea 3：Xvaier
`W = np。random.randn(D,H) / np.sqrt(D)`
Reasonable initialization
但是不适用于ReLU函数

##### Batch Normalization

^3c4fb0

Consider a batch of activations at some layer. 
To make each dimension unit **gaussian**:  $\hat{x}^{(k)}=\frac{x^{(k)}-E[x^{(k)}]}{\sqrt{Var[x^{(k)}]}}$ 
The batch normalization(BN) layer usually inserted after fully connected(FC) or convolutional layers, and before the nonlinearty. 
![[Pasted image 20250724150707.png|200]]
控制饱和程度：$BN_{\gamma,\beta}(x^{(k)})=y^{(k)}=\gamma^{(k)}\hat{x}^{(k)}+\beta^{(k)}$ （缩放、平移）
结果：改进玩过梯度流，具有更高鲁棒性（make it easier to train model） 
Attention: While testing, the BatchNorm layer function is different. It uses a **single fixed empirical mean** of activations, not the mean/std computed based on the batch. 

### 5.3 观察学习过程

1 Double check the loss is reasonable is important
e.g Using Softmax loss in cifar-10: it should be $≈-log0.1$ at first

2 Training 
- Start up with a very small amount of data
- Turn off the regularization and see if the loss go down to zero

3 Start training! 
- Use full training data
- Add a small amount of regularzation
- Make sure the learning rate
	- low learning rate: accurcy grows slowly
	- high learning rate: NaN occurs (loss exploding)
	- ⭐common learning rate ranges **1e-3~1e-5** 
	![[Pasted image 20250724163230.png|300]]
4 Hyperparameter Optimization
- Strategy: cross-validation
- 先用几个epoch确认大致参数区间，再花长时间训练找出最佳的参数
- 通常采用乘法原则，即参数之间步长用乘法

### 5.4 优化

Problems with SGD: 
- zig-zag along steep direction, but slow along shallow dimension(correct direction)
![[Pasted image 20250724171417.png|600]]
	(which is common in higher dimension)
- get stuck at saddle point (because the gradiant is zero)
- the "S" → noisy estimate of the gradiant

###### Solution: SGD + Momontum
```python
vx = 0
while True:
	dx = compute_gradiant(x)
	vx = rho * vx + dx
	x += learning_rate * vx
```
rho ($\rho$):  摩擦系数，一般为0.9或0.99
![[Pasted image 20250724173143.png|500]]
###### Nesterov Momentum
$v_{t+1}=\rho v_t - \alpha \nabla f(x_t+\rho v_t)$ 
$x_{t+1}=x_t+v_{t+1}$ 
可以用换元简化计算 ↓
```python
dx = compute_gradiant(x)
old_v = v
v = rho * v - learning_rate * dx
x += -rho * old_v + (1 + rho) * v
```

*（以下部分可结合前面“反向传播”章节中整理的[[#^9a033a|内容]]来学习）*
###### AdaGrad（一般不使用）
```python
grad_squared = 0
while True:
	dx = compute_gradiant(x)
	grad_squared += dx * dx
	x -= learning_rate * dx / (np.sqrt(grad_squared) + 1e-7)
```
加速梯度小的训练进度，延缓梯度大的训练进度
缺点：步长越变越小

###### RMSProp
```python
grad_squared = 0
while True:
	dx = compute_gradiant(x)
	grad_squared = decay_rate * grad_squared + (1 - decay_rate) * dx * dx
	x -= learning_rate * dx / (np.sqrt(grad_squared) + 1e-7)
```

###### Adam
```python
# Common starting point: beta1 = 0.9, beta2 = 0.999, learning_rate = 5e-4~1e-3 
first_moment = 0
second_moment = 0
while True:
	dx = compute_gradiant(x)
	first_moment = beta1 * first_moment + (1 - beta1) * dx  # momentum
	second_moment = beta2 * second_moment + (1 - beta2) * dx * dx  # AdaGrad/RMSProp
	first_unbias = first_moment / (1 - beta1 ** t)  # Bias correction
	second_unbias = second_moment / (1 - beta2 ** t)
	x -= learning_rate * first_moment / (np.sqrt(second_moment) + 1e-7)
```

##### Learning Rate Decay

在训练过程中，可以不用固定使用一个学习率，可以动态更新调整
![[Pasted image 20250727143640.png|400]]
- Step dacay: 每轮训练减半
- Exponential decay: $\alpha=\alpha_0 e^{-kt}$
- 1/t decay: $\alpha=\alpha_0/(1+kt)$ 
注意：多用于SGD momentum，但是很少用于Adam

##### First Order → Second Order Optimization

Taylor expansion: $J(\theta)=J(\theta_0)+(\theta-\theta_0)^T\nabla_\theta J(\theta_0)+\frac{1}{2}(\theta-\theta_0)^TH(\theta-\theta_0)$ 
Solve with Newton method: $\theta^*=\theta_0-H^{-1}\nabla_\theta J(\theta_0)$ 
(There's no learning rate! )
**L-BFGS**: 典型的二阶优化器（但不是很有用）

##### Beyond Training Error
Target: reducing the gap between train and test error
Method : model ensembles
- train multiple independent models/select snapshots
- average the results
- use polyak averaging

### 5.5 正则化
#### Dropout
目标：防止过拟合
方法：每一层随机去掉部分神经元/激活函数的结果置为0（超参数：去除率$p$，通常为0.5）
在卷积神经网络中，是随机把整个特征映射置为0
![[Pasted image 20250727145550.png|500]]
好处：避免特征之间的相互关联/适应
（比如在判断猫时，猫并不是一定有尾巴，并不是要同时有耳朵和尾巴，有可能只露出了尾巴 → 防止过拟合）

In test time：
在测试时，需要避免随机删去神经元所产生的随机结果
方法：“average out” the randomness at test-time
$y = E_z[f(x,z)] = \int p(z)f(x,z)dz = f(x,z) * p$ 

抽象理解：在训练时加入随机性/噪声，然后在测试时去除掉
##### Another method：Dropconnect
![[Pasted image 20250727153046.png|500]]
#### [[#^3c4fb0|Batch Normalization]]
#### Data Augmentation
- Random flips
- Random crops and scales
- Random rotation, stretching, shearing, ...
- Color jiffer/PCA
#### Fractional max pooling
![[Pasted image 20250727153449.png|400]]
#### Stochastic Depth
训练时随机丢弃部分层；测试时用全部层
![[Pasted image 20250727153418.png|250]]
### 5.6 迁移学习

作用：不需要超大的样本集来训练卷积神经网络
![[Pasted image 20250727154504.png|800]]

## 6 深度学习软硬件
### 6.1 CPU vs GPU

**CPU: central processing unit**
- fewer cores
- faster and more capable
- great at sequential tasks
**GPU: graphics processing unit**
- more cores
- slower and dumber
- great for parallel tasks
- useful in deep learning: NVIDIA

Programming GPUs
- CUDA (NVIDIA only)
- OpenCL

CPU/GPU communication
Solution to accelerate training: 
- read all data into RAM
- use SSD instead of HDD
- use multiple CPU threads to prefresh data
### 6.2 深度学习框架

- Caffe → Caffe2
- Torch → PyTorch
- Theano → Tensorflow

Computational Graphs
![[Pasted image 20250729093625.png|400]]

![[Pasted image 20250729093920.png|200]]
Use numpy: can't run on GPUs, and have to compute  our own gradiant
Use Temsorflow:
```python
import numpy as np
np.random.seed(0)
import tensorflow as tf

N, D = 3, 4

with tf.device('/gpu:0')  # select run device
x=tf.placeholder(tf.float32)
y=tf.placeholder(tf.float32)
z=tf.placeholder(tf.float32)

#define the graph
# forward pass
a = x * y
b = a + z
c = tf.reduce_sum(b)

# compute gradiant
grad_x, grad_y, grad_x = tf.gradients(c, (x, y, z))

# run the graph
with tf.Session() as sess  
	# feed in the numpy arrays
	values = {
		x: np.ranadom.randn(N, D)
		y: np.ranadom.randn(N, D)
		z: np.ranadom.randn(N, D)
	}
	out = sess.run([c, grad_x, grad_y, grad_z], feed_dict = values)
	c_val, grad_x_val, grad_y_val, grad_z_val = out

```

Use PyTorch: 
```python
import torch
from torch.autograd import Variable

N, D = 3, 4
x = Variable(torch.randn(N, D).cuda(), requires_grad = True)
y = Variable(torch.randn(N, D).cuda(), requires_grad = True)
z = Variable(torch.randn(N, D).cuda(), requires_grad = True)

# forward pass
a = x * y
b = a + z
c = torch.sum(b)

# compute gradiant
c.backward()

print(x.grad.data)
print(y.grad.data)
print(z.grad.data)
```
##### Tensorflow: Neural Net
```python
import numpy as np
np.random.seed(0)
import tensorflow as tf

N, D, H = 64, 1000, 100

#define the graph
x = tf.placeholder(tf.float32, shape=(N, D))
y = tf.placeholder(tf.float32, shape=(N, D))

# forward pass
# method 1
w1 = tf.Variable(tf.random_normal((D, H))) # persist in the graph → efficient 
w2 = tf.Variable(tf.random_normal((H, D)))
h = tf.maximum(tf.matmul(x, w1), 0)
y_pred = tf.matmul(h, w2)
# method 2 (use layers)
init = tf.contrib.layers.xavier_initializer()
h = tf.layers.dense(inputs=x, units=H, activation=tf.nn.relu, kernel_initializer=init)
y_pred = tf.layers.dense(inputs=h, units=D, kernel_initializer=init)

#compute loss
# method 1
diff = y_pred - y
loss = tf.reduce_mean(tf.reduce_sum(diff ** 2, axis=1))
# method 2
loss = tf.losses.mean_squaredd_error(y_pred, y)

# compute gradiant
grad_w1, grad_w2 = tf.gradients(loss, [w1, w2]) 

learning_rate = 1e-5

# method 1
new_w1 = w1.assign(w1 - learning_rate * grad_w1)
new_w2 = w2.assign(w2 - learning_rate * grad_w2)
updates = tf.group(new_w1, new_w2) # need to update here

# method 2 (use optimizer)
optimizer = tf.train.GradientDescentOptimizer(1e-5) # use RMSProp here
updates = optimizer.minimize(loss)

# run the graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer()) # initialize w1,w2 (run random_normal)
    values = {x: np.random.randn(N, D),
              y: np.random.randn(N, D),}
    losses = []
    # start training
    for t in range(50):
        loss_val, _ = sess.run([loss, updates], feed_dict=values)
        losses.append(loss_val)
```
##### Keras: High-Level Wrapper
*Keras is a layer on top of TensorFlow*
```python
import numpy as np 
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD

N, D, H = 64, 1000, 100

# Define model object as a sequence of layers
model = Sequential()
model.add(Dense(input_dim=D, output_dim=H))
model.add(Activation('relu'))
model.add(Dense(input_dim=H, output_dim=D))

#define optimizer object
optimizer = SGD(lr=1e0)

# build model 
model.compile(loss='mean_squared_error',
              optimizer=optimizer)

x = np.random.randn(N, D)
y = np.random.randn(N, D)

# train model
history = model.fit(x, y, nb_epoch=50, batch_size=N, verbose=0)
```
##### Pytorch: Three levels of abstraction
**Tensor:** inperative ndarray, but runs on GPU
- To run on GPU, just cast tensors to a cuda datatype
```python
import torch

dtype = torch.cuda.FloatTensor # runs on GPU

# create random tensors
N, D_in, H, D_out = 64, 1000, 100, 10
x = torch.randn(N, D_in).type(dtype)
y = torch.randn(N, D_out).type(dtype)
w1 = torch.randn(D_in, H).type(dtype)
w2 = torch.randn(H, D_out).type(dtype)

learning_rate = 1e-6

for t in range(500):	
	# forward pass
    h = x.mm(w1)
    h_relu = h.clamp(min=0)
    y_pred = h_relu.mm(w2)
    loss = (y_pred - y).pow(2).sum()

	#backward pass
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.t().mm(grad_y_pred)
    grad_h_relu = grad_y_pred.mm(w2.t())
    grad_h = grad_h_relu.clone()
    grad_h[h < 0] = 0
    grad_w1 = x.t().mm(grad_h)

	# gradiant decent
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2
```

**Variable:** Node in a computational graph, stores data and gradiant
- `x.data`is a Tensor
- `x.grad`is a Variable of gradiants
- `x.grad.data`is a Tensor of gradiants
```python
import torch
from torch.autograd import Variable

N, D_in, H, D_out = 64, 1000, 100, 10
x = Variable(torch.randn(N, D_in), requires_grad=False) 
y = Variable(torch.randn(N, D_out), requires_grad=False)
w1 = Variable(torch.randn(D_in, H), requires_grad=True)
w2 = Variable(torch.randn(H, D_out), requires_grad=True)

learning_rate = le-6  
for t in range(500):
	# forward pass
    y_pred = x.mm(w1).clamp(min=0).mm(w2)
    loss = (y_pred - y).pow(2).sum()

	# compute gradient
    if w1.grad: w1.grad.data.zero_()
    if w2.grad: w2.grad.data.zero_()
    loss.backward()

	# make gradient step on weights
    w1.data -= learning_rate * w1.grad.data
    w2.data -= learning_rate * w2.grad.data
```

you can also define your own Autograd functions: 
```python
class ReLU(Function):
    def forward(self, x):
        self.save_for_backward(x)
        return x.clamp(min=0)

    def backward(self, grad_y):
        x, = self.saved_tensors
        grad_input = grad_y.clone()
        grad_input[x < 0] = 0
        return grad_input

# 省略其他代码

for t in range(500):
	relu = ReLU()
	y_pred = relu(x.mm(w1)).mm(w2)
	loss = (y_pred - y).pow(2).sum()

# 省略其他代码
```

**Module:** A neural network layer, may store state or learnable weights

##### nn: High-Level Wrapper in PyTorch
```python
import torch
from torch.autograd import Variable

N, D_in, H, D_out = 64, 1000, 100, 10
x = Variable(torch.randn(N, D_in), requires_grad=False)
y = Variable(torch.randn(N, D_out), requires_grad=False)

# define our model as a sequence of layers
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out)
)

# define loss function
loss_fn = torch.nn.MSELoss(size_average=False)

learning_rate = 1e-4
# optimizer
optimizer = torch.optim.Adam(model,parameters(), lr=learning_rate)

for t in range(500):
	# forward pass
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
	
	# backward pass
    model.zero_grad()
    loss.backward()

	# make gradient step on weights
	# method 1
    for param in model.parameters():
        param.data -= learning_rate * param.grad.data

	# method 2 (use opptimizer)
	optimizer.step()
```

you can also define your own Modules: 
```python
class TwoLayerNet(torch.nn.Module):

	# initializer sets up 2 children (Modules contain modules)
    def __init__(self, D_in, H, D_out):
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

	# define forward pass
	# autograd will handle backward pass so there's no need
    def forward(self, x):
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred

N, D_in, H, D_out = 64, 1000, 100, 10
x = Variable(torch.randn(N, D_in), requires_grad=False)
y = Variable(torch.randn(N, D_out), requires_grad=False)

# construct and train an instance of our model
model = TwoLayerNet(D_in, H, D_out)

criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

for t in range(500):
    y_pred = model(x)
    loss = criterion(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```
##### DataLoaders
A dataloader can wrap a dataset and provides minibatching, shuffing, multithreading
youcan also write your own dataset class
```python
import torch
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader

N, D_in, H, D_out = 64, 1000, 100, 10
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# load data
loader = DataLoader(TensorDataset(x, y), batch_size=8)

model = TwoLayerNet(D_in, H, D_out) 

criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

for epoch in range(10):
	# Iterate over loader to form minibatches
    for x_batch, y_batch in loader:
        x_var, y_var = Variable(x), Variable(y) 
        y_pred = model(x_var)
        loss = criterion(y_pred, y_var)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```
##### Static vs Dynamic Graphs
Optimization: 静态图一直复用层，比较好去做优化
Serialization: 静态图序列化之后，可以直接运行而脱离建图代码；动态图反之
Conditional: 动态图的条件运算更加简洁
Loops: 动态图可以直接使用编程语言原生的循环结构，灵活方便

### 6.3 CNN架构
##### LeNet-5
![[Pasted image 20250729151303.png|500]]
- Digital Recognition Field
- 5x5 conv filters, applied at stride 1
- Pooling layers were 2x2 applied at stride 2
##### AlexNet
![[Pasted image 20250729151557.png|500]]
- Structure: CONV1|MAXPOOL1|NORM1|CONV2|MAXPOOL2|NORM2|CONV3|CONV4|CONV5|MAXPOOL3|FC6|FC7|FC8
- Parameters: (11 * 11 * 3) * 96 = 35K
- first use of ReLU
- heavy data augmentation
##### VGGNet
![[Pasted image 20250729155025.png|300]]
- Small filters, deeper networks (VGG16, VGG19)
- keep simple structure of 3x3 convs with periodic pooling
- Reason: use fewer parameters and stack more of them, instead of larger filters
- use 3x3 by 3 times = use 7x7 by 1 time → the network can go deeper (allow more non-linearities, less parameters)
⭐Commonly, most memory is in **early CONV**, and most params are in **late FC**
##### GoogLeNet
![[Pasted image 20250729161121.png|600]]
- 22 layers, efficent "Inception module"
- No FC layers, only 5 million parameters
- 12x less than AlexNet
- Special structure: Stem Network, AvgPool-1x1 Conv-FC-FC-Softmax, Classifier Output
Inception module: design a good local network topology and stack these modules on top of each other
![[Pasted image 20250729155825.png|400]]
特点：对相同曾的相同输入并行应用不同的过滤操作，最后串联/拼接在一起得到张量输出
问题：计算复杂度
Solution: bootleneck layers that use 1x1 convolutions to reduce feature depth 
![[Pasted image 20250729160953.png|400]]
##### ResNet

  ![[Pasted image 20250730091603.png|800]]
- Architecture: Stack residual blocks, each of which has two 3x3 conv layers; no FC layers (the fc 1000 is just to output result)
- Instead of directly trying to fit a desired underlying mapping, use network layers to fit a residual mapping
![[Pasted image 20250730092424.png|400]]
Use layers to fit residual $F(x)=H(x)-x$ instead of $H(x)$ directly, and then add $x$ back

### 7 循环神经网络

Process Sequences: In RNN, we have many choices of the type of input and output data
![[Pasted image 20250730100145.png|400]]
- One to many: Image Captioning (image → sequence of word)
- Many to one: Sentiment Classification (sequence of words → sentiment)
- Many to many: Machine Translation (seq of words → seq of words)
- Many to many2: Video classification on frams level
- Many to one + one to many: **encoder & decoder**

Vanilla RNN (Recurrent Neural Network): x -> RNN -> y (usually want to predict a vector at some time steps)
Recurrence formula: $h_t=f_W(h_{t-1},x_t)$ 
- $h_t$ : new state
- $h_{t-1}$ : old state
- $x_t$ : input vector at some time step
- the $f_W$ are the same at every time step
The state consists of a single "hidden" vector $h$ : 
	![[Pasted image 20250730112328.png|250]]
	$h_t=tanh(W_{hh}h_{t-1}+W_{xh}x_t)=tanh \begin{pmatrix} {\begin{pmatrix}W_{hh} & W_{hx} \end{pmatrix}} &{\begin{pmatrix} h_{t-1}\\ x_t \end{pmatrix}} \end{pmatrix} = tanh \begin{pmatrix} W &{\begin{pmatrix} h_{t-1}\\ x_t \end{pmatrix}} \end{pmatrix}$
	$y_t=W_{hy}h_t$ 
	
![[Pasted image 20250730102202.png|400]] ![[Pasted image 20250730102310.png|400]] ![[Pasted image 20250730102358.png|400]]

Example: Character-level Language Model
Vocabulary: \[h,e,l,o\]
training sequence: "hello" (try to predict the next letter)
![[Pasted image 20250730102826.png|400]]

Problem: while training, you should forward pass and backpropagation through whole graph → super slow! 
Solution: Truncated Backpropagation through time
- take small batches of data to compute gradiant steps
![[Pasted image 20250730103745.png|400]]
（look at the array at the bottom）

E.g Image Captioning
![[Pasted image 20250730105826.png|400]]

Improvement: Multilayer RNNs
- $h_t^l=tanh W^l\begin{pmatrix} h_t^{l-1}\\h_{t-1}^l \end{pmatrix}$   
- Commonly 2/3/4 layer recurrent neural network
![[Pasted image 20250730111342.png|300]]

Gradient Flow: 
![[Pasted image 20250730112758.png|600]]
Problem: $奇异值_{max}\gt 1$ → Exploding gradients; $奇异值_{max}\lt 1$ → Vanishing gradients
Solution:  For exploding gradients: **Gradient clipping** - scale gradient if its norm is too big
		`grad *= (threshold / grad_norm)` 
	    For Vanishing gradients: move to more complicated RNN architecture → **LSTM**
##### LSTM
![[Pasted image 20250730114801.png|400]] ![[Pasted image 20250730115546.png|320]]
- $i$ : Input gate, whether to write to cell, range \[0,1\]
- $f$ : Forget gate, whether to erase cell, range \[0,1\]
- $o$ : Output gate, how much to reveal cell, range \[0,1\]
- $g$ : Gate gate, how much to write to cell, range \[-1,1\]

$\begin{pmatrix} i \\ f \\ o \\ g \\ \end{pmatrix} = \begin{pmatrix} \sigma \\ \sigma \\ \sigma \\ tanh \\ \end{pmatrix} W \begin{pmatrix} h_{t-1} \\ x_t \\ \end{pmatrix}$ 
$c_t = f \bigodot c_{t-1}+i \bigodot g$  $c_t$ : cell state (internal in LSTM)
$h_t=o \bigodot tanh(c_t)$ 

Backpropagation: 
Backpropagation from $c_t$ to $c_{t-1}$ only elementwise multiplication by $f$ ! And without big matrix $W$ !
(Similar to ResNet)
![[Pasted image 20250730120246.png|600]]

## 8 识别与分割
### 8.1 Semantic Segmentation（语义分割）
Label each pixel in the image with a category label

Idea 1：Sliding windows
- not perfect
- complex
Idea 2: Fully convolutional
- Design a network to make predictions for pixels **all at once**
- use cross entropy（交叉熵）loss to evaluate
- dataset is hard to get/make
- very complex
![[Pasted image 20250730145948.png|500]]
- Improvement: add downsampling annd upsampling inside the network
![[Pasted image 20250730150708.png|500]]
**\*** Addition 1：Unpooling
![[Pasted image 20250730151020.png|400]] ![[Pasted image 20250730151222.png|425]]
**\*** Addition 2：Transpose Convolution/Upconvolution
![[Pasted image 20250730163601.png|500]]
（感觉课上没有讲的很清楚，不如这篇文章：[转置卷积 - 知乎](https://zhuanlan.zhihu.com/p/549164774)）
### 8.2 Classcification + Localization
Treat localization as a regression problem
![[Pasted image 20250730164452.png|500]]
Loss (Multi-tasks loss) : Class scores → Softmax loss; Box Coordinates → L2 loss
Aside: Human Pose Localization

### 8.3 Object Detection
Difficulty: The number of objects is uncertain
Idea 1: Sliding window
Problem: how to choose the crops? too force and complex!
Idea 2: Region Proposals **(R-CNN)**
![[Pasted image 20250730170330.png|400]]
![[Pasted image 20250730170534.png|400]]
- Find ”blobby“ image regions that are likely to contain objects (Regions of interest)
- Then do image classcification
- some is useless but has high recall
- also predict the box coordinates again
- Problem: complex, slow, no learning
- Improvement: **Fast R-CNN**
	- 先通过一个卷积层再划分区域
	- "ROI Pooling" layer + FC
![[Pasted image 20250731093723.png|400]]
- Improvement 2: Faster R-CNN
![[Pasted image 20250731094059.png||400]]
 
Idea 3: **YOLO** (You Only Look Once) & SSD
![[Pasted image 20250731095352.png|400]]
- Divide image into grid, and each grid cell has a set of base boxes (dx, dy, dh, dw, confidence)
- Go from input image to tensor of scores with one big convolutional network
- Predict: 1. the offset of base bounding box; 2. classcification scores
- Output: 7 x 7 x (5 \* B + C)
Aside: Dense Captioning - Object Detection + Captioning
![[Pasted image 20250731100151.png|500]]
### 8.4 Instance Segmentation（物体分割）

Mask R-CNN
![[Pasted image 20250731100554.png|400]]
- 对每个候选框进行进一步微型分割
- 基于Faster R-CNN

### 8.5 Vosualizing & Understanding

What's going on inside ConvNets? 
![[Pasted image 20250731102422.png|600]]
##### First Layer: Visualize Filters
- To visualize, the weight ranges 0~255
- With oriented edges and opposing colors
##### Last Layer: Nearest Neighbors
![[Pasted image 20250731103213.png|400]]
- Instead of in pixel space, find neighbors in **feature space**
##### Last Layer: Dimensionality Reduction
![[Pasted image 20250731103838.png|300]]
- Visualize the space of FC feature vectors by reducing dimensionality of vectors (4096 -> 2)
- Algorithm: PCA（主成分分析）、T-SNE
##### Maximally Activating Patches
![[Pasted image 20250731105230.png|300]]
- find out what image input cause maximal activation in different features/neurons
- visualize the patches from dataset corresponding to the maximul activations of particular feature in particular layer
##### Occlusion & Saliency Maps & Intermediate Features via Backprop
![[Pasted image 20250731110132.png|400]]
- occlusion: 用数据集平均值遮挡图片的某部分，观察分数变动 → 哪一部分是识别的关键
- saliency maps: 扰动图中的像素，观察分数变动 → 哪些像素是特征
- intermediate features via guided backprop（没看太懂）: 微调反向传播，计算网络中某些中间值相对于图像像素的梯度 → 明白哪些像素会影响特定神经元的值
##### Gradient Ascent
Generate a synthetic image that **maximally actives a neuron**
Target: 最大程度激活神经元的值；生成图像是自然的
$I^* = arg\max_I f(I)+R(I)$ 
$f(I)$: Neuron value
$R(I)$: Natural image regularizer
1.  Initialize image to zero
	$arg\max_i S_C(I)-\lambda||I||_2^2$
	better regularizer: Penalize L2 norm of image
	other operation: Gaussian blur image; clip pixels with small values/gradients to zero
and repeat: 
2. Forward image to compute current scores
3. Backprop to get gradient of neuron value with respect to image pixels
4. Make a small update to the image
![[Pasted image 20250731143630.png|500]]

##### Amplify existing features
- jetter image before compute gradient 
- use L1 Normalize gradient
![[Pasted image 20250731145825.png|400]]
##### Feature Inversion（图像反演）
Target: Given a CNN feature vector for an image, output a new image that matches the feature vector, and looks netural
![[Pasted image 20250731150126.png|600]]
##### Texture Synthesis（纹理合成）
![[Pasted image 20250731150557.png|400]]
Idea 1: Nearest Neighbor
Idea 2: Gram Matrix [Gram matrix 详细解读 - 知乎](https://zhuanlan.zhihu.com/p/187345192)
##### Neural Texture Synthesis & Style Transfer
![[Pasted image 20250731152452.png|400]]
- Reconstructing texture from higher layers recovers larger features from the input texture
- texture=artwork
![[Pasted image 20250731152633.png|400]]
![[Pasted image 20250731152651.png|500]]
- loss: content loss & style loss
- you can control how big is the style image before computing gram matrix
- Problem: very slow → Fast Style Transfer (train another neural network to perform style transfer)
![[Pasted image 20250731153513.png|500]]

## 9 Unsupervised Learning
#### Supervised Learning
Data: (x, y) / (data, label)
Goal: learn a function to map x -> y
Example: classcification, object detection, segmentation, captioning, ...
#### Unsupervised Learning
Data: x (just data, no labels)
Goal: learn some **underlying hidden structure** of the data
Example: Clustering, dimensionality reduction, feature learning, density estimation, ...

#### Generated Models
Given training data, generate new samples from same distribution (p_model similar to p_data)
- realistic samples for artwork, etc.
- time-series data -> simulation and planning (reinforcement learning applications)
- enable inference of latent representations
### 9.1 PixelRNN & PixelCNN
##### Fully visible belief network
Use chain rule to decompose likelihood of an image x 
$p(x)=\prod_{i=1}^n p(x_i|x_1,...,x_{i-1})$ -> maximize it
Solution: Express complex distribution by a neural network
##### PixelRNN
![[Pasted image 20250801140718.png|300]]
Generate pixels starting from corner, and use RNN
Problem: sequential generation is slow
##### PixelCNN
![[Pasted image 20250801140908.png|300]]
Generate pixels starting from corner, and use CNN over context region
Output: softmax loss of each pixel
Problem: still slow
Improvement: gated convolutional layers, short-cut connections, discretized logistic loss, ...
#### Autoencoder 
What: an unsupervised approach for learning a lower dimensional feature representation from unlabeled training data
Input data $x$  — **Encoder** —>  Features $z$  — **Decoder** —>  Reconstructed input data $\hat{x}$  
![[Pasted image 20250801142704.png|400]]
($z$ usually smaller than $x$ because of dimensionality reduction)
Type of encoder/decoder: linear+nonliearity, FC, ReLU CNN
Training: L2 loss function $||x-\hat{x}||^2$

Encoder can be used to initialize a supervised model (Benifit: small dataset)
![[Pasted image 20250801143205.png|400]]
### 9.2 Variational Autoencoder (VAEs)
![[Pasted image 20250801143649.png|400]]
$x$ is an image, $z$ is latent factors used to generate $x$
e.g. In face generation, $z$ can be how much smile, position of eyes, size of nose, ...
Goal: estimate the true parameters $\theta^*$ of this generative model
Train: maximize likelihood of training data $p_\theta(x)=\int p_\theta(x|z)dz$ 
	$p_\theta(z)$ can be a simple Gaussian prior, $p_\theta(x|z)$ is a decoder neural network
	Problem: cannot compute integral of every $z$
	Solution: define additional encoder network $q_\phi (z|x)$ that approximates $p_\theta (z|x)$ 
Probabilistic encoder/decoder network: 
![[Pasted image 20250801145022.png|600]]
- Encoder: sample $z$ from $z|x \sim N(\mu_{z|x}, \sum_{z|x})$ 
- Decoder: sample $z$ from $x|z \sim N(\mu_{x|z}, \sum_{x|z})$ 
这一页没看太懂 ...
![[Pasted image 20250801145918.png|800]]
Put them together: 
![[Pasted image 20250801150053.png|800]]
### 9.3 GANs
What if we give up on explicitly modeling density, and just want ability to sample? 
—— Take game-theoretic（博弈）approach
![[Pasted image 20250801150614.png|400]]
Train: consider as a two-player game
- Generator network: try to fool the discriminator by generating real-looking images
- Disceiminator network: try to distinguish between real and fake images
- First train the disceiminator, then train the generator
- Minimax game: $min_{\theta_g}max_{\theta_d} E_{x\sim p_{data}}log \begin{bmatrix} D_{\theta_d}(x)+E_{z\sim p(z)}log(1-D_{\theta_d}(G_{\theta_g}(z))) \end{bmatrix}$ 
- ![[Pasted image 20250801151331.png|600]]
![[Pasted image 20250801150846.png|500]]
Pros: beautiful, state-of-the-art samples
Cons: more unstable to train, canoot solve inference queries

## 10 Reinforcement Learning
### 10.1 Reinforcement Learning Method
Overview: 
![[Pasted image 20250803162053.png|500]]
E.g 1: Cart-Pole Problem 
![[Pasted image 20250803162232.png|300]]
Objective: Balance a pole on top of a movable cart
State: angle, angular speed, position, horizontal velocity
Action: horizonal force applied on the cart
Reward: 1 at each time step if the pole is upright

E.g 2: Robot Locomotion
![[Pasted image 20250803162650.png|400]]
Objective: make the robot move forward
State: Angle and position of the joints
Action: Torques applied on joints
Reward: 1 at each time step upright + forward movement
#### Markov Decision Process (MDP)
- Mathmatical formulation of the RL problem
- Define: $(S,A,R,P,\gamma)=(State, Action, Reward, Transition\ probability, Discount\ factor)$   
- Process: Agent selects action $a_t$ -> Env samples reward $r_t\sim R(.|s_t, a_t)$ -> Env samples next state $s_{t+1} \sim P(.|s_t, a_t)$ -> Agent receive reward $r_t$ and next state $s_{t+1}$ 
- $\pi$ : A policy from S to A that specifies what action to take in each state
- Goal: find policy $\pi^*$ that maximize cumulative discounted reward: $\sum_{t>0}\gamma^t r_t$ 
- Solution: Maximize the expected sum of rewards
#### Value function & Q-value function
- To get good state: $V^\pi (s) = E[\sum_{t≥ 0}\gamma^t r_t|s_0 = s,\pi]$ 
- To get good state-action pair: $Q^\pi (s,a) = E[\sum_{t≥ 0}\gamma^t r_t|s_0 = s,a_0=a,\pi]$ → $Q^*(s,a)=max_\pi Q^\pi (s,a)$ 
- Bellman equation: $Q^*(s,a)=E_{s'\sim \epsilon}[r+\gamma max_{a'}Q^*(s',a')|s,a]$  
- Value iteration algorithm: use Bellman equation as an update (Problem: not scalable)
#### Q-Learning
Use a function approximator to estimate the action-value function $Q(s,a;\theta)≈Q^*(s,a)$ 
Deep Q-Learning: the function approximator is a deep neural network 
- Forward pass: $L_i(\theta_i)=E_{s,a\sim \rho(·)}[(y_i-Q(s,a;\theta_i))^2]$ 
- Backward pass: $∇_{\theta_i}L_i(\theta_i)=E_{s,a \sim \rho(·);s'\sim \epsilon}[r+\gamma max_{\alpha'}Q(s',a';\theta_{i-1})-Q(s,a;\theta_i)-Q(s,a;\theta_i)∇_{\theta_i}Q(s,a;\theta_i)$ 
Q-network Architecture: 
![[Pasted image 20250803165507.png|500]]
Using **experience replay**: continually update a replay memory table of transitions $(s_t,a_t,r_t,s_{t+1})$ as game episodes are played
![[Pasted image 20250803165712.png|500]]
Problem: the Q function is complicated
Solution: Policy Gradient
- define a parametrized policies: $\prod = \{\pi_\theta,\theta \in R^m\}$ 
- for each policy, value: $J(\theta)=E[\sum_{t≥0}\gamma^t r_t|\pi_\theta]$  → $\theta^*=argmax_\theta J(\theta)$ 
#### Actor-Critic Algorithm
combine Policy Gradients and Q-Learning by training both an actor(policy) and a critic(the Q-function)
![[Pasted image 20250803170548.png|400]]

Application: Reccurrent Attention Model (RAM)
![[Pasted image 20250803170651.png|600]]、
### 10.2 Hardware Acceleration
Challenge: Model Size, low speed, energy efficiency
Hardware Family: 
- General Purpose
	- CPU: latency oriented
	- GPU: throughput oriented
- Specialized Hardware
	- FPGA: programmable logic
	- ASIC: fixed logic
Number Representation: $(-1)^S × (1.M) × 2^E$ 
![[Pasted image 20250804143222.png|500]]
Int 8: core use of TPU
#### Algorithm for efficient interface
##### Pruning（剪枝）
![[Pasted image 20250804144114.png|200]]
减少参数 → 提高速度（前提：准确率不受很大影响）
##### Weight Sharing
2.09, 2.12, 1.92, 1.87 × → 2.0 √
![[Pasted image 20250804144335.png|200]]
Trained Quantization
![[Pasted image 20250804144657.png|400]]
Huffman Coding
思想：用更多的数位表示不常出现的权重；用更少的数位表示常出现的权重
##### Quantization（量化）
- Gather the statistics for weight and activation
- Choose proper radix point position
- 定点数前向传播，浮点数反向传播

##### Low Rank Approximation
将一个卷积层拆分成两次卷积
##### Binary/Ternary Net
用两个/三个权重表示神经网络
#### Hardware for efficient interface
##### Google TPU 
TPU: high-level chip architecture
- The Matrix Unit: 65,536 (256x256) 8-bit multiply-accumulate units
- 700 MHz clock rate
- Peak: 92T operations/second    
- 4 MiB of on-chip Accumulator memory
- 24 MiB of on-chip Unified Buffer (activation memory)
- 3.5X as much on-chip memory vs GPU
- Two 2133MHz DDR3 DRAM channels
- 8 GiB of off-chip weight DRAM memory
##### EIE Architecture
[韩松EIE: 论文详解](https://blog.csdn.net/weixin_36474809/article/details/85326634)（觉得这篇比视频讲的清楚）
#### Efficient Training -- Algorithms
##### Parallelization
data parallel: run multiple inputs in parallel
parameter update:
![[Pasted image 20250805103509.png|300]]
Model-Parallel Convolution
- by output region (x,y)
- by output map j
- fully connected layers by output activation
Hyper-Parmeter Parallel
##### Mixed Precision with FP16 and FP32
目标：减少占位和能耗
mixed precision: 
![[Pasted image 20250805104118.png|400]]
mixed precision training:
![[Pasted image 20250805104213.png|400]]
##### Model Distillation
Teacher model --train->Student model
need soft label
##### DSD: Dense-Sparse-Dense Training
![[Pasted image 20250805104611.png|400]]
### 10.3 Adversarial examples & training
Why? 
—— Overfitting/too linearly
![[Pasted image 20250806100409.png|300]]![[Pasted image 20250806100429.png|305]]
Modern deep nets are very piecewise linear
Adversarial examples construction
![[Pasted image 20250806101921.png|400]]
黑色是减少的部分，白色是要增加的部分

The fast gradient sign method (FGSM)
$J(\tilde{x},\theta)≈J(x,\theta)+(\tilde{x}-x)^T\nabla_xJ(x)$ -> maximize
限制：$||\tilde{x}-x||_\infty ≤\epsilon$ （像素不能变太多）
-> $\tilde{x}=x+\epsilon sign(\nabla_xJ(x))$

Maps of adversarial and random cross-sections
![[Pasted image 20250806103644.png|400]]
白色：分类是正确的；其他不同颜色：错分类成不同的类别
现象：左半边基本正确，右半边基本错误，分界线近似是线性的
-> FGSM识别到某个方向内积大，就可以得到对抗样本

Insight: 每个对抗样本都靠近某个线性决策边界，一旦进入对抗子空间，所有其他附近的点都是对抗样本
-> 只要你找对方向（能和梯度产生很大内积），沿着方向稍微移动一点就可以欺骗模型，不需要具体空间坐标

RBFs: behave mode intuitively

Transferability Attack
![[Pasted image 20250806111525.png|400]]
Practical Attack
- fool real classifiers trained by remotely hosted API
- fool malware detector networks
- display adversarial examples in the physical world and fool machine learning systems
Training on adversarial examples:
![[Pasted image 20250806112018.png|400]]

Conclusion: attack is easy; defending is difficult. 