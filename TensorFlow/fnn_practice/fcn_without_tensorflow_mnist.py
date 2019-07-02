import numpy as np
import optim
import os
import pickle
import matplotlib.pyplot as plt


class TwoLayerNet(object):

    def __init__(self, input_dim=3 * 32 * 32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        初始化神经网络模型
        Inputs:
        - input_dim: 单个数据的大小 (3*32*32)
        - hidden_dim: 隐藏层的个数 即神经元的个数 100
        - num_classes: 类型的个数 10
        - dropout: Scalar between 0 and 1 giving dropout strength.
        - weight_scale: 初始化权重参数W 0.001
        - reg: 初始化正则惩罚项  λ  0
        """
        self.params = {}
        self.reg = reg
        # np.random.randn(input_dim, hidden_dim) 生成（3*32*32，100）的[0,1)之间的数据，包含0，不包含1
        # 生成W1 --> (3072, 100)
        self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dim)
        # np.zeros((1, hidden_dim)) 返回来一个给定形状和类型的用0填充的数组
        # 生成b1 --> (1,100)
        self.params['b1'] = np.zeros((1, hidden_dim))
        # 生成W2 --> (100, 10)
        self.params['W2'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        # 生成b2 --> (1,10)
        self.params['b2'] = np.zeros((1, num_classes))

    def loss(self, X, y=None):
        """
        计算小批数据的损失值和梯度。
        Inputs:
        - X: 输入层的输入数据
        - y: 输入数据的标签.
        Returns:
        如果没有标签，进行前向测试并返回测试结果
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
                  scores[i, c] is the classification score for X[i] and class c.
        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
                 names to gradients of the loss with respect to those parameters.
        """

        """正向传播"""
        # 权重参数赋值
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']

        # output1 = ReLU(W1*X+b1)
        h1, cache1 = affine_relu_forward(X, W1, b1)
        # output2 = (W2*output1+b2)
        out, cache2 = affine_forward(h1, W2, b2)
        scores = out  # (N,C)
        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores
        loss, grads = 0, {}
        # 输出结果归一化操作，并求取损失值(交叉熵损失函数)
        data_loss, dscores = softmax_loss(scores, y)
        # 惩罚项的损失值（L2范数）
        reg_loss = 0.5 * self.reg * np.sum(W1 * W1) + 0.5 * self.reg * np.sum(W2 * W2)
        # 总体的损失值
        loss = data_loss + reg_loss

        """反向传播"""
        dh1, dW2, db2 = affine_backward(dscores, cache2)
        dX, dW1, db1 = affine_relu_backward(dh1, cache1)
        # Add the regularization gradient contribution
        dW2 += self.reg * W2
        dW1 += self.reg * W1
        grads['W1'] = dW1
        grads['b1'] = db1
        grads['W2'] = dW2
        grads['b2'] = db2

        return loss, grads


class Solver(object):
    def __init__(self, model, data, **kwargs):
        """
        :param model: 训练模型
        :param data: 训练数据集
        :param kwargs:
            lr_decay=0.95, # 学习率衰减因子
            print_every=100,
            num_epochs=40,
            batch_size=400,
            update_rule='sgd_momentum', # 梯度下降算法（本文使用随机梯度下降算法）
            optim_config={'learning_rate': 5e-4, 'momentum': 0.9})
        """
        self.model = model
        self.X_train = data['X_train']
        self.y_train = data['y_train']
        self.X_val = data['X_val']
        self.y_val = data['y_val']

        # 解压缩关键字参数
        self.update_rule = kwargs.pop('update_rule', 'sgd')
        self.optim_config = kwargs.pop('optim_config', {})
        self.lr_decay = kwargs.pop('lr_decay', 1.0)
        self.batch_size = kwargs.pop('batch_size', 100)
        self.num_epochs = kwargs.pop('num_epochs', 10)
        self.print_every = kwargs.pop('print_every', 10)
        self.verbose = kwargs.pop('verbose', True)

        # 如果有额外的关键字参数，则抛出错误
        if len(kwargs) > 0:
            extra = ', '.join('"%s"' % k for k in kwargs.keys())
            raise ValueError('Unrecognized arguments %s' % extra)

        # 确保神经网络优化器存在，然后用实际函数替换字符串名称
        # 神经网络优化器，主要是为了优化我们的神经网络，使他在我们的训练过程中快起来，
        # 节省社交网络训练的时间。在pytorch中提供了torch.optim方法优化我们的神经网络，
        # torch.optim是实现各种优化算法的包。最常用的方法都已经支持，接口很常规，
        # 所以以后也可以很容易地集成更复杂的方法。
        # SGD是最基础的优化方法，普通的训练方法, 需要重复不断的把整套数据放入神经网络NN中训练,
        # 这样消耗的计算资源会很大.当我们使用SGD会把数据拆分后再分批不断放入
        # NN中计算.每次使用批数据, 虽然不能反映整体数据的情况, 不过却很大程度上加速了
        # NN的训练过程, 而且也不会丢失太多准确率.

        if not hasattr(optim, self.update_rule):
            raise ValueError('Invalid update_rule "%s"' % self.update_rule)
        self.update_rule = getattr(optim, self.update_rule)

        self._reset()

    def _reset(self):
        """
        设置一些簿记变量进行优化。不要手动调用它。
        """
        self.epoch = 0
        self.best_val_acc = 0
        self.best_params = {}
        self.loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.optim_configs = {}
        # 为每个参数创建optim_config的深copy, self.model.params包含权重参数 截距项的字典
        for p in self.model.params:
            d = {k: v for k, v in self.optim_config.items()}
            self.optim_configs[p] = d

    def _step(self):
        """
        梯度更新。由train()调用。
        """
        # 制作一个小批量的培训数据
        num_train = self.X_train.shape[0]
        #  np.random.choice(5, 3) --> array([0, 3, 4])
        #  np.arange(5) 中产生一个size为3的随机采样：
        batch_mask = np.random.choice(num_train, self.batch_size)
        X_batch = self.X_train[batch_mask]
        y_batch = self.y_train[batch_mask]

        # 计算loss和梯度
        loss, grads = self.model.loss(X_batch, y_batch)
        self.loss_history.append(loss)

        # 执行参数更新
        for p, w in self.model.params.items():
            dw = grads[p]
            config = self.optim_configs[p]
            next_w, next_config = self.update_rule(w, dw, config)
            self.model.params[p] = next_w
            self.optim_configs[p] = next_config

    def check_accuracy(self, X, y, num_samples=None, batch_size=100):
        """
        根据所提供的数据检查模型的准确性.
        Inputs:
        - X: 输入数据
        - y: 输入数据的标签值
        - num_samples: 如果没有，则对数据进行子采样，只测试模型在num_samples据点
        - batch_size: 将X和y分成这种大小的批次，以避免使用太多内存。
        Returns:
        - acc: 标量，给出模型正确分类的实例的比例
        """

        N = X.shape[0]
        if num_samples is not None and N > num_samples:
            mask = np.random.choice(N, num_samples)
            N = num_samples
            X = X[mask]
            y = y[mask]

        # 分批计算预测值
        num_batches = N / batch_size
        if N % batch_size != 0:
            num_batches += 1
        y_pred = []
        for i in range(int(num_batches)):
            start = i * batch_size
            end = (i + 1) * batch_size
            scores = self.model.loss(X[start:end])
            y_pred.append(np.argmax(scores, axis=1))
        # 函数原型：numpy.hstack(tup)，其中tup是arrays序列，阵列必须具有相同的形状，除了对应于轴的维度（默认情况下，第一个）。
        # 等价于np.concatenate（tup,axis=1）
        y_pred = np.hstack(y_pred)
        # np.mean(x==y)#返回条件成立的占比
        # np.mean(x)#均值
        acc = np.mean(y_pred == y)
        return acc

    def train(self):
        """
        训练模型。
        """
        # 我们有2000个数据，分成4个batch，那么batch size就是500。
        # 运行所有的数据进行训练，完成1个epoch，需要进行4次iterations。
        # 获取要训练的数据集
        num_train = self.X_train.shape[0]  # X_train.shape(5000,3,32,32)
        # max() 方法返回给定参数的最大值，参数可以为序列。
        # 将整个数据集分为多个batch,不足一个的按一个算，每个patch代表训练一轮
        iterations_per_epoch = max(num_train / self.batch_size, 1)  # (5000/400=12.5)
        num_iterations = self.num_epochs * iterations_per_epoch  # (40*12.5=500)
        for t in range(int(num_iterations)):
            self._step()
            # Maybe print training loss
            if self.verbose and t % self.print_every == 0:
                print('(Iteration %d / %d) loss: %f' % (
                    t + 1, num_iterations, self.loss_history[-1]))

            # 在每个epoch结束时，增加epoch计数器，降低学习速度。
            epoch_end = (t + 1) % iterations_per_epoch == 0
            # 如果epoch训练结束（更新学习率）
            if epoch_end:
                self.epoch += 1
                for k in self.optim_configs:
                    self.optim_configs[k]['learning_rate'] *= self.lr_decay

            # 在第一次迭代、最后一次迭代和每个epoch结束时检查训练和val的准确性
            first_it = (t == 0)
            last_it = (t == 87 + 1)
            if first_it or last_it or epoch_end:
                train_acc = self.check_accuracy(self.X_train, self.y_train, num_samples=1000)
                val_acc = self.check_accuracy(self.X_val, self.y_val)
                self.train_acc_history.append(train_acc)
                self.val_acc_history.append(val_acc)
                if self.verbose:
                    print('(Epoch %d / %d) train acc: %f; val_acc: %f' % (
                        self.epoch, self.num_epochs, train_acc, val_acc))

                # 跟踪最好的模型
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.best_params = {}
                    for k, v in self.model.params.items():
                        self.best_params[k] = v.copy()
        # 在训练结束时，将最好的参数转换到模型中
        self.model.params = self.best_params


def affine_relu_forward(x, w, b):
    """
    Convenience layer that perorms an affine transform followed by a ReLU

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, fc_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, relu_cache)
    return out, cache


def affine_relu_backward(dout, cache):
    """
    Backward pass for the affine-relu convenience layer
    """
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.
    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.
    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)
    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    # Reshape x into rows
    # 建立一个4×2的矩阵c, c.shape[0]为第一维的长度，c.shape[1]为第二维的长度。
    N = x.shape[0]
    # 先前我们不知道x的shape属性是多少，但是想让z变成只有1列，行数不知道多少，
    # 通过`z.reshape(-1,1)`，Numpy自动计算出有16行，新的数组shape属性为(16, 1)，与原来的(4, 4)配套。
    # 将四维数组变为二维数组
    x_row = x.reshape(N, -1)  # (N,D)
    out = np.dot(x_row, w) + b  # (N,M)
    cache = (x, w, b)

    return out, cache


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).
    Input:
    - x: Inputs, of any shape
    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    out = ReLU(x)
    cache = x

    return out, cache


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
         0 <= y[i] < C
    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    # numpy.exp()：返回e的幂次方，e是一个常数为2.71828
    # np.max(axis=1) 每一行取出最大的那一个数（axis=1）
    # keepdims主要用于保持矩阵的二维特性
    probs = np.exp(x - np.max(x, axis=1, keepdims=True))
    # np.sum(probs, axis=1, keepdims=True) 以竖轴为基准 ，同行相加，并保持矩阵的二维特性
    probs /= np.sum(probs, axis=1, keepdims=True)

    N = x.shape[0]
    # 交叉熵损失函数
    # np.log(probs[np.arange(N), y]) 以e为底，求probs[np.arange(N), y]中每一个元素的对数
    # 多维数组的索引 probs[np.arange(N), y]
    # y为标签值，利用其中的一个技巧，ps y=2,那么就去取归一化后的第一个样本中，为2的概率值，正好是索引2
    loss = -np.sum(np.log(probs[np.arange(N), y])) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N

    return loss, dx


def ReLU(x):
    """ReLU non-linearity."""
    # 将比0 小的数字变为0
    return np.maximum(0, x)


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).
    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout
    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    dx = dout
    dx[x <= 0] = 0

    return dx


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.
    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)
    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    # dot()返回的是两个数组的点积
    dx = np.dot(dout, w.T)  # (N,D)
    dx = np.reshape(dx, x.shape)  # (N,d1,...,d_k)
    x_row = x.reshape(x.shape[0], -1)  # (N,D)
    dw = np.dot(x_row.T, dout)  # (D,M)
    db = np.sum(dout, axis=0, keepdims=True)  # (1,M)

    return dx, dw, db


def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        # pickle 模块用于数据序列化与反序列化，pickle是完全用Python来实现的模块，
        # CPickle是用C来实现的，它的速度要比pickle快好多倍，一般建议如果电脑中只要有CPickle的话都应该使用它。
        # pickle.dump()用于序列化，即将python对象转化可储存的二进制数据
        # pickle.load()用于反序列化，即将可储存的二进制数转化据python对象
        datadict = pickle.load(f, encoding="latin1")
        X = datadict['data']  # (10000,3072)
        Y = datadict['labels']
        # print(type(X)) --> <class 'numpy.ndarray'>
        # reshape()函数用于改变数组对象的形状（10000，3072）--> (10000,3,32,32)
        # 修改后新生成的数组与原数组共用一个内存，修改后的数组元素个数与原数组元素个数必须保持一致，若不一致，将会报错：
        # numpy模块是用C语言编写的，因此计算机在处理 ndarray 对象时的速度要远快于 list 对象。
        # 函数transpose(1,0,2)，即代表将轴0和1对换，轴2不变，
        # 亦即将arr[x][y][z]中x和y位置互换，即元素12变为arr[0][1][0]，元素22变为arr[1][2][2]，
        # transpose(0,1,2,3)-->transpose(0,2,3,1) 代表0轴不变，2轴变为1轴，3轴变为2轴，1轴变为2轴，即(10000,3,32,32)--->(10000,32,32,3)
        # astype("float") 修改数组的数据类型为float类型
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        Y = np.array(Y)
        return X, Y


def load_CIFAR10(ROOT):
    """ 加载原始数据（共五个训练集,一个测试集合）"""
    xs = []
    ys = []
    for b in range(1, 6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b,))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    # 按轴axis连接array组成一个新的array
    # a = [
    #     [1,2],
    #     [3,4]
    # ]
    # b = [
    #     [5,6]
    # ]
    # c = np.concatenate((np.array(a),np.array(b)))
    # c = [[1,2][3,4][5,6]]
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte


def get_CIFAR10_data(num_training=5000, num_validation=500, num_test=500):
    """
 获取数据
    """
    # 加载原始数据并处理为训练以及测试用的数组
    cifar10_dir = 'C:/Users/xiaomi/Desktop/DeepLearning/Cifar_NNWithoutTensorFlow/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
    """
    # 二次抽样样品(5000,5500) 即从50000个样品中取得第5000个开始至5500结束的500个样品,
    # 组成新的验证集合 （50000，32，32，3）-->(500,32,32,3)
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    # 组成新的训练集合（50000，32，32，3）-->(5000,32,32,3)
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    # 组成新的测试集合(10000, 32, 32, 3)-->(500,32,32,3)
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]
    # 输入数据标准化，减去均值，ps:只对输入数据进行标准化，标签不进行操作
    # np.mean(a, axis=0) # axis=0，计算每一列的均值
    """

    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    # Transpose so that channels come first
    X_train = X_train.transpose(0, 3, 1, 2).copy()
    X_val = X_val.transpose(0, 3, 1, 2).copy()
    X_test = X_test.transpose(0, 3, 1, 2).copy()

    # Package data into a dictionary
    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
    }


if __name__ == "__main__":
    # 获取数据
    data = get_CIFAR10_data()
    # 搭建模型（前向传播）
    model = TwoLayerNet(reg=0.9)
    # 模型训练
    solver = Solver(model, data,
                    lr_decay=0.95,
                    print_every=100,
                    num_epochs=40,
                    batch_size=400,
                    update_rule='sgd_momentum',
                    optim_config={'learning_rate': 5e-4, 'momentum': 0.9})
    solver.train()

    plt.subplot(2, 1, 1)
    plt.title('Training loss')
    plt.plot(solver.loss_history, 'o')
    plt.xlabel('Iteration')

    plt.subplot(2, 1, 2)
    plt.title('Accuracy')
    plt.plot(solver.train_acc_history, '-o', label='train')
    plt.plot(solver.val_acc_history, '-o', label='val')
    plt.plot([0.5] * len(solver.val_acc_history), 'k--')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    plt.gcf().set_size_inches(15, 12)
    plt.show()

    best_model = model
    y_test_pred = np.argmax(best_model.loss(data['X_test']), axis=1)
    y_val_pred = np.argmax(best_model.loss(data['X_val']), axis=1)

    print('Validation set accuracy: ', (y_val_pred == data['y_val']).mean())
    print('Test set accuracy: ', (y_test_pred == data['y_test']).mean())
