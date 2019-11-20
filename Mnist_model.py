import keras.utils as np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dense, Dropout, Flatten

# 定义超参数
# 总共有多少个类别
num_classes = 10
# 训练迭代次数
epochs = 1
# 训练批次
batch_size = 32

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 数据预处理
x_train = x_train.reshape(-1, 28, 28, 1) / 255.
x_test = x_test.reshape(-1, 28, 28, 1) / 255.
# 转换成独热编码
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)

# 定义一个顺序模型对象
model = Sequential()

# 添加卷积层
model.add(Conv2D(
    # 输入维度，第一层卷积必须要，后面可以不加。None代表不确定输入个数。
    batch_input_shape=(None, 28, 28, 1),
    # 卷积核(滤波器，过滤器，权重)的个数
    filters=32,
    # 卷积核的大小(长和宽)，也可以传一个元组(5, 5)，是一样的
    kernel_size=5,
    # 补0
    padding='same',
))
# 添加激活函数relu，刚开始图像识别部分的建模全都用relu
model.add(Activation('relu'))
# 添加池化层（这是最大池化，还有个AvgPool2D平均池化）
model.add(MaxPooling2D(
    # 池化核大小为2，一般也是设置成2
    pool_size=2,
))
# 添加Dropout层，随机丢掉25%的数据，抑制过拟合
model.add(Dropout(0.25))

model.add(Conv2D(
    filters=256,
    kernel_size=5,
    padding='same'
))
model.add(Activation('relu'))
model.add(MaxPooling2D(
    pool_size=2,
))
model.add(Dropout(0.25))

# 将多维度的数据铺平成一维的
model.add(Flatten())
# 全连接层
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# 最后输出10层，因为有10个分类
model.add(Dense(num_classes))
# 多分类，单标签的情况，输出层的激活函数都是softmax
model.add(Activation('softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

loss, accuracy = model.evaluate(x_test, y_test)

print('loss: ', loss)
print('accuracy: ', accuracy)

model.save('model.h5')
