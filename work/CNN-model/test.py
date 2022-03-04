import numpy as np
from keras.preprocessing import image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
from tensorflow.keras.models import load_model

# 测试计时
T1 = time.time()

# 加载模型
model = load_model('my_model.h5')

# 加载图片
img=mpimg.imread('normal2.png',0)
imgplot = plt.imshow(img)
test_image = image.load_img('normal2.png', target_size = (224, 224))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)

# 预测结果
result = model.predict(test_image)

if result[0][0] == 1:
    prediction = 'normal'
else:
    prediction = 'covid'

# 输出预测结果
print('预测结果为：',prediction)

# 输出运行时间
T2 = time.time()
print('程序运行时间:%s毫秒' % ((T2 - T1)*1000))

# 保存并展示预测结果
plt.title('Prediction is  '+ prediction)

# 保存结果时要对应上原图名称
plt.savefig('test_result/normal2.jpg')
plt.show()