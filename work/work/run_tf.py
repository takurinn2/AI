import numpy as np
from keras.preprocessing import image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
from keras.models import load_model
import os

def run_tf(pic_path,out_path,model_path="my_model.h5") :
    # 测试计时
    T1 = time.time()

    # 加载模型
    model = load_model(model_path)

    if not os.path.isfile(pic_path) :
        return {"code" : 100}
    out_base_path = os.path.dirname(out_path)
    os.makedirs(out_base_path,exist_ok=True)

    # 加载图片
    img=mpimg.imread(pic_path,0)
    imgplot = plt.imshow(img)
    test_image = image.load_img(pic_path, target_size = (224, 224))
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
    plt.savefig(out_path)
    # plt.show()

    return {
        "code" : 200,
        "prediction" : prediction,
        "out_path" : out_path
    }

if __name__ == "__main__" :
    input_path = r"C:\Users\86151\Desktop\学习\数据科学\CNN-model\normal2.png"
    out_path = r"C:\Users\86151\Desktop\学习\数据科学\CNN-model\out_path\out_pic.png"
    run_tf(input_path,out_path)