from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
import glob
import numpy as np
import os.path as path
import os
import cv2
from matplotlib.font_manager import FontProperties # 中文字體
from PIL import ImageTk, Image
import matplotlib.image as mpimg                   # 匯入image 類別，並設定為 mpimg

# 換成中文的字體
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False  # 步驟二（解決座標軸負數的負號顯示問題）


# openCV 讀取圖片 轉 XY (圖片路徑,圖片寬,圖片長,圖片顏色)
def AI_loadImages(IMAGEPATH="imgaes", imageW=64, imageH=64, imageC=1,dimX=1):
    """
    IMAGEPATH = 'images'
    imageW = 64
    imageH = 64
    imageC = 1
    dimX = 1   # MLP
    # dimX=4   # CNN
    """

    dirs = os.listdir(IMAGEPATH)    # 取得images/ 底下所有文件夾
    imgX = []
    imgY = []
    for i in range(0, len(dirs)):
        print("Label:", i, "=", dirs[i])
    i = 0
    for name in dirs:

        # 抓取 文件夾底下所有檔案
        file_paths = glob.glob(path.join(IMAGEPATH+"/"+name, '*.*'))

        # 一個一個檔案處理
        for path3 in file_paths:
            try:
                # print(path3)      # 顯示讀到的檔案名稱
                img = cv2.imread(path3)         # 打開圖片 (如果有不是圖片的檔案 會當機)
                img = cv2.resize(img, (imageW, imageH), interpolation=cv2.INTER_AREA)
                if imageC == 3:
                    im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)       # 彩色
                elif imageC == 1:
                    im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)      # 彩色轉灰階
                imgX.append(im_rgb)
                imgY.append(i)     # 第幾個文件夾
            except:
                print("error:", path3, " <------------------")
        i = i+1

    # 降維度
    X = np.asarray(imgX)
    Y = np.asarray(imgY)

    if dimX == 1:
        X = X.reshape(X.shape[0], imageW*imageH*imageC)   # MLP
    else:
        X = X.reshape(X.shape[0], imageW, imageH, imageC)   # CNN

    # 標準化輸入資料
    X = X.astype('float32')
    X = X / 255

    # 資料拆分
    category = len(dirs)  # 多少答案(種類)
    dim = X.shape[1]  # 有多少特徵值
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.05)
    print(x_train.shape)
    print(y_train.shape)

    # 回傳 拆分資料 X Y , 圖片答案個數,特徵值, 讀取圖片的X Y ,圖片Label
    return x_train, x_test, y_train, y_test, category, dim, imgX, imgY, dirs


x_train, x_test, y_train, y_test, category, dim, imgX, imgY, dirs = AI_loadImages("Cars Dataset", 64, 64, 3, 4)


# 將數字轉為 One-hot 向量-----------------------------------
y_train2 = tf.keras.utils.to_categorical(y_train, category)
y_test2 = tf.keras.utils.to_categorical(y_test, category)
print("y_train2 to_categorical shape=", y_train2.shape)     # (290,6)

# 圖片產生
newCarPic = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=8, width_shift_range=0.08, shear_range=0.3,
                                                            height_shift_range=0.08, zoom_range=0.08)

# 建立模型  ----------------------------
model = tf.keras.models.Sequential()

# CNN 訓練 第一層
model.add(tf.keras.layers.Conv2D(filters=3,
                                 kernel_size=(3, 3),
                                 padding="same",
                                 activation='relu',
                                 input_shape=(64, 64, 3)))

# 第二層處理
model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

# 第三層處理
model.add(tf.keras.layers.Conv2D(filters=9,
                                 kernel_size=(2, 2),
                                 padding="same",
                                 activation='relu'))

# 第三層處理
model.add(tf.keras.layers.Dropout(rate=0.5))    # 丟掉 50% 的圖
model.add(tf.keras.layers.Conv2D(filters=6,
                                 kernel_size=(2, 2),
                                 padding="same",
                                 activation='relu'))


model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(100, activation='relu'))
model.add(tf.keras.layers.Dense(units=category,
                                activation=tf.nn.softmax))

model.compile(optimizer='adam',
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=['accuracy'])

# 顯示模型
model.summary()

train_generator = newCarPic.flow(x_train, y_train2, batch_size=10)

# 讀取模型架構
try:

 #   # 保存模型架構
  #  with open("model_ImageDataGenerator_彩色.json", "w") as json_file:
   #     json_file.write(model.to_json())

    with open('model_ImageDataGenerator_彩色.h5', 'r') as load_weights:
        # 讀取模型權重
        json_file = open("model_ImageDataGenerator_彩色.json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model.load_weights("model_ImageDataGenerator.h5")

except IOError:
    print("File not accessible")

checkpoint = tf.keras.callbacks.ModelCheckpoint("model_ImageDataGenerator_彩色.h5", monitor='accuracy', verbose=1,
                                                save_best_only=True, mode='auto', save_freq=1)

# 保存模型架構
with open("model_ImageDataGenerator_彩色.json", "w") as json_file:
          json_file.write(model.to_json())

# 訓練模型
history = model.fit(train_generator,    # 進行訓練的因和果的資料
                    epochs=50,          # 設定訓練的次數，也就是機器學習的次數
                    callbacks=[checkpoint]
                    )

# 測試
score = model.evaluate(x_test, y_test2, batch_size=128)        # 計算測試正確率
print("資料遺失率LOSS:", round(score[0], 8), "資料準確率:", score[1])         # 輸出測試正確率

# 預測
predict = model.predict(x_test)
x_test2 = x_test * 255

ans = ""
for x in range(0, 10):
    ans = ans + str(np.argmax(predict[x])) + " "
print("預測答案(前10筆):", ans)           # 輸出預測答案
print("原始答案(前10筆)", y_test[:10])    # 實際測試的結果


# 顯示預測的前九張圖片 (預測資料,圖片答案名稱(Label), nrow*cols 要顯示幾乘幾的圖, imageW imageH要轉回2維的數值)
def AI_ShowManyImages(predict, dirs, nrow=3, cols=3, imageW=64, imageH=64, color=1):
    """
    imageW=64
    imageH=64
    nrow=3
    cols=3
    color=1  # 1 MLP  3(CNN)
    """
    # print(plt.style.available)  # 顯示圖片顏色樣式
    # plt.style.use('_mpl-gallery')   # 顯示圖片的顏色
    fig, axs = plt.subplots(nrows=nrow, ncols=cols, figsize=(nrow, cols))

    for t1 in range(nrow):
        for t2 in range(cols):
            i = (t1*cols)+t2
            t3 = x_test[i].reshape(imageW, imageH, color)   # 轉回2維維度
            axs[t1, t2].imshow(t3)                   # 顯示圖片
            ans = np.argmax(predict[i], axis=-1)
            str1 = "預測為:"+str(ans) + " , "+str(dirs[ans])
            axs[t1, t2].set_title(str1)              # 顯示圖片

    # 存圖片
    # plt.savefig('my.png')   # 存圖片
    plt.tight_layout()
    plt.show()  # 繪製


AI_ShowManyImages(predict, dirs, 3, 3, 64, 64, 3)
