## 代码使用简介

1. 下载好数据集，下载地址:【超级会员V5】通过百度网盘分享的文件：swdata.z…
    链接:https://pan.baidu.com/s/13py_RjUj0MotVWytEm1i2Q?pwd=8kzg 
    提取码:8kzg
    复制这段内容打开「百度网盘APP 即可获取」(建议自己制作数据集)
2. 在`train.py`脚本中将`--data-path`设置成解压后的`swdatasets`文件夹绝对路径
3. 下载预训练权重，根据自己使用的模型下载对应预训练权重: https://pan.baidu.com/s/1ouX0UmjCsmSx3ZrqXbowjw  密码: 090i
4. 在`train.py`脚本中将`--weights`参数设成下载好的预训练权重路径
5. 设置好数据集的路径`--data-path`以及预训练权重的路径`--weights`就能使用`train.py`脚本开始训练了(训练过程中会自动生成`class_indices.json`文件)
6. 在`predict.py`脚本中导入和训练脚本中同样的模型，并将`model_weight_path`设置成训练好的模型权重路径(默认保存在weights文件夹下)
7. 在`predict.py`脚本中将`img_path`设置成你自己需要预测的图片绝对路径
8. 设置好权重路径`model_weight_path`和预测的图片路径`img_path`就能使用`predict.py`脚本进行预测了
9. 如果要使用自己的数据集，请按照花分类数据集的文件结构进行摆放(即一个类别对应一个文件夹)，并且将训练以及预测脚本中的`num_classes`设置成你自己数据的类别数

### 封装接口

- 模型训练完成过后，按照如下方式封装成接口，供移动端调用。
- 具体代码参考server.py

1. **添加 Flask 相关代码**：
   - 使用 `Flask` 创建一个服务端。
   - 将识别功能放在 `/recognize` 路由中。
2. **修改循环机制**：
   - 将任务检查和处理逻辑移到 Flask 路由的处理函数中。
   - 在 `recognize` 函数中，原本无限循环的逻辑可以在 Flask 的处理函数中实现（为了演示方便，保留了循环逻辑）。
3. **启动 Flask 应用**：
   - 修改 `if __name__ == '__main__':` 部分，启动 Flask 应用。

### 使用说明

1. **启动 Flask 应用**：

   ```
   python3 server.py
   ```

   该应用将运行在 `0.0.0.0` 的 5000 端口。

2. **发送请求**： 您可以通过 HTTP POST 请求访问 `http://localhost:5000/recognize` 来启动识别任务。

![image-20240710213042975](./../../../AppData/Roaming/Typora/typora-user-images/image-20240710213042975.png)
