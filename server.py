from flask import Flask, request, jsonify
import torch
import torchvision
from PIL import Image
import cv2
import requests
import json
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import zipfile
import os
import shutil
from pathlib import Path
from pre01 import Eff

app = Flask(__name__)
CORS(app)
baseurl = "https://landbigdata.swjtu.edu.cn/deep/"  #"http://192.168.31.100:8086/"
outputpath = "/root/temp/output/"
inputpath = "/root/temp/input/"
zip_inputpath = "/root/temp/"
font_path = "/root/EfficientNet-PyTorch/simhei.ttf"


def checktask(run_docker):
    url = baseurl + "api/aitaskcheck"
    headers = {'Content-Type': 'application/json;charset=UTF-8'}
    jsondata = {"runmode": "2", "taskstatus": "2", "rundocker": run_docker}
    try:
        response = requests.post(url, headers=headers, data=json.dumps(jsondata))
        return response.text
    except Exception as e:
        print("Task detection exception：", e)
        return None


def updatetask(aitaskid, taskstatus, outmessage, out_path):
    url = baseurl + "api/aitaskupdate"
    headers = {'Content-Type': 'application/json;charset=UTF-8'}
    timestr = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    if outmessage != "" and len(outmessage) > 4:
        jsondata = {"aitaskid": "" + aitaskid + "", "taskstatus": "" + taskstatus + "", "resultdata": outmessage,
                    "resultfile": "" + out_path + "", "overtime": "" + timestr + ""}
    else:
        jsondata = {"aitaskid": "" + aitaskid + "", "taskstatus": "" + taskstatus + ""}
    try:
        y = json.dumps(jsondata)
        print(y)
        response = requests.post(url, headers=headers, data=json.dumps(jsondata))
        print(response.text)
        return response.text
    except Exception as e:
        print("Server Exception...", e)


@app.route('/recognize', methods=['POST'])
def recognize():
    print("object recognize server start ...")
    Emode = Eff()
    print("model load complete ...")

    while True:
        text = checktask("nvidia/cuda:10.2-efficientnet")
        if text is not None:
            try:
                task = json.loads(text)
            except Exception as e:
                print("Detection task error:", e)
                continue
            if str(task['code']) == '843':
                try:
                    task_type = str(task['data']['tasktype'])
                    task_id = str(task['data']['aitaskid'])
                    image_name = str(task['data']['submitfile'])
                    print("读取任务！")
                    updatetask(task_id, "3", "", "")
                    if task_type == '1':
                        source_image_path = inputpath + image_name
                        result_image_path = outputpath + image_name

                        print("开始识别1！")
                        result = Emode.predict(source_image_path)
                        print("识别结束1！")
                        outmessage = result
                        shutil.copyfile(source_image_path, result_image_path)
                        print("source_image_path：", source_image_path)

                        outmessage = json.dumps(outmessage)

                        updatetask(task_id, "4", outmessage, image_name)

                    elif task_type == '2':
                        source_image_path = inputpath + image_name
                        result_image_path = outputpath + image_name

                        print("开始识别2！")
                        result = meter_reader.predict(source_image_path,
                                                      save_dir='meter_model/meter_read/output/')

                        print("mp4 outmessage:")
                        outmessage = result
                        outmessage = json.dumps(outmessage)
                        sf = "meter_model/meter_read/output/result.jpg"
                        shutil.copyfile(sf, result_image_path)
                        print("source_image_path：", source_image_path)
                        updatetask(task_id, "4", outmessage, image_name)
                    else:
                        message = "不支持此类型数据预测"
                        outmessage = "{\"类型错误\":\"" + str(message) + "\"}"
                        updatetask(str(task_id), "6", json.dumps(outmsg), image_name)
                except Exception as e:
                    outmsg = {"error information": " image inference error:{0}".format(e)}
                    print("Algorithm running error, error is: ", e)
                    updatetask(str(task_id), "6", outmsg, image_name)
        time.sleep(1)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
