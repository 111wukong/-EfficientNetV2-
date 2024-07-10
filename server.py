import sys
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


baseurl = "https://landbigdata.swjtu.edu.cn/deep/"  #"http://192.168.31.100:8086/"
#baseurl = "http://192.168.31.200:8086/"
outputpath = "/root/temp/output/"               
inputpath = "/root/temp/input/"
zip_inputpath = "/root/temp/"
font_path = "/root/EfficientNet-PyTorch/simhei.ttf"


"""
function: task check
online server:check runmode=2
parameters：run_docker：algorithm docker 
"""
def checktask(run_docker):
    url = baseurl + "api/aitaskcheck"
    headers = {'Content-Type': 'application/json;charset=UTF-8'}
    jsondata = {"runmode":"2","taskstatus": "2", "rundocker":run_docker}
    try:
        reponse = requests.post(url, headers=headers, data=json.dumps(jsondata))
        return reponse.text
    except Exception as e:
        print("Task detection exception：",e)
        return None


"""
function:task state update
parameters: aitaskid: task id
        taskstatus: task status:{1:队列中,2:预处理中,3:计算中,4:计算完成,5:取值完成，6：计算出错}
        outmessage: result information(json data)
        outpath: output result files(for example:image name;video name;zip name)
"""
def updatetask(aitaskid, taskstatus, outmessage, out_path):

    url = baseurl + "api/aitaskupdate"
    headers = {'Content-Type':'application/json;charset=UTF-8'}
    timestr = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    if outmessage != "" and len(outmessage)>4 :
        jsondata = {"aitaskid": ""+aitaskid+"", "taskstatus": ""+taskstatus+"", "resultdata": outmessage,"resultfile": ""+out_path+ "","overtime": "" +timestr+""}
    else:
        jsondata = {"aitaskid": ""+aitaskid+"", "taskstatus": ""+taskstatus+""}
    try:
        y = json.dumps(jsondata)
        print(y)
        
        reponse = requests.post(url, headers=headers, data = json.dumps(jsondata))
        print(reponse.text)
        return reponse.text
    except Exception as e:
        print("Server Exception...",e)
        
def recognition():
    print("object recognize server start ...")   
    Emode=Eff()  
    print("model load complete ...")
    
    while True:
        text = checktask("nvidia/cuda:10.2-efficientnet")
       # print(text)
        if text != None:
            try:
                task = json.loads(text)
            except Exception as e:
                print("Detection task error:",e)
                continue                
            if str(task['code']) == '843':
                try:
                    task_type = str(task['data']['tasktype'])
                    task_id = str(task['data']['aitaskid'])
                    image_name = str(task['data']['submitfile'])
                    print("读取任务！")
                    updatetask(task_id,"3","","")                  
                    if task_type == '1':
                        source_image_path = inputpath + image_name
                        result_image_path = outputpath + image_name
                        
                        #只修改这一行  输入一张图片，输出效果图片和json
                        print("开始识别1！")
#                         r_image = img = cv2.imread(source_image_path)
                        result = Emode.predict(source_image_path)
                        print("识别结束1！") 
                        outmessage = result                     
                        shutil.copyfile(source_image_path,result_image_path)                       
                        print("source_image_path：",source_image_path)
                        
                        outmessage=json.dumps(outmessage)
                        
                        updatetask(task_id,"4",outmessage,image_name)
                        
                    elif task_type == '2':
                        source_image_path = inputpath + image_name
                        result_image_path = outputpath + image_name
                        
                        print("开始识别2！")
                        result = meter_reader.predict(source_image_path, 
                                            save_dir='meter_model/meter_read/output/')

                        print("mp4 outmessage:")
                        outmessage = result
                        outmessage=json.dumps(outmessage)
                        sf="meter_model/meter_read/output/result.jpg"
                        shutil.copyfile(sf,result_image_path)                       
                        print("source_image_path：",source_image_path)
                        updatetask(task_id,"4",outmessage,image_name)                          
                    else:
                        message = "不支持此类型数据预测"
                        outmessage = "{\"类型错误\":\""+str(message)+ "\"}"
                        updatetask(str(task_id),"6",json.dumps(outmsg),image_name)
                except Exception as e:
                    outmsg = {"error information": " image inference error:{0}".format(e)}
                    print("Algorithm running error,error is: ",e)
                    updatetask(str(task_id),"6",outmsg,image_name)                   
        time.sleep(1)
            
if __name__ == '__main__':
    recognition()    
   