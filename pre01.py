import os
import json
import cv2

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
        



from model import efficientnetv2_s as create_model


#测试图片
class Eff():
    def __init__(self):       
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
       
        
        # read class_indict
        json_path = './class_indices.json'
        assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
        with open(json_path, "r") as f:
            self.class_indict = json.load(f)
        # create model
        self.model = create_model(num_classes=5).to(self.device)
        # load model weights
        model_weight_path = "./weights/model-399.pth"
        self.model.load_state_dict(torch.load(model_weight_path, map_location=self.device))
        self.model.eval()

    def forword(self,img_path):
        img = cv2.imread(img_path) #读取图像
        # plt.imshow(img) # 显示原图
        # cv2.arrowedLine参数概述 
        # cv2.arrowedLine( 输入图像，起始点(x,y)，结束点(x,y)，线段颜色，线段厚度，线段样式，位移因数， 箭头因数)
        img = cv2.arrowedLine(img, (500,1750), (500,1000), (0,0,255),10,3,0,0.5)
        plt.imshow(img) # 显示添加箭头线段后的图片
    
    def predict(self,img_path):
        img_size = {"s": [300, 384],  # train_size, val_size
                "m": [384, 480],
                "l": [384, 480]}
        num_model = "s"          
        assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
        img = Image.open(img_path)
       # plt.imshow(img)
        # [N, C, H, W]
        data_transform = transforms.Compose(
        [transforms.Resize(img_size[num_model][1]),
         transforms.CenterCrop(img_size[num_model][1]),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        img = data_transform(img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)
        with torch.no_grad():
            # predict class
            output = torch.squeeze(self.model(img.to(self.device))).cpu()
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()
       # print_res = "aclass: {}   prob: {:.3}".format(self.class_indict[str(predict_cla)],
       #                                          predict[predict_cla].numpy())
        print_res={"direct":""+self.class_indict[str(predict_cla)]+"","rate":"{:.3}".format(predict[predict_cla].numpy())}
        
        print(print_res)
        return print_res

if __name__ == '__main__':
    img_path = "./pre_img/img1420.jpg"
    test_img=Eff()   
    
    
    test_img.predict(img_path)
    test_img.forword(img_path)