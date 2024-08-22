import sys
import os
import torch
import torch.nn as nn
import cv2
import numpy as np
import pandas as pd
import time
import tkinter as tk
from PIL import Image,ImageTk
from torchvision.ops import nms
sys.path.append('C:/Users/Mehmet/yolov7')
from yolov7.models.experimental import attempt_load
from yolov7.utils.torch_utils import select_device


weights_path = r'C:\Users\Mehmet\yolov7\yolov7.pt'
device = select_device('') 
model = attempt_load(weights_path, map_location=device)
classes = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "TV",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush"
]

#kütüphane ve genel tanımlar

def predicted_objects(frame):
        original_height, original_width = frame.shape[:2]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame,(640,640))
        frame = frame/255
        input= torch.from_numpy(frame).float()
        input=input.permute(2, 0, 1).unsqueeze(0)
        input=input.to(device)

        with torch.no_grad():
            preds = model(input)[0]  
            for boxs in preds: #tahminler 25200 uzunluktaki boxs dizisine bölünür
                item=[]
                for box in boxs:
                    box=box.cpu().numpy()
                    if box[4]>0.6:#nesnenin 4 indeksteki değeri güven aralığını ifade eder %70 doğruluk payı threshold ekliyoruz
                        x, y, w, h = box[0:4]
                        x1 = int(x- w/2) 
                        y1 = int(y- h/2)
                        x2 = int(x + w/2) 
                        y2 = int(y + h/2)
                        c = box[5:].argmax()
                        s = box[4]
                        item.append([[x1, y1, x2, y2], c, s])
                if len(item[0]) > 0:
                    item=np.array(item,dtype=object)
                    boxes = torch.tensor(item[:,0].tolist(),  dtype=torch.float32)
                    scores = torch.tensor(item[:,1].tolist(), dtype=torch.float32)
                    classes = torch.tensor(item[:, 1].tolist(),dtype=torch.int64)

                    indices = nms(boxes, scores, iou_threshold=0.6)#%60 benzerliğe sahip nesnelerden güven skoru düşük olanları nms fonksiyonuyla eledik bunun içinde numpy verilerimizi torch türüne çevirdik

                    filtered_boxes = item[:,0][indices]         #elde edilen indisler bütün indislerden elenmiş olanların çıkarılmış halidir
                    filtered_scores = item[:,2][indices]        #bu indislerin değerlerinide filtrelenmiş olarak tekrardan ele alıyoruz
                    filtered_classes = item[:, 1][indices]
                    all_predictions=np.column_stack((filtered_boxes, filtered_scores, filtered_classes))

                    scale_x = original_width / 640
                    scale_y = original_height / 640
                    for pred in all_predictions:
                        pred[0][0] *= scale_x
                        pred[0][1] *= scale_y
                        pred[0][2] *= scale_x
                        pred[0][3] *= scale_y
                    return all_predictions #bütün tahminleri birleştirip döndürüyoruz
#frame den tahminleri alıyoruz

def detect(cam_path):
    video =cv2.VideoCapture(cam_path)
    while True:
        back,frame=video.read()
        if not back:
            break
        predictions=predicted_objects(frame) #nesne sayısı uzunluğunda liste halinde 3 elemanlı listeden oluşan(1.konum 2.score 3.sınıf) bir dizi döndürür
        for pred in predictions:
            x1,y1,x2,y2=pred[0]
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 100, 0), 2)
            text=(str(classes[int(pred[2])])+", "+(str((float(int(pred[1]*10)/10)))))
            cv2.putText(frame, text, (int(x1)-10,int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2)
            cv2.imshow('Car1',frame)
        if cv2.waitKey(1)==ord('q'):
            break
    video.release()
    cv2.destroyAllWindows()
#tahminleri çerçeve üzerine yazdırıyoruz

cam_path=r'C:\Users\Mehmet\Desktop\traffic\cam1.mp4'
detect(cam_path)
#kodu çalıştırıyoruz

 