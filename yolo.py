import cv2
import time
import numpy as np
import playsound
import os
from gtts import gTTS

max_frames = 0
def LoadYolo():
    ''' This function is used to load yolo model '''
    classes = []
    with open('coco.names','r') as f:
        classes = f.read().split('\n')
        
    net = cv2.dnn.readNetFromDarknet('yolov3.cfg','yolov3.weights')
    output_layers = net.getUnconnectedOutLayersNames()
    colors = np.random.randint(0,255,(len(classes),3))

    return net,classes,colors,output_layers

def LoadImg(img_path):
    ''' This function is used to load image for detection ''' 
    img = cv2.imread(img_path)
    img = cv2.resize(img,None,fx=0.4, fy=0.4)
    height, width, channels = img.shape

    return img,height,width,channels

def ImagePreprocess(img,net,output_layers):
    ''' This function is used for performing preprocessing of image '''
    blob = cv2.dnn.blobFromImage(img, scalefactor=1/255, size=(192,192), mean=(0,0,0), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    return blob,outputs

def GetDimensions(height,width,outputs):
    ''' This function is used to find location,confidence scores and class Ids of detected objects '''
    boxes = []
    confs = []
    class_ids = []

    for x in outputs:
        for y in x:
            score = y[5:] # score is a list which stores confidence_score of all the objects
            id = np.argmax(score) # id stores index of the class which has the highest confidence_score 
            confidence_score = score[id] 

            if confidence_score > 0.4:
                X_center = int(y[0] * width)
                Y_center = int(y[1] * height)

                ''' Co-ordinates of the located object in the image'''
                W = int(y[2] * width)
                H = int(y[3] * height)
                X = int(X_center - W/2)
                Y = int(Y_center - H/2)

                boxes.append([X,Y,W,H])
                confs.append(float(confidence_score))
                class_ids.append(id)

    return boxes, confs, class_ids    

def DrawBBox(img, classes, colors, boxes, confs, class_ids):
    global max_frames
    ''' This function is used to draw a bounding box around the detected object '''
    # Applying Non-Max suppression to remove weak overlapping bounding boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_DUPLEX
    texts = []

    for i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        col = (int(colors[i][0]), int(colors[i][1]), int(colors[i][2]))

        cv2.rectangle(img, (x,y), (x+w , y+h), col, 2)
        cv2.putText(img, label.upper(), (x, y-5), font, 1, col, 2)

        texts.append(label)

    max_frames+=1
    if max_frames % 10 == 0:
        if texts:
            print(texts)
            ''' Uncomment below lines to enable text to speech'''
            # desc = ', '.join(texts)
            # filename = "voice.mp3"
            # tts = gTTS(desc, lang="en")
            # tts.save(filename)
            # playsound.playsound(filename)
            # os.remove(filename)

    cv2.imshow("Frame",img)

def start_webcam():
    model, classes, colors, output_layers = LoadYolo()
    cap = cv2.VideoCapture(0)
    frame_cnt = 0
    start = time.time()
    while True:
        frame_cnt+=1
        if(frame_cnt % 10 == 0):
            end = time.time()
            print('{:.2f} FPS'.format(10/(end-start)))
            start = time.time()
        ret,frame = cap.read()
        if ret == True:
            height, width, channels = frame.shape   
            blob, outputs = ImagePreprocess(frame, model, output_layers)
            boxes, confs, class_ids = GetDimensions(height, width, outputs)
            DrawBBox(frame, classes, colors, boxes, confs, class_ids)
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # press q on keyboard to interrupt webcam
    start_webcam()
