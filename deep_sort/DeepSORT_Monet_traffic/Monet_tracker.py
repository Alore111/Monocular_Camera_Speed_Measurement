import time, random
import numpy as np
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import draw_outputs, convert_boxes

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from PIL import Image
from hyperlpr import *
from PIL import Image, ImageDraw, ImageFont

flags.DEFINE_string('classes', './data/labels/coco.names', 'path to classes file')
flags.DEFINE_string('weights', './weights/yolov3.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('video', './data/video/test.mp4',
                    'path to video file or number for webcam)')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')


def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontStyle = ImageFont.truetype(
        "simsun.ttc", textSize, encoding="utf-8")
    draw.text((left, top), text, textColor, font=fontStyle)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


# 判断是否在斑马线上
def on_banmaxian(k,b,c,x1,x2,y1,y2,x,y):

    if x>y1+c and x < y2 and y > x1 and y < x2:
        return True
    else: return False


def main(_argv):
    # 实线定义
    with open("shixian.txt", 'r') as f:
        k = [line.rstrip('\n') for line in f]
    # 斑马线定义
    with open("banmaxian.txt", 'r') as f:
        banma = [line.rstrip('\n') for line in f]

    # Definition of the parameters
    max_cosine_distance = 0.5
    nn_budget = None
    nms_max_overlap = 1.0
    
    #initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    if FLAGS.tiny:
        yolo = YoloV3Tiny(classes=FLAGS.num_classes)
    else:
        yolo = YoloV3(classes=FLAGS.num_classes)

    yolo.load_weights(FLAGS.weights)
    logging.info('weights loaded')

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')

    try:
        vid = cv2.VideoCapture(int(FLAGS.video))
    except:
        vid = cv2.VideoCapture(FLAGS.video)

    out = None

    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))
        list_file = open('detection.txt', 'w')
        frame_index = -1 
    
    fps = 0.0
    count = 0 
    #车牌信息字典，全局变量，实现对车牌识别信息的择优迭代
    chepaixinxi = {}
    on_bmx1 = []
    on_bmx2 = []
    on_bmx0 = 1
    while True:
        _, img = vid.read()
        # print(img.shape)
        if img is None:
            logging.warning("Empty Frame")
            time.sleep(0.1)
            count+=1
            if count < 3:
                continue
            else: 
                break
        
        img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        img_in = tf.expand_dims(img_in, 0)
        img_in = transform_images(img_in, FLAGS.size)

        t1 = time.time()
        boxes, scores, classes, nums = yolo.predict(img_in,steps = 1)
        classes = classes[0]
        names = []
        for i in range(len(classes)):
            names.append(class_names[int(classes[i])])
        #print(names)
        names = np.array(names)
        #print(names)
        


        converted_boxes = convert_boxes(img, boxes[0])
        features = encoder(img, converted_boxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(converted_boxes, scores[0], names, features)]
        
      
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

  
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]        


        tracker.predict()
        tracker.update(detections)
        num = 0
        # 统计车和行人的数量
        car_num = 0
        person_num = 0
        
        # 记录行人位置
        xingren = []

        # 在帧上标注车道线
        for i in banma:
            i = eval(i)
            cv2.rectangle(img, (i['x1'],i['y1']+i['c']), (i['x2'],i['y2']), (0,255,0), 4)#画斑马线
        for i in k:
            i = eval(i)
            if i['k'] != 0:
                for xxx in range(i['y1'],i['y2']):
                    xq = (xxx-i['b'])/i['k']
                    xq = int(xq)
                    cv2.rectangle(img, (xq+5,xxx), (xq-5,xxx), (0,0,255), -1)
            else:
                for xxx in range(i['x1'],i['x2']):
                    yq = i['b']
                    cv2.rectangle(img, (xxx,yq+5), (xxx,yq-5), (0,0,255), -1)


        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            
            bbox = track.to_tlbr()
            class_name = track.get_class()

            # 统计车辆和行人数量
            if class_name == 'car' or class_name == 'bus' or class_name == 'truck':
                car_num += 1
            elif class_name == 'person':
                person_num += 1 

            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)

            cv2.rectangle(img, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
            cv2.putText(img, class_name + "-" + str(track.track_id),(int(bbox[0]) , int(bbox[1]-10)),0, 0.75, (255,255,255),2)
      
            if class_name == 'person':
                # 计算出行人的位置（用行人脚的位置定义行人所在位置）
                per = int((int(bbox[2])+int(bbox[0]))/2)

                xingren.append(((int(bbox[3]),per)))
   
            if on_bmx0 == 1:
                for i in xingren:
                    for ix in banma:
                        ix = eval(ix)
                        if on_banmaxian(ix['k'], ix['b'], ix['c'], ix['x1'], ix['x2'],ix['y1'],ix['y2'],i[0],i[1]):
                            on_bmx1.append(i[1])
                on_bmx1.sort()
            else:
                for i in xingren:
                    for ix in banma:
                        ix = eval(ix)
                        if on_banmaxian(ix['k'], ix['b'], ix['c'], ix['x1'], ix['x2'],ix['y1'],ix['y2'],i[0],i[1]):
                            on_bmx2.append(i[1])
                on_bmx2.sort()
            if class_name == 'car' and (int(bbox[3])-int(bbox[1]))>100 and (int(bbox[2])-int(bbox[0])>200):
                # 截取车辆图片
                cropped = img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                xinxi = []
                # 传入HyperLPR模型识别车牌信息
                if str(track.track_id) in chepaixinxi:
                    chepai = chepaixinxi[str(track.track_id)]
                    if chepai[1] < 90:
                        xinxi = HyperLPR_plate_recognition(cropped)
                else:

                    try:
                        if cropped.shape[0] != 0 and cropped.shape[1] != 0:
                            xinxi = HyperLPR_plate_recognition(cropped)
                        else:
                            xinxi = []
                    except subprocess.CalledProcessError as e:
                        xinxi = []

                # 计算车的中心点坐标
                diet1 = (int(bbox[3])-int(bbox[1]))/2
                diet2 = (int(bbox[2])-int(bbox[0]))/2
                x = int(bbox[1]) + diet1
                y = int(bbox[0]) + diet2
                # 如果车的中心点落在实线的”范围“内，就判断车辆非法越线。
                for i in k:
                    i = eval(i)
                    # 修改
                    if x > i['y1'] and x < i['y2']:
                        if x > y*i['k']+i['b']-15 and x < y*i['k']+i['b']+15:
                            img = cv2ImgAddText(img,'非法越线',int(bbox[0]) , int(bbox[1]+20),(255,0,0),20)
                            print('car违规！违规类型：越实线！',xinxi)
                
                if on_bmx0 == 1:
                    on_bmx = on_bmx1
                else:
                    on_bmx = on_bmx2
                
                # 实现车牌信息的择优迭代
                if str(track.track_id) in chepaixinxi:
                    if xinxi:
                        chepai = chepaixinxi[str(track.track_id)]
                        if chepai[1] < xinxi[0][1]:
                            cheche = xinxi[0]
                            chepaixinxi[str(track.track_id)] = cheche
                            img = cv2ImgAddText(img,chepaixinxi[str(track.track_id)][0]+':'+str(round(chepaixinxi[str(track.track_id)][1], 2)),int(bbox[0]) , int(bbox[1]),(0,0,255),20)
                        else:
                            img = cv2ImgAddText(img,chepaixinxi[str(track.track_id)][0]+':'+str(round(chepaixinxi[str(track.track_id)][1], 2)),int(bbox[0]) , int(bbox[1]),(0,0,255),20)
                elif xinxi :
                    cheche = xinxi[0]
                    chepaixinxi[str(track.track_id)] = cheche
                    img = cv2ImgAddText(img,chepaixinxi[str(track.track_id)][0]+':'+str(round(chepaixinxi[str(track.track_id)][1], 2)),int(bbox[0]) , int(bbox[1]),(0,0,255),20)

                # 车辆斑马线不礼让行人检测，并记录它的车牌信息和违规情况
                for ix in banma:
                    ix = eval(ix)
                    if on_banmaxian(ix['k'],ix['b'],ix['c'],ix['x1'],ix['x2'],ix['y1'],ix['y2'],x,y):
                        if len(on_bmx) != 0:
                
                            if str(track.track_id) in chepaixinxi:
                                img = cv2ImgAddText(img,'不礼让行人',int(bbox[0]) , int(bbox[1]+40),(255,0,0),20)
                                print('car违规:没有礼让行人!',chepaixinxi[str(track.track_id)][0])
                            else:
                                img = cv2ImgAddText(img,'不礼让行人',int(bbox[0]) , int(bbox[1]+40),(255,0,0),20)
                                print('car违规:没有礼让行人!')
        
        print('car name:',car_num,'person num:',person_num)

        if on_bmx0 == 1:
            on_bmx0 = 2
        else:
            on_bmx0 = 1
        print(chepaixinxi.items())

        fps  = ( fps + (1./(time.time()-t1)) ) / 2
        cv2.putText(img, "FPS: {:.2f}".format(fps)+'  car num:'+str(car_num)+'    '+'person num:'+str(person_num), (0, 30),
                          cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
        cv2.imshow('output', img)
        if FLAGS.output:
            out.write(img)
            frame_index = frame_index + 1
            list_file.write(str(frame_index)+' ')
            if len(converted_boxes) != 0:
                for i in range(0,len(converted_boxes)):
                    list_file.write(str(converted_boxes[i][0]) + ' '+str(converted_boxes[i][1]) + ' '+str(converted_boxes[i][2]) + ' '+str(converted_boxes[i][3]) + ' ')
            list_file.write('\n')

        # press q to quit
        if cv2.waitKey(1) == ord('q'):
            break
    vid.release()
    if FLAGS.ouput:
        out.release()
        list_file.close()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
    # app.run(main)


#运行
#python object_tracker.py --video ./data/video/test1.mp4 --output ./data/video/results.avi
#python Monet_tracker.py --video ./data/video/test.mp4 --output ./data/video/results.avi

