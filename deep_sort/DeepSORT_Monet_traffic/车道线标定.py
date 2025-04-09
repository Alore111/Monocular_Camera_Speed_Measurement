import cv2
from PIL import Image
from pylab import *
import csv
import codecs
from matplotlib.pyplot import ginput,ion,ioff
import matplotlib.pyplot as plt

#1.读取背景图片
img = cv2.imread('background.png')
# cishu是需要标记的实线的个数。这个需要标记4处实线。
cishu = 2
sx = []
bm = []
im = array(Image.open('background.png'))

#2.画实线框点和矩形点
#打开交互模式:https://blog.csdn.net/SZuoDao/article/details/52973621
ion()
imshow(im)

for cs in range(cishu):
    print('Please click 2 points')
    x = ginput(2)
    print('you clicked:',x)
    sx.append(x)

#这里我只处理了一处斑马线
print('Please click 4 points')
x = ginput(4)
print('you clicked:',x)
bm.append(x)
#显示前关掉交互模式:在plt.ioff()后面紧跟着写上plt.show()防止程序在绘图结束后闪退
ioff()
show()
print(im.shape)


#3.画实线框和矩形框
jinzhi = []
banmaxian = []
#实线
def shixian(x1,y1,x2,y2):
    if x1 == x2:
        k = -999
        b = 0
    else:
        k = (y2-y1)/(x2-x1)
        b = y1 - x1 * k
        # k = int(k)
        # b = int(b)
    return k,b

#3.1画实线框
#data1 = [{'x1':int(x[0][0]),'y1':int(x[0][1]),'x2':int(x[1][0]),'y2':int(x[1][1])}]
# 计算实线的值 y = kx + b
for i in sx:
    x = {}
    x1 = int(i[0][0])
    y1 = int(i[0][1])
    x2 = int(i[1][0])
    y2 = int(i[1][1])
    k, b = shixian(x1,y1,x2,y2)
    #cv2.rectangle(img, (x1+15,y1), (x2-15,y2), (0,0,255), -1)
    if y1 > y2:
        yy = y2
        xx = x2
        y2 = y1
        x2 = x1
        y1 = yy
        x1 = xx
    if k != 0:
        for xxx in range(y1,y2):
            xq = (xxx-b)/k
            xq = int(xq)
            cv2.rectangle(img, (xq+15,xxx), (xq-15,xxx), (0,0,255), -1)
    else:
        for xxx in range(x1,x2):
            yq = b
            cv2.rectangle(img, (xxx,yq+15), (xxx,yq-15), (0,0,255), -1)
    x['k'] = k
    x['b'] = b
    x['x1'] = x1
    x['x2'] = x2
    x['y1'] = y1
    x['y2'] = y2
    print('k:',k,'b:',b)
    jinzhi.append(x)
print(jinzhi)


#画斑马线区域
# data2 = [{'x1':400, 'y1':0,'x2':800,'y2':0,
#           'x3':400, 'y3':800, 'x4':800, 'y4':800}]
# 计算斑马线的各项值 y = kx + b
for i in bm:
    x = {}
    x1 = int(i[0][0])
    y1 = int(i[0][1])
    x2 = int(i[1][0])
    y2 = int(i[1][1])
    y3 = int(i[2][1])
    k, b = shixian(x1,y1,x2,y2)
    c = y3 - y1
    x['k'] = k
    x['b'] = b
    x['c'] = c
    x['x1'] = x1
    x['x2'] = x2
    x['y1'] = y1
    x['y2'] = y2
    x['y3'] = y3
    cv2.rectangle(img, (x1,y1+c), (x2,y2), (0,255,0), 4)
    banmaxian.append(x)

print(banmaxian)
cv2.imwrite('out.png', img)
cv2.imshow("label",img)

cv2.waitKey(0)


# 将获得的实线和斑马线信息写入相应的文件。
with open("shixian.txt", 'w') as f:
    for s in jinzhi:
        f.write(str(s) + '\n')
with open("banmaxian.txt", 'w') as f:
    for s in banmaxian:
        f.write(str(s) + '\n')

