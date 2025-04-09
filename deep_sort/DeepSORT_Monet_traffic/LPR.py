"""
@Datetime： 2021/6/17 20:00 
@Author： Sjh
@IDE： PyCharm
"""
#导入包
from hyperlpr import *
import numpy
from PIL import Image, ImageDraw, ImageFont
#导入OpenCV库
import cv2


#第一种方式:加入中文 https://blog.csdn.net/qq_41895190/article/details/90301459
def image_add_text(img1, text, left, top, text_color, text_size):
    # 判断图片是否为ndarray格式，转为RGB图片
    if isinstance(img1, numpy.ndarray):
        image = Image.fromarray(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(image)
    # 参数依次为 字体、字体大小、编码
    font_style = ImageFont.truetype("font/simsun.ttc", text_size, encoding='utf-8')
    # 参数依次为位置、文本、颜色、字体
    draw.text((left, top), text, text_color, font=font_style)
    # 图片转换为opencv格式
    return cv2.cvtColor(numpy.asarray(image), cv2.COLOR_RGB2BGR)

#定义显示窗口的大小
cv2.namedWindow("enhanced",0)
cv2.resizeWindow("enhanced", 640, 480)

#读入图片
image = cv2.imread(r"D:\Project\Monet_traffic\data\photo\1.png")

# #改变图片大小
# image = cv2.resize(image,None,fx=0.5,fy=0.5)

#识别结果
print(HyperLPR_plate_recognition(image))

#识别信息
xinxi = HyperLPR_plate_recognition(image)[0][0]
conf = HyperLPR_plate_recognition(image)[0][1]
location = HyperLPR_plate_recognition(image)[0][2]
print(xinxi)
print(location)

# opencv添加文字
# font = cv2.FONT_HERSHEY_SIMPLEX
# # print(xinxi.split('J')[0])
# if xinxi.split('J')[0] == "川":
#     xinxi = "川" + xinxi.split('J')[0]
# # cv2.putText(image, xinxi, (600, 500), font, 1, (0, 0, 255), 3, cv2.LINE_AA)

#转换为PIL的image格式,使用PIL绘制文字,再转换为OpenCV的图片格式
image = image_add_text(image, xinxi,location[0], location[1], (255, 0, 0), 50)

#画框
cv2.rectangle(image, (location[0], location[1]), (location[2], location[3]), (0, 255, 0), 2)

#展示
cv2.imshow("image",image)
cv2.waitKey(0)






#方法二:https://blog.csdn.net/qq_44740544/article/details/106177945
# #读入单张图片：加中文
# image = cv2.imread(r"D:\Project\Monet_traffic\data\photo\1.png")
#
# # 判断图片是否为ndarray格式，转为RGB图片
# if (isinstance(image, numpy.ndarray)):
#     img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#
# draw = ImageDraw.Draw(img)
# # 参数依次为 字体、字体大小、编码
#
# fontStyle = ImageFont.truetype("font/simsun.ttc", 20, encoding="utf-8")
# # 参数依次为位置、文本、颜色、字体
# draw.text((10, (1* 50) + 23), '这是', (255, 255, 255), font=fontStyle)
#
# # 转回BGR图片、ndarray格式
# image = cv2.cvtColor(numpy.asarray(img), cv2.COLOR_RGB2BGR)
#
# cv2.imshow("image",image)
#
# cv2.waitKey(0)



#车牌识别:https://huchenyang.net/py/80.html
#添加中文的思路
"""
因为使用cv2.putText() 只能显示英文字符，中文会出现乱码问题，
因此使用PIL在图片上绘制添加中文，可以指定字体文件。
大体思路：
OpenCV图片格式转换成PIL的图片格式；
"""
