import cv2

# 这里我们取视频的第一帧来进行标注。注意⚠️不要使用截图，因为截图会使的图像大小不一致。
vidcap = cv2.VideoCapture('data/video/test.mp4')
success,image = vidcap.read()
n=1
while n < 30:
	success, image = vidcap.read()
	n+=1
imag = cv2.imwrite('background.png',image)
if imag ==True:
	print('提取背景成功')

