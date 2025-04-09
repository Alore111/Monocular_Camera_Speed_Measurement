# 单目摄像头跟踪测速

____
本项目基于`YOLO + DeepSort + DepthAnything`算法，实现单目摄像头下的车辆跟踪测速。

## 1. 环境配置
建议使用conda创建虚拟环境，安装依赖包
```
pip install -r requirements.txt
```
## 2. 数据
数据链接：https://pan.baidu.com/s/14GX60xn2zOmYjakkVrQMdQ?pwd=z5r6 
提取码：z5r6 

一共有四个文件，分别移动至下面存放位置：
```
项目根目录
├─deep_sort
│  └─deep_sort
│     └─deep
│        └─checkpoint
│           └─ckpt.t7
├─model
│  ├─car_class.pt
│  └─depth_anything_v2_vits.pt
└─videos
   └─4K交通监控测试视频.mp4
```
## 3. 运行
在项目根目录下运行
```
python Reasoning.py
```
运行时按`q`退出程序。
