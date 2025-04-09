import colorsys
import os

import cv2
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from depth_anything_v2.dpt import DepthAnythingV2

from deep_sort.deep_sort import DeepSort
from deep_sort.utils.parser import get_config

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class ObjectDetector:
    """目标检测与跟踪类"""

    def __init__(self,
                 model_path='model/car_class.pt',
                 deepsort_config='deep_sort/configs/deep_sort.yaml',
                 conf_threshold=0.7,
                 device=DEVICE,
                 model=None):
        # 初始化YOLO检测模型
        if model is None:
            self.model = YOLO(model_path).to(device)
        else:
            self.model = model

        self.conf_threshold = conf_threshold
        self.device = device

        # 初始化DeepSort跟踪器
        cfg = get_config()
        cfg.merge_from_file(deepsort_config)
        self.deepsort = DeepSort(
            cfg.DEEPSORT.REID_CKPT,
            max_dist=cfg.DEEPSORT.MAX_DIST,
            min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
            nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
            max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
            max_age=cfg.DEEPSORT.MAX_AGE,
            n_init=cfg.DEEPSORT.N_INIT,
            nn_budget=cfg.DEEPSORT.NN_BUDGET,
            use_cuda=torch.cuda.is_available()
        )

    def _xyxy_to_xywh(self, bbox_xyxy):
        """将检测框从 [x1,y1,x2,y2] 转换为 [x_center,y_center,w,h]"""
        x1, y1, x2, y2 = bbox_xyxy
        w = x2 - x1
        h = y2 - y1
        x_center = x1 + w / 2
        y_center = y1 + h / 2
        return [x_center, y_center, w, h]

    def preprocess_frame(self, frame, target_size=(640, 480)):
        """预处理图像：调整大小到指定尺寸"""
        return cv2.resize(frame, target_size)

    def detect_objects(self, frame):
        """执行检测+跟踪，返回带跟踪ID的结果"""
        # 获取原始图像尺寸
        original_h, original_w = frame.shape[:2]

        # 预处理图像（缩小尺寸）
        target_size = (640, 480)
        processed_frame = self.preprocess_frame(frame, target_size)

        # YOLO检测
        results = self.model(processed_frame, conf=self.conf_threshold, verbose=False)


        # 计算缩放比例
        target_w, target_h = target_size
        scale_x = original_w / target_w  # 宽度缩放比例
        scale_y = original_h / target_h  # 高度缩放比例

        # 准备DeepSort输入数据
        bbox_list = []
        confidences = []
        class_ids = []

        for result in results:
            boxes = result.boxes.cpu().numpy()
            for i in range(len(boxes)):
                confidence = boxes.conf[i]
                if confidence < self.conf_threshold:
                    continue

                # 获取原始检测框坐标（基于缩放后图像）
                class_id = int(boxes.cls[i])
                x1, y1, x2, y2 = boxes.xyxy[i]

                # 坐标缩放回原始图像尺寸
                x1_orig = x1 * scale_x
                y1_orig = y1 * scale_y
                x2_orig = x2 * scale_x
                y2_orig = y2 * scale_y

                # 转换为DeepSort需要的xywh格式（基于原始图像）
                converted_bbox = self._xyxy_to_xywh([x1_orig, y1_orig, x2_orig, y2_orig])

                # 收集数据
                bbox_list.append(converted_bbox)
                confidences.append(float(confidence))
                class_ids.append(class_id)

        # 转换为NumPy数组（关键修正）
        if len(bbox_list) == 0:
            return []

        bbox_np = np.array(bbox_list, dtype=np.float32)
        confidences_np = np.array(confidences, dtype=np.float32)
        class_ids_np = np.array(class_ids, dtype=np.int32)

        # DeepSort跟踪（输入必须为CPU上的NumPy数组）
        tracks = self.deepsort.update(
            bbox_np,  # 输入格式: [[x_center,y_center,w,h], ...]
            confidences_np,  # 输入格式: [conf1, conf2, ...]
            frame,  # 原始图像用于ReID特征提取
            class_ids_np  # 输入格式: [class_id1, class_id2, ...]
        )

        # 格式化为输出
        detections = []
        for track in tracks:
            x1, y1, x2, y2, track_id, class_id = track
            detections.append({
                "track_id": int(track_id),
                "class": self.model.names[int(class_id)],
                "confidence": float(confidences[i]),
                "bbox": [float(x1), float(y1), float(x2), float(y2)]
            })

        return detections


class DepthEstimator:
    """深度估计类，增加相对速度计算和标签背景颜色"""

    def __init__(self, compute_method='da', smoothing_window=10, device=DEVICE):
        # 模型配置
        self.model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        self.compute_method = compute_method

        # 加载模型
        model_path = 'model/depth_anything_v2_vits.pth'
        model_name = model_path.split('.')[0].split('_')[-1]
        self.model_da = DepthAnythingV2(**self.model_configs[model_name])
        self.model_da.load_state_dict(torch.load(model_path, map_location=device))
        self.model_da = self.model_da.to(device).eval()


        self.depth_scale_factor = 50  # 经验参数
        self.smoothing_window = smoothing_window  # 速度计算的帧数范围
        self.depth_history = {}  # 记录历史深度数据
        self.time_history = {}  # 记录时间戳数据

        self.tracked_objects = []  # 存储跟踪对象
        self.max_age = 30  # 最大存活帧数（约1秒，假设30fps）
        self.similarity_threshold = 5  # 相似度阈值（越小越相似）



    def preprocess_frame(self, frame, target_size=(320, 240)):
        """预处理图像：调整大小并进行必要的转换"""
        frame_resized = cv2.resize(frame, target_size)
        return frame_resized


    def compute_depth_map_with_da(self, frame):
        """计算整帧深度图 (DepthAnything)"""
        small_frame = self.preprocess_frame(frame)
        depth_map = self.model_da.infer_image(small_frame)
        depth_map = cv2.resize(depth_map, (frame.shape[1], frame.shape[0]))
        depth_map = cv2.GaussianBlur(depth_map, (5, 5), 0)
        return depth_map


    def _find_best_match(self, det_obj):
        """在跟踪列表中找到最佳匹配"""
        for obj in self.tracked_objects:
            if obj['temp_id'] == det_obj['track_id']:
                return obj
        return None

    def estimate_distance_speed (self, curr_time, frame, depth_map, det_obj):
        """基于检测框特征的深度和速度估计"""
        # 提取当前帧特征
        x1, y1, x2, y2 = map(int, det_obj['bbox'])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        w, h = x2 - x1, y2 - y1
        current_features = (cx, cy, w, h)

        # 对象匹配 ------------------------------------------------
        best_match = self._find_best_match(det_obj)

        # 更新或创建对象 ------------------------------------------
        if best_match:
            # 更新现有对象
            best_match['features'] = current_features
            best_match['features_history'].append(current_features)
            best_match['age'] = 0  # 重置存活计时
            current_obj = best_match
        else:
            # 创建新对象
            current_obj = {
                'features': current_features,
                'features_history': [],
                'frame': frame,
                'depth_history': [],
                'calc_depth_history': [],
                'time_history': [],
                'temp_id': det_obj['track_id'],
                'age': 0
            }
            self.tracked_objects.append(current_obj)

        # 清理过期对象 --------------------------------------------
        self.tracked_objects = [
            obj for obj in self.tracked_objects
            if obj['age'] <= self.max_age
        ]

        # 统一增加对象年龄
        for obj in self.tracked_objects:
            obj['age'] += 1

        # # 深度计算 -----------------------------------------------
        # 提取有效深度数据
        x1, y1, x2, y2 = map(int, det_obj['bbox'])
        object_depth = depth_map[y1:y2, x1:x2]
        valid_depths = object_depth[object_depth > 0]

        if valid_depths.size == 0:
            return -1, 0, current_obj  # 无有效深度数据

        # 使用IQR过滤异常值
        q1, q3 = np.percentile(valid_depths, [25, 75])
        iqr = q3 - q1
        if iqr < 1e-6:  # 防止零IQR
            filtered_depths = valid_depths
        else:
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            filtered_depths = valid_depths[(valid_depths >= lower_bound) & (valid_depths <= upper_bound)]

        # 计算加权深度（高斯加权）
        depth_median = np.median(filtered_depths)
        weights = np.exp(-((filtered_depths - depth_median) ** 2) / (2 * (iqr ** 2 + 1e-6)))
        stable_depth = np.average(filtered_depths, weights=weights) * 7

        # 转换为实际距离
        distance_calced = self.depth_scale_factor / (stable_depth + 1e-6)  # 防止除零
        current_obj['calc_depth_history'].append(distance_calced)

        if 'depth_history' in current_obj and current_obj['depth_history']:
            alpha = 0.3

            # 计算 `calc_depth_history` 的趋势均值（取最近 3~5 次数据）
            trend_window = min(5, len(current_obj['calc_depth_history']))
            trend_avg = np.mean(current_obj['calc_depth_history'][-trend_window:])

            # 计算指数平滑值，融合历史趋势
            smoothed_distance = alpha * distance_calced + (1 - alpha) * np.mean(current_obj['depth_history'])

            # 限制最大加速度，防止突变
            max_acceleration = 5.0
            last_distance = current_obj['depth_history'][-1] if len(
                current_obj['depth_history']) > 1 else distance_calced
            time_diff = current_obj['time_history'][-1] - current_obj['time_history'][-2] if len(
                current_obj['time_history']) > 1 else 1

            max_delta = max_acceleration * time_diff ** 2
            if abs(smoothed_distance - last_distance) > max_delta:
                smoothed_distance = last_distance + np.sign(smoothed_distance - last_distance) * max_delta

            # 如果 `distance_calced` 和 `trend_avg` 相差较大，向趋势方向调整
            trend_weight = 0.4  # 趋势影响权重
            smoothed_distance = (1 - trend_weight) * smoothed_distance + trend_weight * trend_avg

            distance = smoothed_distance
        else:
            distance = distance_calced

        # 速度计算 -----------------------------------------------
        # 记录历史数据
        current_obj['depth_history'].append(distance)
        current_obj['time_history'].append(curr_time)

        # 保持窗口大小
        if len(current_obj['depth_history']) > self.smoothing_window:
            current_obj['depth_history'].pop(0)
            current_obj['time_history'].pop(0)
            current_obj['calc_depth_history'].pop(0)
            current_obj['features_history'].pop(0)

        # 计算速度（仅在有足够数据时）
        velocity = 0.0
        if len(current_obj['depth_history']) >= 2:
            time_diff = current_obj['time_history'][-1] - current_obj['time_history'][0]
            depth_diff = current_obj['depth_history'][-1] - current_obj['depth_history'][0]
            velocity = depth_diff / (time_diff + 1e-6)  # 防止除零

            # 应用平滑滤波
            velocity = 0.7 * velocity + 0.3 * np.mean([
                (current_obj['depth_history'][i] - current_obj['depth_history'][i - 1]) /
                (current_obj['time_history'][i] - current_obj['time_history'][i - 1]) + 1e-6
                for i in range(1, len(current_obj['depth_history']))
            ])

        return max(distance, 0), velocity, current_obj  # 确保非负距离



class VideoProcessor:
    """视频处理类"""
    def __init__(self, video_path, font_path="font/微软雅黑.ttf"):
        self.cap = cv2.VideoCapture(video_path)
        self.font = ImageFont.truetype(font_path, 12)

    def get_frame(self):
        """读取视频帧"""
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame


    def visualize_results(self, frame, detections, depth_map):
        """可视化目标检测和深度图"""
        img_pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(img_pil)

        def id_to_color(temp_id):
            """根据 temp_id 生成稳定且区分度高的颜色"""
            hue = (temp_id * 2654435761 % 360) / 360  # 2654435761 是黄金比例乘积
            rgb = colorsys.hsv_to_rgb(hue, 1.0, 0.9)  # H, S=1.0, V=0.9 颜色更亮
            rgb_int = tuple(int(c * 255) for c in rgb)  # 转换为 0-255 范围
            color = "#{:02x}{:02x}{:02x}".format(*rgb_int)  # 转换为 HEX
            return color

        for obj in detections:
            x1, y1, x2, y2 = map(int, obj["bbox"])
            label_text = f"{obj['class']} Dist: {obj['distance']:.2f}m Speed: {obj['velocity']:.2f}m/s"
            # 计算唯一颜色
            fill_color = id_to_color(obj['obj']['temp_id'])
            draw.rectangle([(x1, y1), (x2, y2)], outline=fill_color, width=2)
            draw.text((x1, y1 - 20), label_text, font=self.font, fill=(255, 255, 255, 255))

        # 归一化深度图
        depth_map_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        depth_colormap = cv2.applyColorMap(depth_map_norm, cv2.COLORMAP_JET)

        # 合并原始图像与深度图
        combined_output = np.hstack((np.array(img_pil), depth_colormap))
        return combined_output

    def release(self):
        """释放视频资源"""
        self.cap.release()

class MainApp:
    """主程序"""
    def __init__(self, video_path):
        self.detector = ObjectDetector()
        self.depth_estimator = DepthEstimator()
        self.video = VideoProcessor(video_path)

        # 获取视频信息
        self.frame_width = int(self.video.cap.get(cv2.CAP_PROP_FRAME_WIDTH)*2)
        self.frame_height = int(self.video.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = max(1, int(self.video.cap.get(cv2.CAP_PROP_FPS)))  # 避免 FPS 为 0

        # 确保输出文件夹存在
        os.makedirs("outputs", exist_ok=True)

        # 定义视频写入器
        self.out_writer = cv2.VideoWriter(
            f'outputs/{os.path.basename(video_path).split(".")[0]}.avi',
            cv2.VideoWriter_fourcc(*'MJPG'),  # 兼容性更好的编码格式
            self.fps,
            (self.frame_width, self.frame_height)
        )

    def run(self):
        while True:
            frame = self.video.get_frame()
            if frame is None:
                break  # 处理完所有帧后退出

            # 检测对象 & 深度估计
            detections = self.detector.detect_objects(frame)
            depth_map = self.depth_estimator.compute_depth_map_with_da(frame)

            # 获取当前帧时间（毫秒），转换为 Unix 时间戳
            curr_time = self.video.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # 视频内时间（秒）

            # 计算目标的距离和速度
            for obj in detections:
                obj["distance"], obj["velocity"], obj["obj"] = self.depth_estimator.estimate_distance_speed(curr_time, frame, depth_map, obj)

            # 渲染可视化结果
            output_frame = self.video.visualize_results(frame, detections, depth_map)

            # 确保写入的帧大小匹配
            output_frame = cv2.resize(output_frame, (self.frame_width, self.frame_height))

            # 确保是 BGR 三通道格式
            if len(output_frame.shape) == 2 or output_frame.shape[2] == 1:
                output_frame = cv2.cvtColor(output_frame, cv2.COLOR_GRAY2BGR)

            # 显示视频
            cv2.imshow("Detection & Depth Map", output_frame)

            # 写入输出视频
            self.out_writer.write(output_frame)

            # 按 'q' 退出
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # 释放资源
        self.video.release()
        self.out_writer.release()  # 释放视频写入器
        cv2.destroyAllWindows()



if __name__ == "__main__":
    app = MainApp("videos/4K交通监控测试视频.mp4")
    app.run()
