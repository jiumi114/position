#! /usr/bin/env python3

# Extract images from a bag file.

import os
import rosbag
import cv2
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge, CvBridgeError

TOPIC = '/camera/image/compressed'  # topic
OUTPUT_DIR = '/home/orangepi/demo2/src/demoros2/test'  # 保存目录
FRAME_INTERVAL = 1  # 提取图像的帧间隔

class ImageExtractor:
    def __init__(self):
        self.bridge = CvBridge()
        self.frame_count = 0  # 帧计数器

        # 检查并创建输出目录
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)

        # 处理 bag 文件
        with rosbag.Bag('/home/orangepi/rosbag2/data0604xiwuzhengshi_2025-05-25-06-19-02_49.bag', 'r') as bag:
            for topic, msg, t in bag.read_messages(topics=[TOPIC]):
                if self.frame_count % FRAME_INTERVAL == 0:
                    try:
                        cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='passthrough')
                        filename = f"{t.to_nsec()}.jpg"  # 使用时间戳作为文件名
                        save_path = os.path.join(OUTPUT_DIR, filename)
                        cv2.imwrite(save_path, cv_image)
                        print(f"Saved image: {save_path}")
                    except CvBridgeError as e:
                        print(f"Error converting image: {e}")
                        continue
                self.frame_count += 1

if __name__ == '__main__':
    try:
        ImageExtractor()
    except Exception as e:
        print(f"Error: {e}")