#!/usr/bin/env python3

import torch
import cv2, os, time, yaml, struct
import numpy as np
from typing import List, Tuple
from sklearn.cluster import DBSCAN
import rospy
import logging
logging.basicConfig(level=logging.INFO)
rospy.init_node('data_collector', anonymous=True)
from cv_bridge import CvBridge
import message_filters
from sensor_msgs.msg import PointCloud2, CompressedImage
import sensor_msgs.point_cloud2 as pc2
from position.msg import Detection, Detections
from geometry_msgs.msg import Point
import geometry_msgs.msg
import threading
import queue
import tf2_ros
from tf2_msgs.msg import TFMessage
from tf.transformations import quaternion_matrix, translation_matrix

# 配置参数
RKNN_MODEL = "/home/orangepi/position/src/position/src/models/yolov8n-1.6.rknn"
CLASSES_PATH = "/home/orangepi/position/src/position/src/models/coco.yaml"

# topic
PC_TOPIC = "/ouster/points"
IMG_TOPIC = "/camera/image/compressed"
PUB_TOPIC = "other_vehicle_detections"
GLOBAL_MAP = 'odom'
CAR_MAP = 'ouster_sensor_link'
CAR_ID = "car-001"

MAX_X = 100
MAX_Y = 25.0
MIN_Z = -2.0
MIN_REFLECTIVITY = 0.01

NMS_THRESH = 0.25
OBJ_THRESH = 0.45
MAX_QUEUE_SIZE = 5  # 最大队列大小
NUM_WORKER_THREADS = 3  # 工作线程数量

class DetectedObject:
    def __init__(self):
        # 检测信息
        self.class_id = 0
        self.class_name = ""
        self.confidence = 0.0
        self.box_2d = None       # [x, y, w, h]
        
        # 3D信息
        self.position = None     # (x, y, z)
        self.bbox_3d = None      # (min_x, min_y, min_z, max_x, max_y, max_z)
        self.point_count = 0

class DataCollector:
    def __init__(self):
        self.bridge = CvBridge()
        self.data_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
        self.lock = threading.Lock()
        
        pc_sub = message_filters.Subscriber(PC_TOPIC, PointCloud2)
        img_sub = message_filters.Subscriber(IMG_TOPIC, CompressedImage)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.latest_tf = None

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [pc_sub, img_sub],
            queue_size=10,
            slop=0.05
        )
        self.ts.registerCallback(self.sync_callback)
        
        print("Data collector initialized with message_filters...")

    def sync_callback(self, pc_msg, img_msg):
        try:
            # 如果队列已满，丢弃最旧的数据
            if self.data_queue.full():
                print("队列已满，丢弃最旧的数据")
                self.data_queue.get_nowait()
            try:
                print("尝试获取TF")
                transform = self.tf_buffer.lookup_transform(GLOBAL_MAP, CAR_MAP, pc_msg.header.stamp, rospy.Duration(0.1))
                print("TF获取成功")
                self.latest_tf = transform
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                # rospy.logwarn(f"TF获取失败: {str(e)}")
                print(f"TF获取失败: {str(e)}")
                return
            self.data_queue.put_nowait((pc_msg, img_msg, self.latest_tf))
        except queue.Full:
            # print("queue.Full")
            pass
        except Exception as e:
            print(f"Error in sync_callback: {str(e)}")

    def get_data(self):
        try:
            return self.data_queue.get_nowait()
        except queue.Empty:
            # print("queue.Empty")
            return None, None, None

class ThreadSafeRKNN:
    def __init__(self):
        self.lock = threading.Lock()
        self.rknnlite = self._load_rknn_model()
        
    def _load_rknn_model(self):
        """加载并初始化RKNN模型"""
        from rknnlite.api import RKNNLite
        rknnlite = RKNNLite()
        
        print('--> Loading model')
        ret = rknnlite.load_rknn(RKNN_MODEL)
        if ret != 0:
            raise RuntimeError('Load model failed!')
        
        print('--> Init runtime environment')
        ret = rknnlite.init_runtime()
        if ret != 0:
            raise RuntimeError('Init runtime environment failed!')
        
        return rknnlite
    
    def release(self):
        """Release RKNN resources"""
        with self.lock:
            if hasattr(self, 'rknn') and self.rknnlite is not None:
                self.rknnlite.release()
                self.rknnlite = None
    
    def run_detection(self, img, class_names):
        with self.lock:
            return RKNNProcessor.run_detection(self.rknnlite, img, class_names)

class LidarProcessor:
    @staticmethod
    def load_calibration_data():
        """加载相机标定矩阵"""
        RT = np.array([
            [-4.670000e-03, -9.999900e-01, -8.800000e-04, -3.125000e-02],
            [-2.516000e-02,  1.000000e-03, -9.996800e-01, -4.237000e-02],
            [ 9.996700e-01, -4.650000e-03, -2.517000e-02, -1.401700e-01],
            [ 0.000000e+00,  0.000000e+00,  0.000000e+00,  1.000000e+00]
        ])

        # 现在的标定有点偏，我加了个水平偏移的量
        compensation_x = 0.10
        RT[0, 3] += compensation_x
        
        R_rect = np.array([
            [1.000000e+00, 0.000000e+00, 0.000000e+00, 0.0],
            [0.000000e+00, 1.000000e+00, 0.000000e+00, 0.0],
            [0.000000e+00, 0.000000e+00, 1.000000e+00, 0.0],
            [0, 0, 0, 1]
        ])
        
        P_rect = np.array([
            [2.296087e+03, 0.000000e+00, 1.201338e+03, 0.000000e+00],
            [0.000000e+00, 2.298004e+03, 1.046457e+03, 0.000000e+00],
            [0.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00]
        ])
        
        return P_rect, R_rect, RT

    @staticmethod
    def process_point_cloud(pc_msg):
        """处理ROS PointCloud"""
        points = []
        gen = pc2.read_points(pc_msg, field_names=("x", "y", "z", "intensity"), skip_nans=False)
        
        for p in gen:
            x, y, z, intensity = p[0], p[1], p[2], p[3]
            r = intensity/255
            
            if not (x > MAX_X or x < 0.0 or 
                   abs(y) > MAX_Y or z < MIN_Z or 
                   r < MIN_REFLECTIVITY):
                points.append((x, y, z, r))
        
        return points

    @staticmethod
    def cluster_points(points, eps=0.5, min_samples=10):
        """使用DBSCAN聚类点"""
        if not points:
            return []
        
        points_array = np.array([[p[0], p[1], p[2]] for p in points])
        dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(points_array)
        labels = dbscan.labels_
        
        clusters = []
        unique_labels = set(labels)
        
        for label in unique_labels:
            if label == -1:
                continue
            cluster_points = [points[i] for i in range(len(points)) if labels[i] == label]
            clusters.append(cluster_points)
        
        return clusters

    @staticmethod
    def compute_3d_bbox(points):
        """计算3D框"""
        if not points:
            return (0, 0, 0), (0, 0, 0, 0, 0, 0)
        
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        z_coords = [p[2] for p in points]
        
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        min_z, max_z = min(z_coords), max(z_coords)
        
        center = ((min_x + max_x)/2, (min_y + max_y)/2, (min_z + max_z)/2)
        bbox = (min_x, min_y, min_z, max_x, max_y, max_z)
        
        return center, bbox

class RKNNProcessor:
    @staticmethod
    def run_detection(rknnlite, img, class_names):
        """目标检测"""
        h_orig, w_orig = img.shape[:2]
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized, ratio, (dw, dh) = RKNNProcessor.letterbox(img_rgb)
        img_input = np.expand_dims(img_resized, 0)
        
        outputs = rknnlite.inference(inputs=[img_input], data_format=['nhwc'])
        boxes, classes, scores = RKNNProcessor.post_process(outputs)
        
        detections = []
        if boxes is not None:
            for box, cls, score in zip(boxes, classes, scores):
                x1, y1, x2, y2 = box
                x1 = (x1 - dw) / ratio[0]
                y1 = (y1 - dh) / ratio[1]
                x2 = (x2 - dw) / ratio[0]
                y2 = (y2 - dh) / ratio[1]
                
                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = min(w_orig, int(x2)), min(h_orig, int(y2))
                
                detection = DetectedObject()
                detection.class_id = int(cls)
                detection.confidence = float(score)
                detection.box_2d = [x1, y1, x2 - x1, y2 - y1]
                detections.append(detection)
        
        return detections, class_names

    @staticmethod
    def dfl(position):
        x = torch.tensor(position)
        n,c,h,w = x.shape
        p_num = 4
        mc = c//p_num
        y = x.reshape(n,p_num,mc,h,w)
        y = y.softmax(2)
        acc_metrix = torch.tensor(range(mc)).float().reshape(1,1,mc,1,1)
        y = (y*acc_metrix).sum(2)
        return y.numpy()

    @staticmethod
    def box_process(position):
        grid_h, grid_w = position.shape[2:4]
        col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
        col = col.reshape(1, 1, grid_h, grid_w)
        row = row.reshape(1, 1, grid_h, grid_w)
        grid = np.concatenate((col, row), axis=1)
        stride = np.array([640//grid_h, 640//grid_w]).reshape(1,2,1,1)

        position = RKNNProcessor.dfl(position)
        box_xy  = grid +0.5 -position[:,0:2,:,:]
        box_xy2 = grid +0.5 +position[:,2:4,:,:]
        xyxy = np.concatenate((box_xy*stride, box_xy2*stride), axis=1)

        return xyxy

    @staticmethod
    def post_process(outputs):
        boxes, scores, classes_conf = [], [], []
        defualt_branch=3
        pair_per_branch = len(outputs)//defualt_branch
        
        for i in range(defualt_branch):
            boxes.append(RKNNProcessor.box_process(outputs[pair_per_branch*i]))
            classes_conf.append(outputs[pair_per_branch*i+1])
            scores.append(np.ones_like(outputs[pair_per_branch*i+1][:,:1,:,:], dtype=np.float32))

        def sp_flatten(_in):
            ch = _in.shape[1]
            _in = _in.transpose(0,2,3,1)
            return _in.reshape(-1, ch)

        boxes = [sp_flatten(_v) for _v in boxes]
        classes_conf = [sp_flatten(_v) for _v in classes_conf]
        scores = [sp_flatten(_v) for _v in scores]

        boxes = np.concatenate(boxes)
        classes_conf = np.concatenate(classes_conf)
        scores = np.concatenate(scores)

        boxes, classes, scores = RKNNProcessor.filter_boxes(boxes, scores, classes_conf)

        if len(boxes) > 0:
            keep = RKNNProcessor.nms_boxes(boxes, scores)
            boxes = boxes[keep]
            classes = classes[keep]
            scores = scores[keep]

        return boxes, classes, scores

    @staticmethod
    def filter_boxes(boxes, box_confidences, box_class_probs, OBJ_THRESH=OBJ_THRESH):
        box_confidences = box_confidences.reshape(-1)
        candidate, class_num = box_class_probs.shape

        class_max_score = np.max(box_class_probs, axis=-1)
        classes = np.argmax(box_class_probs, axis=-1)

        _class_pos = np.where(class_max_score* box_confidences >= OBJ_THRESH)
        scores = (class_max_score* box_confidences)[_class_pos]

        boxes = boxes[_class_pos]
        classes = classes[_class_pos]

        return boxes, classes, scores

    @staticmethod
    def nms_boxes(boxes, scores, NMS_THRESH=NMS_THRESH):
        x = boxes[:, 0]
        y = boxes[:, 1]
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]

        areas = w * h
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x[i], x[order[1:]])
            yy1 = np.maximum(y[i], y[order[1:]])
            xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
            yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

            w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
            h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
            inter = w1 * h1

            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= NMS_THRESH)[0]
            order = order[inds + 1]
        keep = np.array(keep)
        return keep

    @staticmethod
    def letterbox(im, new_shape=(640, 640), color=(0, 0, 0)):
        shape = im.shape[:2]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        ratio = r, r
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

        dw /= 2
        dh /= 2

        if shape[::-1] != new_unpad:
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return im, ratio, (dw, dh)

class FusionProcessor:
    def __init__(self):
        self.bridge = CvBridge()
        self.class_names = self.load_class_names()
        self.rknnlite = ThreadSafeRKNN()
        self.P_rect, self.R_rect, self.RT = LidarProcessor.load_calibration_data()
        self.detection_pub = rospy.Publisher(PUB_TOPIC, Detections, queue_size=10)
        self.task_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
        self.worker_threads = []
        self.stop_event = threading.Event()
        
        # 创建工作线程
        for i in range(NUM_WORKER_THREADS):
            t = threading.Thread(target=self.worker_loop, daemon=True)
            t.start()
            self.worker_threads.append(t)
        
    def load_class_names(self):
        with open(CLASSES_PATH, 'r') as f:
            data = yaml.safe_load(f)
            return data.get('names', [])
    
    def worker_loop(self):
        while not self.stop_event.is_set():
            try:
                # 获取任务数据
                task = self.task_queue.get(timeout=0.1)
                if task is None:
                    continue
                
                img, detections, pc_msg, transform = task
                
                # 处理3D信息
                results = self.get_object_3d_coordinates(LidarProcessor.process_point_cloud(pc_msg), detections, transform)
                
                # 发布
                self.publish_detections(results, pc_msg.header)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in worker thread: {str(e)}")
    
    def process_data(self, pc_msg, img_msg, transform):
        """主处理流程"""
        try:
            img = self.bridge.compressed_imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
            
            # 2. 运行目标检测
            detections, _ = self.rknnlite.run_detection(img, self.class_names)
            
            # 3. 将任务放入队列
            self.task_queue.put((img, detections, pc_msg, transform))
            
            return True
        
        except Exception as e:
            print(f"Processing error: {str(e)}")
            return False
    
    def publish_detections(self, results, pc_header):
        """发布"""
        detections_msg = Detections()
        detections_msg.header = pc_header
        detections_msg.car_id = CAR_ID
        
        for i, obj in enumerate(results):
            det_msg = Detection()
            det_msg.object_id = f"obj_{i}_{pc_header.seq}"
            det_msg.type = obj.class_name
            det_msg.confidence = obj.confidence
            det_msg.box_2d = obj.box_2d
            
            pos = Point()
            pos.x, pos.y, pos.z = obj.position
            det_msg.position = pos
            
            det_msg.bbox_3d = list(obj.bbox_3d)
            detections_msg.detections.append(det_msg)

        self.print_detection_message(detections_msg)
        self.detection_pub.publish(detections_msg)

    def print_detection_message(self, msg):
        print("\n" + "="*50)
        print("发布检测结果：")
        print(f"  车ID: {msg.car_id}")
        print(f"  时间戳: {msg.header.stamp.to_sec()}")
        print(f"  检测到 {len(msg.detections)} 个对象:")
        
        for i, det in enumerate(msg.detections, 1):
            print(f"  [{i}] {det.type} (ID: {det.object_id})")
            print(f"    置信度: {det.confidence:.2f}")
            print(f"    2D框: [x:{det.box_2d[0]} y:{det.box_2d[1]} w:{det.box_2d[2]} h:{det.box_2d[3]}]")
            print(f"    3D位置: x={det.position.x:.2f}m, y={det.position.y:.2f}m, z={det.position.z:.2f}m")
            print(f"    3D边界框:")
            print(f"      min: ({det.bbox_3d[0]:.2f}, {det.bbox_3d[1]:.2f}, {det.bbox_3d[2]:.2f})")
            print(f"      max: ({det.bbox_3d[3]:.2f}, {det.bbox_3d[4]:.2f}, {det.bbox_3d[5]:.2f})")
        print("="*50 + "\n")

    def get_object_3d_coordinates(self, lidar_points, detections, transform):
        results = []

        if transform is None:
            print("没有可用的TF变换")
            return results
        
        rotation = transform.transform.rotation
        translation = transform.transform.translation
        q = [rotation.x, rotation.y, rotation.z, rotation.w]
        t = [translation.x, translation.y, translation.z]

        rot_matrix = quaternion_matrix(q)
        trans_matrix = translation_matrix(t)
        transform_matrix = np.dot(trans_matrix, rot_matrix)
        
        for detection in detections:
            bbox_points = []
            projected_points = []
            
            for point in lidar_points:
                X = np.array([point[0], point[1], point[2], 1.0])
                Y = self.P_rect @ self.R_rect @ self.RT @ X
                pt_x = Y[0] / Y[2]
                pt_y = Y[1] / Y[2]
                
                if (detection.box_2d[0] <= pt_x <= detection.box_2d[0] + detection.box_2d[2] and
                    detection.box_2d[1] <= pt_y <= detection.box_2d[1] + detection.box_2d[3]):
                    bbox_points.append(point)
                    projected_points.append((pt_x, pt_y))
            
            if bbox_points:
                clusters = LidarProcessor.cluster_points(bbox_points)
                
                if clusters:
                    largest_cluster = max(clusters, key=len)
                    
                    center, bbox = LidarProcessor.compute_3d_bbox(largest_cluster)

                    center_global = np.dot(transform_matrix, np.array([center[0], center[1], center[2], 1]))[:3]
                    
                    bbox_corners = [
                        [bbox[0], bbox[1], bbox[2], 1],  # min_x, min_y, min_z
                        [bbox[0], bbox[1], bbox[5], 1],  # min_x, min_y, max_z
                        [bbox[0], bbox[4], bbox[2], 1],  # min_x, max_y, min_z
                        [bbox[0], bbox[4], bbox[5], 1],  # min_x, max_y, max_z
                        [bbox[3], bbox[1], bbox[2], 1],  # max_x, min_y, min_z
                        [bbox[3], bbox[1], bbox[5], 1],  # max_x, min_y, max_z
                        [bbox[3], bbox[4], bbox[2], 1],  # max_x, max_y, min_z
                        [bbox[3], bbox[4], bbox[5], 1]   # max_x, max_y, max_z
                    ]

                    global_corners = [np.dot(transform_matrix, corner)[:3] for corner in bbox_corners]
                    min_x = min(c[0] for c in global_corners)
                    min_y = min(c[1] for c in global_corners)
                    min_z = min(c[2] for c in global_corners)
                    max_x = max(c[0] for c in global_corners)
                    max_y = max(c[1] for c in global_corners)
                    max_z = max(c[2] for c in global_corners)


                    obj = DetectedObject()
                    obj.class_id = detection.class_id
                    obj.class_name = self.class_names[detection.class_id]
                    obj.confidence = detection.confidence
                    obj.box_2d = detection.box_2d
                    # obj.position = center
                    # obj.bbox_3d = bbox
                    obj.position = center_global
                    obj.bbox_3d = (min_x, min_y, min_z, max_x, max_y, max_z)
                    obj.point_count = len(largest_cluster)
                    
                    results.append(obj)
        
        return results

    def shutdown(self):
        self.stop_event.set()
        for t in self.worker_threads:
            t.join()

def main():
    try:
        print("Initializing program...")
        collector = DataCollector()
        processor = FusionProcessor()
        
        while not rospy.is_shutdown():
            # print("\n" + "="*50)
            # print("Waiting for new sensor data...")
            
            # 获取数据
            try:
                pc_msg, img_msg, tf_msg = collector.get_data()
            except Exception as e:
                print(f"Exception: {e}")
            if pc_msg is None or img_msg is None or tf_msg is None:
                rospy.sleep(0.1)
                continue
            else:
                print("get data once")
                
            print("Processing new frame...")
            start_time = time.time()
            success = processor.process_data(pc_msg, img_msg, tf_msg)
            
            if success:
                print(f"Frame processing started in {time.time()-start_time:.2f}s")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        processor.shutdown()
        if hasattr(processor, 'rknn'):
            processor.rknnlite.release()
        print("Program shutdown.")

if __name__ == "__main__":
    main()
