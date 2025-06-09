#!/usr/bin/env python3

import torch
import cv2, os, time, yaml, struct
import numpy as np
from typing import List, Tuple
from sklearn.cluster import DBSCAN

import rospy
import logging
logging.basicConfig(level=logging.INFO)
rospy.init_node('data_collector_visX', anonymous=True)
from cv_bridge import CvBridge
import message_filters
from sensor_msgs.msg import PointCloud2, CompressedImage
import sensor_msgs.point_cloud2 as pc2
from position.msg import Detection, Detections
from geometry_msgs.msg import Point
import threading
from scipy.spatial.transform import Rotation
import sys

# config
RKNN_MODEL = "/home/orangepi/position/src/position/src/models/yolov8n-no-building.rknn"
CLASSES_PATH = "/home/orangepi/position/src/position/src/models/dataset2.yaml"
VISIMG_SAVE_DIR = "/home/orangepi/position/src/position/vis-test1"

PC_TOPIC = "/sctX/ouster/points"
IMG_TOPIC = "/camera/image/compressed"
CAR_ID = "carX-vis"

MAX_X = 100
MAX_Y = 50.0
MIN_Z = -2.0
MIN_REFLECTIVITY = 0.01
MAX_DISTANCE_FOR_COLOR = 10.0

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
        print("init DataCollector")
        self._init_components()

    def _init_components(self):
        self.bridge = CvBridge()
        self.sync_data = None
        self.lock = threading.Lock()
        
        pc_sub = message_filters.Subscriber(PC_TOPIC, PointCloud2)
        img_sub = message_filters.Subscriber(IMG_TOPIC, CompressedImage)
        
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [pc_sub, img_sub],
            queue_size=10,
            slop=0.05
        )
        self.ts.registerCallback(self.sync_callback)

    def sync_callback(self, pc_msg, img_msg):
        with self.lock:
            self.sync_data = (pc_msg, img_msg)

    def get_data(self):
        while not rospy.is_shutdown():
            with self.lock:
                if self.sync_data is not None:
                    data = self.sync_data
                    self.sync_data = None
                    print("get one data")
                    return data
            rospy.sleep(0.01)
        return None, None, None

class LidarProcessor:
    @staticmethod
    def load_calibration_data():
        """Load camera calibration matrices"""
        RT1 = np.array([
            [-4.670000e-03, -9.999900e-01, -8.800000e-04, -3.125000e-02],
            [-2.516000e-02,  1.000000e-03, -9.996800e-01, -4.237000e-02],
            [ 9.996700e-01, -4.650000e-03, -2.517000e-02, -1.401700e-01],
            [ 0.000000e+00,  0.000000e+00,  0.000000e+00,  1.000000e+00]
        ])

        RT2 = np.array([
            [ 1.605000e-02, -9.998700e-01, -2.060000e-03, -9.482000e-02],
            [-1.267000e-02,  1.850000e-03, -9.999200e-01, -1.262700e-01],
            [ 9.997900e-01,  1.607000e-02, -1.264000e-02, -6.129000e-01],
            [ 0.000000e+00,  0.000000e+00,  0.000000e+00,  1.000000e+00]
        ])

        RT = RT1

        # compensation_x = 0.10
        # RT[0, 3] += compensation_x
        
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
        """Process ROS PointCloud2 message into filtered points"""
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
        """Cluster points using DBSCAN"""
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
        """Compute 3D bounding box for a cluster of points"""
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
    def load_rknn_model():
        """Load and initialize RKNN model"""
        from rknnlite.api import RKNNLite
        rknn = RKNNLite()
        
        ret = rknn.load_rknn(RKNN_MODEL)
        if ret != 0:
            raise RuntimeError('Load model failed!')
        
        print('--> Init runtime environment')
        ret = rknn.init_runtime()
        if ret != 0:
            raise RuntimeError('Init runtime environment failed!')
        
        return rknn

    @staticmethod
    def run_detection(rknn, img, class_names):
        """Run object detection on image"""
        h_orig, w_orig = img.shape[:2]
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized, ratio, (dw, dh) = RKNNProcessor.letterbox(img_rgb)
        img_input = np.expand_dims(img_resized, 0)
        
        outputs = rknn.inference(inputs=[img_input], data_format=['nhwc'])
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
                detection.box = [x1, y1, x2 - x1, y2 - y1]
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
    def filter_boxes(boxes, box_confidences, box_class_probs, OBJ_THRESH=0.45):
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
    def nms_boxes(boxes, scores, NMS_THRESH=0.25):
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
        print("init Processor")
        self.bridge = CvBridge()
        self.class_names = self.load_class_names()
        self.rknn = RKNNProcessor.load_rknn_model()
        self.P_rect, self.R_rect, self.RT = LidarProcessor.load_calibration_data()
        self.detection_pub = rospy.Publisher('detection', Detections, queue_size=10)

    def load_class_names(self):
        with open(CLASSES_PATH, 'r') as f:
            data = yaml.safe_load(f)
            return data.get('names', [])

    def process_data(self, pc_msg, img_msg):
        """Main processing pipeline"""
        try:

            # 1. Process image
            try:
                img = self.bridge.compressed_imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
            except Exception as e:
                print(f"Process image error: {str(e)}")
            # 2. Process point cloud
            try:
                lidar_points = LidarProcessor.process_point_cloud(pc_msg)
            except Exception as e:
                print(f"Process points error: {str(e)}")
            # 3. Run object detection
            try:
                detections, _ = RKNNProcessor.run_detection(self.rknn, img, self.class_names)
            except Exception as e:
                print(f"run_detection error: {str(e)}")
                raise
            # 4. Get 3D coordinates
            try:
                results = self.get_object_3d_coordinates(lidar_points, detections)
            except Exception as e:
                print(f"get_object_3d_coordinates error: {str(e)}")
            # 5. Visualize results
            vis_img = None
            try:
                vis_img = self.visualize_results(img, lidar_points, results)
            except Exception as e:
                print(f"visualize_results error: {str(e)}")
            # 6. Publish results
            # try:
            #     self.publish_detections(results, pc_msg.header)
            # except Exception as e:
            #     print(f"publish_detections error: {str(e)}")

            return results, vis_img
        
        except Exception as e:
            print(f"Processing error: {str(e)}")
            return [], None
    
    def publish_detections(self, results, pc_header):
        """Publish results as ROS message"""
        detections_msg = Detections()
        detections_msg.header = pc_header
        detections_msg.car_id = CAR_ID
        

        for i, obj in enumerate(results):
            det_msg = Detection()
            det_msg.object_id = f"obj_{i}_{pc_header.seq}"
            det_msg.type = obj.class_name  # use cache
            det_msg.confidence = obj.confidence
            det_msg.box_2d = obj.box_2d
            
            pos = Point()
            pos.x, pos.y, pos.z = obj.position
            det_msg.position = pos
            
            det_msg.bbox_3d = list(obj.bbox_3d)
            detections_msg.detections.append(det_msg)

        # print
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

    def get_object_3d_coordinates(self, lidar_points, detections):
        """Get 3D coordinates of detected objects"""
        results = []
        
        for detection in detections:
            bbox_points = []
            projected_points = []
            
            # Filter and project points
            for point in lidar_points:
                X = np.array([point[0], point[1], point[2], 1.0])
                Y = self.P_rect @ self.R_rect @ self.RT @ X
                pt_x = Y[0] / Y[2]
                pt_y = Y[1] / Y[2]
                
                if (detection.box[0] <= pt_x <= detection.box[0] + detection.box[2] and
                    detection.box[1] <= pt_y <= detection.box[1] + detection.box[3]):
                    bbox_points.append(point)
                    projected_points.append((pt_x, pt_y))
            
            if bbox_points:
                # Cluster points within detection box
                clusters = LidarProcessor.cluster_points(bbox_points)
                
                if clusters:
                    # Find largest cluster
                    largest_cluster = max(clusters, key=len)
                    
                    # Compute 3D bbox and center
                    center, bbox = LidarProcessor.compute_3d_bbox(largest_cluster)
                    
                    # Create result objects
                    obj = DetectedObject()
                    obj.class_id = detection.class_id
                    obj.class_name = self.class_names[detection.class_id]  # cache
                    obj.confidence = detection.confidence
                    obj.box_2d = detection.box
                    obj.position = center
                    obj.bbox_3d = bbox
                    obj.point_count = len(largest_cluster)
                    
                    results.append(obj)
        
        return results

    def visualize_results(self, img, lidar_points, results):
        """Visualize detection and fusion results"""
        vis_img = img.copy()
        overlay = img.copy()
        
        # Project all LiDAR points
        for point in lidar_points:
            X = np.array([point[0], point[1], point[2], 1.0])
            Y = self.P_rect @ self.R_rect @ self.RT @ X
            pt_x = Y[0] / Y[2]
            pt_y = Y[1] / Y[2]
            
            if 0 <= pt_x < img.shape[1] and 0 <= pt_y < img.shape[0]:
                val = point[0]
                max_val = MAX_DISTANCE_FOR_COLOR
                red = min(255, int(255 * abs((val - max_val) / max_val)))
                green = min(255, int(255 * (1 - abs((val - max_val) / max_val))))
                cv2.circle(overlay, (int(pt_x), int(pt_y)), 5, (0, green, red), -1)
        
        # Blend images
        opacity = 0.6
        cv2.addWeighted(overlay, opacity, vis_img, 1 - opacity, 0, vis_img)
        
        # Draw detection results
        for i, result in enumerate(results):
            
            # Random color for each detection
            color = np.random.randint(0, 256, size=3).tolist()
            
            # Draw bounding box
            x, y, w, h = result.box_2d
            cv2.rectangle(vis_img, (x, y), (x + w, y + h), color, 2)
            
            # Prepare label
            label = f"{self.class_names[result.class_id]} {result.confidence:.2f}"
            
            # Draw label background
            cv2.rectangle(vis_img, (x, y - 20), (x + len(label) * 10, y), color, -1)
            
            # Draw label text
            cv2.putText(vis_img, label, (x, y - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            
            # Print 3D info
            print(f"  Object {i+1}:")
            print(f"    Detection: {result.class_name} (Confidence: {result.confidence:.2f})")
            print(f"    3D Position: ({result.position[0]:.2f}, {result.position[1]:.2f}, {result.position[2]:.2f})")
            print(f"    3D BBox: min({result.bbox_3d[0]:.2f}, {result.bbox_3d[1]:.2f}, {result.bbox_3d[2]:.2f}) "
                  f"max({result.bbox_3d[3]:.2f}, {result.bbox_3d[4]:.2f}, {result.bbox_3d[5]:.2f})")
            print(f"    Points: {result.point_count}")
        
        return vis_img

def main():
    try:
        print("Initializing program...")
        collector = DataCollector()
        processor = FusionProcessor()

        while not rospy.is_shutdown():

            try:
                pc_msg, img_msg= collector.get_data()
            except Exception as e:
                print(f"get data failed: {e}")
        
            if pc_msg is None or img_msg is None:
                continue
                
            print("Processing new frame...")
            start_time = time.time()
            results, vis_img = processor.process_data(pc_msg, img_msg)
            
            # save img（timestamp）
            result_dir = VISIMG_SAVE_DIR
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
                print(f"Created result directory: {result_dir}")

            result_path = os.path.join(result_dir, f"fusion-result-{time.strftime('%Y%m%d-%H%M%S')}.jpg")
            cv2.imwrite(result_path, vis_img)
            print(f"Result saved to {result_path}")
            
            print(f"Frame processed in {time.time()-start_time:.2f}s")
            
    except Exception as e:
        print(f"main() catch Error: {e}")
    finally:
        if 'processor' in locals() and hasattr(processor, 'rknn'):
            processor.rknn.release()
        print("Program shutdown.")

if __name__ == "__main__":
    main()
