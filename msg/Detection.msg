string object_id # 这一帧中目标检测目标的序号id
string type #类型
float32 confidence #置信度
int32[4] box_2d        # 2D边界框 [x_min, y_min, x_max, y_max]
geometry_msgs/Point position  # 3D坐标 (x, y, z)
float32[6] bbox_3d     # 3D边界框