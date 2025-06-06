命令见command.txt

请自行创建目录和名为position的ros包，之后可以将msg、scripts、src转移到ros包内

修改CMakeList.txt和package.xml后编译

先roscore 再rosrun运行主程序rknnlite_test

先修改最前面的参数地址

CLASSES_PATH：对应目标检测类别文件

PUB_TOPIC：选择检测结果要发布的topic

CAR_ID：当前部署车ID标识

MAX_QUEUE_SIZE：获取数据的size

NUM_WORKER_THREADS：工作线程数量

全局坐标转换相关参数：

CAR_MAP -> GLOBAL_MAP（CAR_MAP向GLOBAL_MAP转换）

transform = self.tf_buffer.lookup_transform(GLOBAL_MAP, CAR_MAP, pc_msg.header.stamp, rospy.Duration(0.1))

投影矩阵：

相机内参（camera_info）  projection_matrix -> P_rect    rectification_matrix -> R_rect

雷达外参（顺序x,y,z,qx,qy,qz,qw）-> RT

外参RT转换：

网站  https://staff.aist.go.jp/k.koide/workspace/matrix_converter/matrix_converter.html

数据粘贴进去后，先点击“TUM[tx ty tz qx qy qz qw]”按钮，再点击“Inverse”按钮，即获得RT矩阵，填进去即可

base env:

pip install opencv-python==4.11.0.86 fast-histogram==0.13 numpy==1.24.4 onnx==1.17.0 onnxruntime==1.16.3 protobuf==4.25.4 ruamel.yaml==0.18.10 psutil==7.0.0 scipy==1.10.1 tqdm==4.67.1 scikit-learn==1.3.0 pyyaml==5.3.1 -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install torch==1.10.1 -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install rknn_toolkit_lite2-2.3.2-cp38-cp38-manylinux_2_17_aarch64.manylinux2014_aarch64

sudo mv /usr/lib/librknnrt.so /usr/lib/librknnrt.so.bak

sudo cp librknnrt.so-1.6.0 /usr/lib/librknnrt.so
