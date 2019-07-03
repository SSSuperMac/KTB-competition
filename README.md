
“空天杯”竞赛

## 实验环境
- python3.6
- pytorch 1.0.0
- CUDA 9.0
- cudnn 7.3
- 

## 技术方案：
1. 第一步，先实现单帧目标检测，确定主干网络（backbone network）。  
代码：`Single-Frame-Detection-2`,主要基于`CornerNet`修改，估计每个目标的中心点位置，free-anchor

2. 第二步，在1的基础上引入多分支结合LSTM实现多帧目标的检测。  
代码：Multi-Frame-Detection, 核心代码convLSTM

3. 第三步，在2的基础上引入跟踪模块  
代码：Tracking，主要参考SiamFC/SiamRPN网络，进行模板与图像的相关匹配

