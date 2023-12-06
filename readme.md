# 任务概要

本任务要求大家在本程序的main函数中的“代码区"中书写代码（可以额外添加其他程序文件），识别Mat src图像中的：

![img](./image/target.png)

![img](./image/R.png)

并使用cv::circle在长方形与R的中间绘制一个圆形，同时计算单次循环识别速度

其中图像中的风车（能量机关）将以正弦速度旋转，同时为了考虑现实中的观测噪声，我们为其添加了一定的噪声，其公式大致为：

$$
v = A * sin{(\omega t + \alpha)}+ b
$$

为大家提供三档要求：

- 识别能量机关标靶
- **拟合**能量机关旋转各参数，且已知参数范围（查看源码获得参数）
- **拟合**能量机关旋转各参数（请勿直接查看源码获得参数）

我们会检查大家的代码以确定大家的到底处于哪一档

# [第一步]
    为了减少运算量，首先现需要将生成的每一帧图片二值化，二值化后希望黑色区域仍未黑色，而红色图像变成白色。  
# [第二步]
    用cv自带的函数进行连通域检测，得到五个扇叶和中心R的位置。
# [第三步]
    获得各个连通域的位置后，需要筛选符合条件的图像:统计各个连通域内的白色像素的数目，显然，其它四个扇叶的白色像素数目在四千左右，而只有目标扇叶的像素数目在3500左右，并且显然R的像素会更少。
    因而通过这种方法，可以获得目标的位置
# [第四步]
    获取扇叶和R的重心位置，并计算连线角度，并统一角度衡量标准。定义时间刻度，每多少次收集到足够数据就作为样本计算。
# [第五步]
    利用ceres库进行拟合求解，获得参数输出。
# [实验结果]
    可以准确得识别特殊能量机关的位置和R的位置，并且通过300个数据获得拟合参数，较为准确地符合源文件中设定的数值。
