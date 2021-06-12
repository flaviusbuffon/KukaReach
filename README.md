# Kuka Reach Point

This project is used to simulate human manipulation of the robotic arm and obtain coordinate records to compare the motion trajectory of the robotic arm under the reinforcement learning algorithm.



## Dir Structure

```bash
├── kuka-reach-point
    ├── .git
    ├── .idea
    ├── data        `contain cube urtf files`
    ├── main.py		`source code`
    ├── pos.txt		`Movement trajectory of the end of the robotic arm`
    ├── README.md
    ├── logRobot.txt  `Movement trajectory of the end of the robotic arm !Under RL`
    ├── warpx.png	`dtw score of x coordinate`
    ├──  warpy.png	`dtw score of y coordinate`
    └── log
        ├── "%Y%m%d_%H:%M:%S Task&"		`log files`
        ├──  ......
```

# Update History

## update GMT+8 6-12 04:40

* 机械臂运动更流畅
* 导出机械臂运动数据to log.txt 导出频率为 $20Hz$
* 添加了对四个task的比较完整的支持和接口

## update GMT+8 6-13 12:06

* 修复任务2的问题
* 使轨迹更真实可靠
* 添加中断，时间中断和条件（已触碰）中断
* 摄像头添加和初始摄像头设置

## update GMT+8 6-13 2:50

* 添加 `clue`
* 添加 `dtw score`