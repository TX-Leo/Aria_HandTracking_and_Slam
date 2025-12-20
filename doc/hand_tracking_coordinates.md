### Phase 2: 坐标系统一与重定向 (Coordinate Unification & Retargeting)

这是最核心的数学部分，你需要把“人的手”变成“机器人的末端轨迹”。

**1. 坐标系转换 (Transforms)**
你需要算出**手在世界坐标系下的绝对位姿**。
公式如下：
$$ T_{world \to hand} = T_{world \to device} \times T_{device \to cam} \times T_{cam \to hand} $$

*   $T_{world \to device}$: 来自 Aria MPS (SLAM结果)。
*   $T_{device \to cam}$: 来自 Aria 的出厂标定参数 (Extrinsics, IMU to RGB Camera)。
*   $T_{cam \to hand}$: 来自 HaWoR 的输出。

