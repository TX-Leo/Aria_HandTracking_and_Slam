这是一个非常好的问题。Navigation Retargeting（导航重定向）是 EMMA 能够直接利用人类数据来训练轮式机器人导航的核心技术。

人类行走是**全向的（Omnidirectional）**，我们可以横着走、斜着走，且头部朝向（相机的朝向）和身体移动方向可以解耦。
而 EMMA 使用的 AgileX Tracer 底盘是**差速驱动（Differential Drive）**，它受限于非完整约束（Non-holonomic constraints），只能做直线运动或圆弧运动，无法横向平移，且相机固定在底盘上，视野随底盘转动。

为了填补这个“运动学鸿沟”，作者设计了一个优化算法。以下是详细的拆解：

---

### 1. 算法流程 (Pipeline)

重定向过程发生在**离线数据处理阶段**，目的是把人类的轨迹变成机器人“能走出来”的轨迹，作为训练标签（Ground Truth）。

1.  **3D 转 2D 投影：** 获取人类佩戴 Aria 眼镜的 3D 头部位姿 $H_d$，将其投影到 2D 地平面上，得到 $(x, y, \theta)$。
2.  **稀疏航点采样 (Waypoint Sparsity)：** 人类数据密度很高（30Hz），直接逐帧转换会有很多噪声（比如人站着不动时头部的微颤）。算法根据位移距离（例如每移动 0.25m 或 0.5m）采样关键航点 $\{h_{base}^k\}$。
3.  **轨迹优化 (Optimization)：** 这是核心。构建一个数学优化问题，寻找一系列机器人底盘的速度指令 $v$ (线速度) 和 $\omega$ (角速度)，使得机器人生成的轨迹尽可能贴合人类的航点，同时满足差速驱动的物理限制。

---

### 2. 数学原理 (Mathematical Formulation)

这是一个典型的**受约束优化问题（Constrained Optimization Problem）**。

#### 目标函数 (Cost Function)
我们需要找到速度序列 $\mathbf{z} = [(v_1, \omega_1), ..., (v_K, \omega_K)]$ 来最小化以下损失：

$$
\min_{\mathbf{z}} \sum_{k=1}^{K} \left[ \lambda_{pos} \| p_k(\mathbf{z}) - p_k^d \|^2_2 + \lambda_{yaw} \text{wrap}(\theta_k(\mathbf{z}) - \theta_k^d)^2 + \lambda_{smooth} \left( (v_k - v_{k-1})^2 + (\omega_k - \omega_{k-1})^2 \right) \right]
$$

**各项含义：**
1.  **位置误差 ($\lambda_{pos} \| ... \|^2$)：** 机器人的位置 $p_k$ 必须尽可能接近人类的航点 $p_k^d$。
2.  **朝向误差 ($\lambda_{yaw} \text{wrap}(...)^2$)：** 机器人的朝向 $\theta_k$ 必须尽可能接近人类的朝向 $\theta_k^d$。
    *   *注意：* 对于差速机器人，为了看清侧面的东西，必须转动整个底盘。
3.  **平滑项 ($\lambda_{smooth} ...$)：** 限制加速度，防止速度 $v$ 和角速度 $\omega$ 突变，保证机器人运行平稳。

#### 约束条件 (Constraints)
优化必须服从差速驱动的运动学方程：

1.  **运动学模型 (Kinematics)：**
    *   $x_{k+1} = x_k + v_k \cos(\theta_k) \Delta t$
    *   $y_{k+1} = y_k + v_k \sin(\theta_k) \Delta t$
    *   $\theta_{k+1} = \theta_k + \omega_k \Delta t$
    *(这意味着机器人不能横向移动，y轴的变化完全依赖于当前的朝向和线速度)*

2.  **物理限制 (Physical Limits)：**
    *   $v_{min} \leq v_k \leq v_{max}$ (最大线速度)
    *   $\omega_{min} \leq \omega_k \leq \omega_{max}$ (最大角速度)

---

### 3. 代码实现逻辑 (Conceptual Python Code)

虽然我们没有源码，但可以使用 Python 的 `scipy.optimize` 库复现这个逻辑。

```python
import numpy as np
from scipy.optimize import minimize

def kinematics_step(state, control, dt):
    x, y, theta = state
    v, w = control
    # 差速驱动更新逻辑
    new_x = x + v * np.cos(theta) * dt
    new_y = y + v * np.sin(theta) * dt
    new_theta = theta + w * dt
    return np.array([new_x, new_y, new_theta])

def loss_function(controls_flat, start_state, target_waypoints, dt, lambdas):
    # controls_flat 是优化器传入的一维数组，需要reshape成 (K, 2)
    controls = controls_flat.reshape(-1, 2)
    K = len(target_waypoints)
    current_state = start_state
    loss = 0
    
    prev_v, prev_w = 0, 0 # 用于计算平滑度
    
    for k in range(K):
        v, w = controls[k]
        
        # 1. 运行运动学模型，计算下一步位置
        current_state = kinematics_step(current_state, [v, w], dt)
        
        # 2. 获取当前步的目标点 (来自人类数据)
        target_pos = target_waypoints[k][:2]
        target_theta = target_waypoints[k][2]
        
        # 3. 计算各项 Loss
        pos_error = np.sum((current_state[:2] - target_pos)**2)
        yaw_error = (current_state[2] - target_theta)**2 # 实际需处理角度回绕
        smooth_error = (v - prev_v)**2 + (w - prev_w)**2
        
        loss += lambdas['pos'] * pos_error + \
                lambdas['yaw'] * yaw_error + \
                lambdas['smooth'] * smooth_error
                
        prev_v, prev_w = v, w
        
    return loss

def retarget_trajectory(human_waypoints):
    # 初始化
    K = len(human_waypoints)
    initial_guess = np.zeros(K * 2) # 初始猜测所有速度为0
    start_state = np.array([0, 0, 0]) # 假设从原点开始
    dt = 0.1 
    lambdas = {'pos': 32.0, 'yaw': 2.0, 'smooth': 1.0} # 论文中的参数
    
    # 速度限制边界
    bounds = [(-1.0, 1.0), (-np.pi, np.pi)] * K 
    
    # 运行优化
    result = minimize(
        loss_function, 
        initial_guess, 
        args=(start_state, human_waypoints, dt, lambdas),
        method='SLSQP',
        bounds=bounds
    )
    
    return result.x.reshape(-1, 2) # 返回优化后的机器人速度指令 (v, w)
```

### 4. 模型的输入与输出 (Model I/O)

**注意：** 上面提到的重定向算法是在**数据预处理阶段**完成的。处理完之后，我们得到了一组“干净的、机器人可执行的”轨迹数据。接下来是训练神经网络模型（EMMA Policy）。

**在训练和推理（Deployment）阶段：**

*   **输入 (Model Input):**
    1.  **视觉流 (Vision Stem):** 来自 Aria 眼镜的 RGB 图像（Egocentric View）。
    2.  **本体感知 (Proprioception Stem):**
        *   如果是人类数据：手部 3D 位姿。
        *   如果是机器人数据：机械臂关节状态、夹爪状态。
    3.  **当前状态：** 过去的观测历史（Context length）。

*   **输出 (Model Output):**
    模型并不直接输出速度 $(v, \omega)$，而是预测**未来的航点 (Waypoints)**。
    *   **Navigation Head:** 输出未来 $K$ 步的底盘航点 $(x, y, \theta)$。
        *   *为什么输出航点而不是速度？* 因为航点更稳定，且空间分布一致。
    *   **底层控制器 (Low-level Controller):** 在推理时，模型输出预测的航点，然后由一个简单的控制器（如 PID 或 纯追踪 Pure Pursuit）将这些航点实时转换为机器人的电机速度指令。

*   **特殊的相位开关 (Phase Switch):**
    模型还会输出一个 `phase` 预测（0代表操作，1代表导航）。
    *   如果预测是**操作**：强制将导航航点设为 0（底盘锁死），只动手。
    *   如果预测是**导航**：底盘跟随航点移动。

### 总结
EMMA 的导航重定向本质上是一个**基于优化的数据清洗过程**。它强行将灵活的人类轨迹“压”进了笨拙的轮式机器人的运动模型中，以此生成训练数据。训练后的模型学会了看图预测“机器人应该走的路径点”，从而实现了从人到机器人的能力迁移。