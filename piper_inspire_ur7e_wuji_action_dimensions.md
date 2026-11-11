# PiPER + Inspire 与 UR7e + Wuji 完整动作维度说明

核对日期：2026-11-10。本文说明当前项目实际写入 Parquet 的动作格式，依据组装代码、IK 输出代码，以及当前使用的 URDF / Pinocchio 关节顺序。

## 1. 通用约定

| 项目        | 约定                                                                     |
| ----------- | ------------------------------------------------------------------------ |
| 索引        | 从 **0** 开始                                                      |
| Python 切片 | `[start:end]`，包含 start，不包含 end                                  |
| 左右顺序    | **右臂及右手在前，左臂及左手在后**                                 |
| 数据类型    | Parquet 中为 `list<float64>`，每行是一帧的完整关节位置向量             |
| 旋转关节    | 单位为 **弧度 rad**                                                |
| 平移关节    | 单位为 **米 m**，仅 PiPER 保留的原平行夹爪坐标使用此单位           |
| 数值语义    | 模型关节位置 `qpos`，不是速度、力矩或相邻帧差分                        |
| 统计归一化  | 当前组装步骤没有用 `stats.json` 对这些关节值进行统计归一化             |
| 末端位姿    | 向量中不包含末端 XYZ、欧拉角或四元数；末端位姿由机械臂关节角通过 FK 得到 |

关节角是模型零位下的坐标。数值为 0 不保证整只手或机械臂处于完全伸直状态；正负方向由所用 URDF 的关节轴和零位决定，不能将左右手角度统一取反来互换。

## 2. 两个字段的整体结构

| 字段                     | 总维度 | 拼接顺序                                                |
| ------------------------ | -----: | ------------------------------------------------------- |
| `action.piper_inspire` |     40 | 右 PiPER 8 + 右 Inspire 12 + 左 PiPER 8 + 左 Inspire 12 |
| `action.ur7e_wuji`     |     52 | 右 UR7e 6 + 右 Wuji 20 + 左 UR7e 6 + 左 Wuji 20         |

```text
action.piper_inspire，40 维
  [ 0: 6] 右 PiPER 机械臂，6 个旋转关节
  [ 6: 8] 右 PiPER 原平行夹爪，2 个平移关节
  [ 8:20] 右 Inspire，12 个关节位置
  [20:26] 左 PiPER 机械臂，6 个旋转关节
  [26:28] 左 PiPER 原平行夹爪，2 个平移关节
  [28:40] 左 Inspire，12 个关节位置

action.ur7e_wuji，52 维
  [ 0: 6] 右 UR7e 机械臂，6 个旋转关节
  [ 6:26] 右 Wuji，20 个旋转关节
  [26:32] 左 UR7e 机械臂，6 个旋转关节
  [32:52] 左 Wuji，20 个旋转关节
```

## 3. `action.piper_inspire`：全部 40 维

### 3.1 右侧：索引 0～19

| 全局索引 | 组件           | 模型关节名称                   | 含义                           | 单位 | 独立 / 联动关系 |
| -------: | -------------- | ------------------------------ | ------------------------------ | ---- | --------------- |
|        0 | PiPER 右臂     | `joint1`                     | 第1旋转关节，底座回转          | rad  | 机械臂关节      |
|        1 | PiPER 右臂     | `joint2`                     | 第2旋转关节，肩部              | rad  | 机械臂关节      |
|        2 | PiPER 右臂     | `joint3`                     | 第3旋转关节，肘部              | rad  | 机械臂关节      |
|        3 | PiPER 右臂     | `joint4`                     | 第4旋转关节，腕部第1轴         | rad  | 机械臂关节      |
|        4 | PiPER 右臂     | `joint5`                     | 第5旋转关节，腕部第2轴         | rad  | 机械臂关节      |
|        5 | PiPER 右臂     | `joint6`                     | 第6旋转关节，腕部第3轴         | rad  | 机械臂关节      |
|        6 | PiPER 右原夹爪 | `joint7`                     | 原平行夹爪一侧手指的滑动坐标   | m    | `+g_right`    |
|        7 | PiPER 右原夹爪 | `joint8`                     | 原平行夹爪另一侧手指的滑动坐标 | m    | `-g_right`    |
|        8 | Inspire 右手   | `index_proximal_joint`       | 食指近端弯曲                   | rad  | 独立            |
|        9 | Inspire 右手   | `index_intermediate_joint`   | 食指中间关节弯曲               | rad  | 跟随索引 8      |
|       10 | Inspire 右手   | `middle_proximal_joint`      | 中指近端弯曲                   | rad  | 独立            |
|       11 | Inspire 右手   | `middle_intermediate_joint`  | 中指中间关节弯曲               | rad  | 跟随索引 10     |
|       12 | Inspire 右手   | `pinky_proximal_joint`       | 小指近端弯曲                   | rad  | 独立            |
|       13 | Inspire 右手   | `pinky_intermediate_joint`   | 小指中间关节弯曲               | rad  | 跟随索引 12     |
|       14 | Inspire 右手   | `ring_proximal_joint`        | 无名指近端弯曲                 | rad  | 独立            |
|       15 | Inspire 右手   | `ring_intermediate_joint`    | 无名指中间关节弯曲             | rad  | 跟随索引 14     |
|       16 | Inspire 右手   | `thumb_proximal_yaw_joint`   | 拇指根部侧向转动 / 对掌        | rad  | 独立            |
|       17 | Inspire 右手   | `thumb_proximal_pitch_joint` | 拇指近端弯曲                   | rad  | 独立            |
|       18 | Inspire 右手   | `thumb_intermediate_joint`   | 拇指中间关节弯曲               | rad  | 跟随索引 17     |
|       19 | Inspire 右手   | `thumb_distal_joint`         | 拇指远端关节弯曲               | rad  | 跟随索引 17     |

### 3.2 左侧：索引 20～39

| 全局索引 | 组件           | 模型关节名称                   | 含义                           | 单位 | 独立 / 联动关系 |
| -------: | -------------- | ------------------------------ | ------------------------------ | ---- | --------------- |
|       20 | PiPER 左臂     | `joint1`                     | 第1旋转关节，底座回转          | rad  | 机械臂关节      |
|       21 | PiPER 左臂     | `joint2`                     | 第2旋转关节，肩部              | rad  | 机械臂关节      |
|       22 | PiPER 左臂     | `joint3`                     | 第3旋转关节，肘部              | rad  | 机械臂关节      |
|       23 | PiPER 左臂     | `joint4`                     | 第4旋转关节，腕部第1轴         | rad  | 机械臂关节      |
|       24 | PiPER 左臂     | `joint5`                     | 第5旋转关节，腕部第2轴         | rad  | 机械臂关节      |
|       25 | PiPER 左臂     | `joint6`                     | 第6旋转关节，腕部第3轴         | rad  | 机械臂关节      |
|       26 | PiPER 左原夹爪 | `joint7`                     | 原平行夹爪一侧手指的滑动坐标   | m    | `+g_left`     |
|       27 | PiPER 左原夹爪 | `joint8`                     | 原平行夹爪另一侧手指的滑动坐标 | m    | `-g_left`     |
|       28 | Inspire 左手   | `index_proximal_joint`       | 食指近端弯曲                   | rad  | 独立            |
|       29 | Inspire 左手   | `index_intermediate_joint`   | 食指中间关节弯曲               | rad  | 跟随索引 28     |
|       30 | Inspire 左手   | `middle_proximal_joint`      | 中指近端弯曲                   | rad  | 独立            |
|       31 | Inspire 左手   | `middle_intermediate_joint`  | 中指中间关节弯曲               | rad  | 跟随索引 30     |
|       32 | Inspire 左手   | `pinky_proximal_joint`       | 小指近端弯曲                   | rad  | 独立            |
|       33 | Inspire 左手   | `pinky_intermediate_joint`   | 小指中间关节弯曲               | rad  | 跟随索引 32     |
|       34 | Inspire 左手   | `ring_proximal_joint`        | 无名指近端弯曲                 | rad  | 独立            |
|       35 | Inspire 左手   | `ring_intermediate_joint`    | 无名指中间关节弯曲             | rad  | 跟随索引 34     |
|       36 | Inspire 左手   | `thumb_proximal_yaw_joint`   | 拇指根部侧向转动 / 对掌        | rad  | 独立            |
|       37 | Inspire 左手   | `thumb_proximal_pitch_joint` | 拇指近端弯曲                   | rad  | 独立            |
|       38 | Inspire 左手   | `thumb_intermediate_joint`   | 拇指中间关节弯曲               | rad  | 跟随索引 37     |
|       39 | Inspire 左手   | `thumb_distal_joint`         | 拇指远端关节弯曲               | rad  | 跟随索引 37     |

### 3.3 PiPER 原夹爪维度

当前字段保留了 PiPER 原平行夹爪的两个坐标。它们来自原 PiPER IK 输出，不属于 Inspire，也不是两个额外的机械臂旋转关节。

每侧夹爪在 IK 中按以下方式赋值：

```text
g = clip(grasp × 0.035, 0, 0.035)   # 单位：米
joint7 = +g
joint8 = -g
```

因此 40 维中共有 **36 个角度坐标和 4 个平移坐标**。在将 PiPER 末端替换成 Inspire 的组合模型中，这四个原夹爪坐标属于当前数据格式保留的冗余；本文按实际保存格式列出，不删除或重新编号。

### 3.4 Inspire 的 12 个坐标与 6 个独立控制量

Inspire 每只手的局部顺序是：

```text
0  食指近端         1  食指中间
2  中指近端         3  中指中间
4  小指近端         5  小指中间
6  无名指近端       7  无名指中间
8  拇指根部侧转     9  拇指近端弯曲
10 拇指中间         11 拇指远端
```

**小指在无名指前，拇指在最后。** 这是当前 Pinocchio `qpos` 顺序，不是按拇指到小指排列。

局部索引 `0、2、4、6、8、9` 是六个独立关节；其余六个是 URDF 的 mimic 联动关节。当前模型的联动规则如下，角度均为 rad：

```text
q[1]  = 1.06399 × q[0] - 0.04545
q[3]  = 1.06399 × q[2] - 0.04545
q[5]  = 1.06399 × q[4] - 0.04545
q[7]  = 1.06399 × q[6] - 0.04545
q[10] = 1.334   × q[9]
q[11] = 0.667   × q[9]
```

优化器展开联动坐标后会按关节限位裁剪，后续还会进行时序滤波，因此以上公式描述的是模型联动规则，不应当用作所有输出帧的严格浮点等式校验。

这 12 维是模型完整关节位置，不是六个硬件电机命令的重复存储，也不是硬件协议的 0～1000 指令值。

## 4. `action.ur7e_wuji`：全部 52 维

### 4.1 右侧：索引 0～25

| 全局索引 | 组件          | 模型关节名称             | 含义                           | 单位 |
| -------: | ------------- | ------------------------ | ------------------------------ | ---- |
|        0 | UR7e 右臂     | `shoulder_pan_joint`   | 肩部水平回转                   | rad  |
|        1 | UR7e 右臂     | `shoulder_lift_joint`  | 肩部抬升                       | rad  |
|        2 | UR7e 右臂     | `elbow_joint`          | 肘部屈伸                       | rad  |
|        3 | UR7e 右臂     | `wrist_1_joint`        | 腕部第1轴                      | rad  |
|        4 | UR7e 右臂     | `wrist_2_joint`        | 腕部第2轴                      | rad  |
|        5 | UR7e 右臂     | `wrist_3_joint`        | 腕部第3轴                      | rad  |
|        6 | Wuji 右拇指   | `right_finger1_joint1` | 拇指根部内收 / 外展，CMC 第1轴 | rad  |
|        7 | Wuji 右拇指   | `right_finger1_joint2` | 拇指根部侧摆，CMC 第2轴        | rad  |
|        8 | Wuji 右拇指   | `right_finger1_joint3` | 拇指近端屈伸                   | rad  |
|        9 | Wuji 右拇指   | `right_finger1_joint4` | 拇指远端屈伸                   | rad  |
|       10 | Wuji 右食指   | `right_finger2_joint1` | 食指掌指关节屈伸               | rad  |
|       11 | Wuji 右食指   | `right_finger2_joint2` | 食指掌指关节侧摆               | rad  |
|       12 | Wuji 右食指   | `right_finger2_joint3` | 食指近端指间关节 PIP 弯曲      | rad  |
|       13 | Wuji 右食指   | `right_finger2_joint4` | 食指远端指间关节 DIP 弯曲      | rad  |
|       14 | Wuji 右中指   | `right_finger3_joint1` | 中指掌指关节屈伸               | rad  |
|       15 | Wuji 右中指   | `right_finger3_joint2` | 中指掌指关节侧摆               | rad  |
|       16 | Wuji 右中指   | `right_finger3_joint3` | 中指近端指间关节 PIP 弯曲      | rad  |
|       17 | Wuji 右中指   | `right_finger3_joint4` | 中指远端指间关节 DIP 弯曲      | rad  |
|       18 | Wuji 右无名指 | `right_finger4_joint1` | 无名指掌指关节屈伸             | rad  |
|       19 | Wuji 右无名指 | `right_finger4_joint2` | 无名指掌指关节侧摆             | rad  |
|       20 | Wuji 右无名指 | `right_finger4_joint3` | 无名指近端指间关节 PIP 弯曲    | rad  |
|       21 | Wuji 右无名指 | `right_finger4_joint4` | 无名指远端指间关节 DIP 弯曲    | rad  |
|       22 | Wuji 右小指   | `right_finger5_joint1` | 小指掌指关节屈伸               | rad  |
|       23 | Wuji 右小指   | `right_finger5_joint2` | 小指掌指关节侧摆               | rad  |
|       24 | Wuji 右小指   | `right_finger5_joint3` | 小指近端指间关节 PIP 弯曲      | rad  |
|       25 | Wuji 右小指   | `right_finger5_joint4` | 小指远端指间关节 DIP 弯曲      | rad  |

### 4.2 左侧：索引 26～51

| 全局索引 | 组件          | 模型关节名称            | 含义                           | 单位 |
| -------: | ------------- | ----------------------- | ------------------------------ | ---- |
|       26 | UR7e 左臂     | `shoulder_pan_joint`  | 肩部水平回转                   | rad  |
|       27 | UR7e 左臂     | `shoulder_lift_joint` | 肩部抬升                       | rad  |
|       28 | UR7e 左臂     | `elbow_joint`         | 肘部屈伸                       | rad  |
|       29 | UR7e 左臂     | `wrist_1_joint`       | 腕部第1轴                      | rad  |
|       30 | UR7e 左臂     | `wrist_2_joint`       | 腕部第2轴                      | rad  |
|       31 | UR7e 左臂     | `wrist_3_joint`       | 腕部第3轴                      | rad  |
|       32 | Wuji 左拇指   | `left_finger1_joint1` | 拇指根部内收 / 外展，CMC 第1轴 | rad  |
|       33 | Wuji 左拇指   | `left_finger1_joint2` | 拇指根部侧摆，CMC 第2轴        | rad  |
|       34 | Wuji 左拇指   | `left_finger1_joint3` | 拇指近端屈伸                   | rad  |
|       35 | Wuji 左拇指   | `left_finger1_joint4` | 拇指远端屈伸                   | rad  |
|       36 | Wuji 左食指   | `left_finger2_joint1` | 食指掌指关节屈伸               | rad  |
|       37 | Wuji 左食指   | `left_finger2_joint2` | 食指掌指关节侧摆               | rad  |
|       38 | Wuji 左食指   | `left_finger2_joint3` | 食指近端指间关节 PIP 弯曲      | rad  |
|       39 | Wuji 左食指   | `left_finger2_joint4` | 食指远端指间关节 DIP 弯曲      | rad  |
|       40 | Wuji 左中指   | `left_finger3_joint1` | 中指掌指关节屈伸               | rad  |
|       41 | Wuji 左中指   | `left_finger3_joint2` | 中指掌指关节侧摆               | rad  |
|       42 | Wuji 左中指   | `left_finger3_joint3` | 中指近端指间关节 PIP 弯曲      | rad  |
|       43 | Wuji 左中指   | `left_finger3_joint4` | 中指远端指间关节 DIP 弯曲      | rad  |
|       44 | Wuji 左无名指 | `left_finger4_joint1` | 无名指掌指关节屈伸             | rad  |
|       45 | Wuji 左无名指 | `left_finger4_joint2` | 无名指掌指关节侧摆             | rad  |
|       46 | Wuji 左无名指 | `left_finger4_joint3` | 无名指近端指间关节 PIP 弯曲    | rad  |
|       47 | Wuji 左无名指 | `left_finger4_joint4` | 无名指远端指间关节 DIP 弯曲    | rad  |
|       48 | Wuji 左小指   | `left_finger5_joint1` | 小指掌指关节屈伸               | rad  |
|       49 | Wuji 左小指   | `left_finger5_joint2` | 小指掌指关节侧摆               | rad  |
|       50 | Wuji 左小指   | `left_finger5_joint3` | 小指近端指间关节 PIP 弯曲      | rad  |
|       51 | Wuji 左小指   | `left_finger5_joint4` | 小指远端指间关节 DIP 弯曲      | rad  |

### 4.3 Wuji 关节顺序与自由度

每只手的局部顺序为：

```text
[ 0: 4] finger1：拇指，joint1 → joint4
[ 4: 8] finger2：食指，joint1 → joint4
[ 8:12] finger3：中指，joint1 → joint4
[12:16] finger4：无名指，joint1 → joint4
[16:20] finger5：小指，joint1 → joint4
```

Wuji 当前模型每只手有 **20 个独立旋转关节**，没有 Inspire 那样的 mimic 联动。四根非拇指的第1关节负责根部屈伸，第2关节负责侧摆，第3、第4关节负责后续两节弯曲；不能把每根手指的四维都理解成同一个方向的弯曲。

当前代码还会对右小指的局部索引 `18、19`，即全局索引 `24、25`，执行基于关键点弯曲角的修正和平滑；这不会改变它们的关节含义、索引或 rad 单位。

`action.ur7e_wuji` 的 **52 维全部是角度坐标**，没有平行夹爪位移或额外的手腕位姿维度。

## 5. 有效性字段

| 字段                           | 类型          | 是否占用上述动作维度  |
| ------------------------------ | ------------- | --------------------- |
| `action.piper_inspire_valid` | 每帧一个 bool | 否，独立的 Parquet 列 |
| `action.ur7e_wuji_valid`     | 每帧一个 bool | 否，独立的 Parquet 列 |

当前组装实现将双手源姿态有效性与对应机械臂的双臂 IK 有效性取逻辑与，生成组合动作的有效标志。它是整帧标志，不是 40 或 52 个逐关节有效标志，也不代表对每个灵巧手关节单独完成了跟踪误差验证。
