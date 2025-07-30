# 迁移指南：从Bimanual_Ur10e到BimanualRobot

## 概述

本指南说明如何将代码从使用分离的`DexLeft_Ur10e`和`DexRight_Ur10e`类迁移到使用合并USD文件的`BimanualRobot`类。

## USD文件结构

### dual_ur10e.usd 文件结构

你的`dual_ur10e.usd`文件具有以下结构：

```
/World/BimanualRobot/
├── ur10e_shadow_left_hand_glb/
│   ├── ee_link (左手末端执行器)
│   ├── fftip (左手指尖)
│   └── 关节 (shoulder_pan_joint, shoulder_lift_joint, ...)
└── ur10e_shadow_right_hand_glb/
    ├── ee_link (右手末端执行器)
    ├── fftip (右手指尖)
    └── 关节 (shoulder_pan_joint, shoulder_lift_joint, ...)
```

### 关节名称保持原样

- **左手关节**：`shoulder_pan_joint`, `shoulder_lift_joint`, `elbow_joint`, `wrist_1_joint`, `wrist_2_joint`, `wrist_3_joint`
- **左手手部关节**：`WRJ2`, `WRJ1`, `FFJ4`, `FFJ3`, `FFJ2`, `FFJ1`, `MFJ4`, `MFJ3`, `MFJ2`, `MFJ1`, `RFJ4`, `RFJ3`, `RFJ2`, `RFJ1`, `LFJ5`, `LFJ4`, `LFJ3`, `LFJ2`, `LFJ1`, `THJ5`, `THJ4`, `THJ3`, `THJ2`, `THJ1`
- **右手关节**：与左手相同的名称（通过索引范围区分）

### 关节索引分配

由于两个手臂有相同的关节名称，我们通过索引范围来区分：

- **左手手臂关节**：索引 0-5 (前6个关节)
- **左手手部关节**：索引 6-29 (接下来24个关节)
- **右手手臂关节**：索引 30-35 (接下来6个关节)
- **右手手部关节**：索引 36-59 (最后24个关节)

## 主要变化

### 1. 导入语句变化

**之前：**
```python
from Env_Config.Robot.BimanualDex_Ur10e import Bimanual_Ur10e
```

**现在：**
```python
from Env_Config.Robot.BimanualRobot import BimanualRobot
```

### 2. 类实例化变化

**之前：**
```python
# 需要分别指定左右手的位置和方向
bimanual_robot = Bimanual_Ur10e(
    world=world,
    dexleft_pos=np.array([0.0, 0.3, 0.0]),    # 左手位置
    dexleft_ori=np.array([0.0, 0.0, 0.0]),    # 左手方向
    dexright_pos=np.array([0.0, -0.3, 0.0]),  # 右手位置
    dexright_ori=np.array([0.0, 0.0, 0.0])    # 右手方向
)
```

**现在：**
```python
# 只需要指定一个基座位置和方向
bimanual_robot = BimanualRobot(
    world=world,
    translation=np.array([0.0, 0.0, 0.0]),    # 机器人基座位置
    orientation=np.array([0.0, 0.0, 0.0])     # 机器人基座方向
)
```

### 3. 方法调用保持不变

好消息是，所有的方法调用都保持不变：

```python
# 这些方法调用完全相同
bimanual_robot.set_both_hand_state(left_hand_state="open", right_hand_state="close")
bimanual_robot.dense_move_both_ik(left_pos, left_ori, right_pos, right_ori)
bimanual_robot.move_both_with_blocks(left_pos, left_ori, right_pos, right_ori)
```

## 迁移步骤

### 步骤1：更新导入语句

```python
# 旧代码
from Env_Config.Robot.BimanualDex_Ur10e import Bimanual_Ur10e

# 新代码
from Env_Config.Robot.BimanualRobot import BimanualRobot
```

### 步骤2：更新实例化代码

```python
# 旧代码
bimanual_robot = Bimanual_Ur10e(
    world=world,
    dexleft_pos=np.array([0.0, 0.3, 0.0]),
    dexleft_ori=np.array([0.0, 0.0, 0.0]),
    dexright_pos=np.array([0.0, -0.3, 0.0]),
    dexright_ori=np.array([0.0, 0.0, 0.0])
)

# 新代码
bimanual_robot = BimanualRobot(
    world=world,
    translation=np.array([0.0, 0.0, 0.0]),
    orientation=np.array([0.0, 0.0, 0.0])
)
```

### 步骤3：测试功能

运行你的代码，确保所有功能正常工作。代码会自动打印关节信息来帮助调试：

```
Total DOFs: 60
DOF names: [...]
Left arm indices: [0, 1, 2, 3, 4, 5]
Left hand indices: [6, 7, 8, ..., 29]
Right arm indices: [30, 31, 32, 33, 34, 35]
Right hand indices: [36, 37, 38, ..., 59]
```

## 优势

1. **性能提升**：只需要加载一个USD文件，减少了内存使用和加载时间
2. **简化管理**：只需要管理一个机器人实例，而不是两个独立的实例
3. **更好的同步**：两个手臂的运动可以更好地同步，因为它们属于同一个机器人
4. **代码简化**：减少了代码复杂度
5. **保持兼容性**：所有现有的方法调用都保持不变

## 注意事项

1. **关节索引**：代码会自动处理关节索引的分配，无需手动配置
2. **坐标系**：确保合并后的USD文件中的坐标系设置正确
3. **碰撞检测**：合并后的机器人可能有不同的碰撞检测行为
4. **物理属性**：确保物理属性（质量、惯性等）设置正确

## 故障排除

如果遇到问题，请检查：

1. USD文件路径是否正确
2. 关节数量是否为60个（6+24+6+24）
3. 末端执行器路径是否正确：`/World/BimanualRobot/ur10e_shadow_left_hand_glb/ee_link` 和 `/World/BimanualRobot/ur10e_shadow_right_hand_glb/ee_link`
4. 指尖路径是否正确：`/World/BimanualRobot/ur10e_shadow_left_hand_glb/fftip` 和 `/World/BimanualRobot/ur10e_shadow_right_hand_glb/fftip`

## 示例代码

完整的迁移示例：

```python
import numpy as np
from isaacsim.core.api import World
from Env_Config.Robot.BimanualRobot import BimanualRobot

# 创建世界
world = World()

# 创建双手机器人
bimanual_robot = BimanualRobot(
    world=world,
    translation=np.array([0.0, 0.0, 0.0]),
    orientation=np.array([0.0, 0.0, 0.0])
)

# 添加到场景
world.scene.add(bimanual_robot)
world.reset()

# 使用机器人（方法调用保持不变）
bimanual_robot.set_both_hand_state(left_hand_state="open", right_hand_state="close")

left_pos = np.array([0.5, 0.3, 0.5])
left_ori = np.array([1.0, 0.0, 0.0, 0.0])
right_pos = np.array([0.5, -0.3, 0.5])
right_ori = np.array([1.0, 0.0, 0.0, 0.0])

success = bimanual_robot.dense_move_both_ik(
    left_pos=left_pos,
    left_ori=left_ori,
    right_pos=right_pos,
    right_ori=right_ori
) 