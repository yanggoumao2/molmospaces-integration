#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
joystick_bi_yam.py — 北通/Xbox 手柄遥操作 MuJoCo 仿真双臂 YAM (Bimanual YAM)
=============================================================================
基于 joystick_yam.py 构建, 控制两台 YAM 机械臂:
  左臂: 基座位于世界 (0, +0.22, 0), 右臂: 基座位于世界 (0, -0.22, 0)
单臂遥操作逻辑与 joystick_yam.py 一致:
  手柄输入 -> 末端目标位姿(增量累积, 末端局部系三维旋转) -> DLS 雅可比 IK
  -> 写入关节 -> 仿真步进

与 joystick_yam.py 的按键差异 (用户自定义):
  A 键 ──> 切换当前控制的机械臂 (left <-> right)
  B 键 ──> 夹爪开合切换 (同一个键控制张开/闭合)
  X 键 ──> 回到 home    Y 键 ──> 打印位姿    Start ──> 退出
  摇杆/扳机/肩键只控制当前激活的机械臂; 切换后另一臂保持原位

手柄映射: 直接继承 joystick_yam.py 的 XboxController 映射层, 轴序/标定/
去漂移/扳机量程/平滑逻辑与其完全一致; 两个脚本共用 joystick_mapping.json:
  左摇杆 ──> 末端世界 X/Y 平移
  右摇杆 Y ──> 末端世界 Z 平移    右摇杆 X ──> 末端偏航 yaw
  LT/RT ──> 末端横滚 roll          LB/RB ──> 末端俯仰 pitch

未检测到手柄时自动回退为键盘控制 (同 MolmoSpaces viewer_keyboard_teleop.py):
  W/S: X±  A/D: Y±  R/F: Z±   (世界系平移, 作用于当前激活臂)
  Q/E: roll±  Z/C: pitch±  B/N: yaw±   (末端局部系旋转)
  Space: 夹爪开合切换   Tab: 切换机械臂   H: 回 home   P: 打印   Esc: 退出

运行方式:
  conda activate mujoco-learning               # 带有mujoco和pygame的环境均可
  python joystick_bi_yam.py --calibrate        # 交互式检测映射(如果不更换手柄，可以与单臂版共用文件)
  python joystick_bi_yam.py --verify           # 启动前实时验证映射
  python joystick_bi_yam.py

  python joystick_bi_yam.py --debug            # 实时打印归一化输入/raw 轴值
  python joystick_bi_yam.py --headless 500     # 无窗口自检模式(调试用)
"""

import argparse
import os
import sys
import time

import numpy as np
import pygame

import mujoco
import mujoco.viewer
from mujoco import MjSpec, mjtGeom, mjtObj

# 复用单臂版手柄映射层: 轴/按键标定、SDL GameController 修复、去漂移、
# 扳机量程标准化、输入平滑与 joystick_yam.py 完全一致
from joystick_yam import XboxController as SingleArmXboxController

# ---------------------------------------------------------------------------
# 常量: 模型路径 / 初始位形 / 工作空间
# ---------------------------------------------------------------------------
DEFAULT_XML = "model/yam/bimanual_yam.xml"

# 双臂初始位形 (与 MolmoSpaces BimanualYamRobotConfig.init_qpos 一致)
# qpos 顺序: 左臂6 + 左夹爪2 + 右臂6 + 右夹爪2 = 16
HOME_QPOS = np.array([
    # 左臂
    0.0, 0.0, 0.0, -0.0, 0.0, 0.0,
    # 左夹爪 (left_left_finger, left_right_finger) — 张开
    0.0, 0.0,
    # 右臂
    0.0, 0.0, 0.0, -0.0, 0.0, 0.0,
    # 右夹爪 — 张开
    0.0, 0.0,
])

# 双臂基座世界偏移 (左臂 +Y, 右臂 -Y, 间距 0.44m)
ARM_BASE_OFFSET = {
    "left": np.array([0.0, 0.22, 0.0]),
    "right": np.array([0.0, -0.22, 0.0]),
}
ARMS = ("left", "right")

# 末端目标工作空间边界 (基座相对坐标, 软约束)
EE_X_MIN, EE_X_MAX = 0.15, 0.50
EE_Y_MIN, EE_Y_MAX = -0.22, 0.22
EE_Z_MIN, EE_Z_MAX = 0.04, 0.35

# 手柄灵敏度 (满量程时: 平移约 0.06 m/s, 旋转约 10 deg/s @60fps)
POS_SENSITIVITY = 0.001    # 每单位摇杆位移对应的末端平移 (m)
ROT_SENSITIVITY = 0.003    # 每单位摇杆位移/按键对应的末端旋转 (rad)
DEADZONE = 0.1             # 摇杆死区
INPUT_SMOOTHING = 0.25     # 输入速度一阶平滑系数 (越小起步越缓)
MAX_ACCUM_ROT_DEG = 175.0  # 累计旋转安全上限 (度/轴), 防止异常输入导致机械臂缠绕

GRIPPER_OPEN_CTRL = 0.041  # 夹爪张开 ctrl
GRIPPER_CLOSE_CTRL = 0.0   # 夹爪闭合 ctrl


# ---------------------------------------------------------------------------
# 旋转工具
# ---------------------------------------------------------------------------
def rotmat_to_rotvec(R: np.ndarray) -> np.ndarray:
    """3x3 旋转矩阵 -> 旋转向量 (轴角), 用于计算姿态误差."""
    theta = np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    if abs(theta) < 1e-10:
        return np.zeros(3)
    axis = np.array([R[2, 1] - R[1, 2],
                     R[0, 2] - R[2, 0],
                     R[1, 0] - R[0, 1]]) / (2.0 * np.sin(theta))
    return theta * axis


def rodrigues(axis_angle: np.ndarray) -> np.ndarray:
    """轴角旋转向量 -> 旋转矩阵 (Rodrigues 公式)."""
    angle = np.linalg.norm(axis_angle)
    if angle < 1e-10:
        return np.eye(3)
    axis = axis_angle / angle
    K = np.array([[0.0, -axis[2], axis[1]],
                  [axis[2], 0.0, -axis[0]],
                  [-axis[1], axis[0], 0.0]])
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


# ---------------------------------------------------------------------------
# XboxController — 手柄控制器 (双臂版)
# ---------------------------------------------------------------------------
class XboxController(SingleArmXboxController):
    """北通/Xbox 手柄控制器 (双臂版).

    直接继承 joystick_yam.XboxController, 因此轴映射、SDL GameController 修复、
    joystick_mapping.json 标定/加载、静止值去漂移、扳机量程标准化、输入平滑
    与单臂版完全一致.

    与单臂版的差异:
      - 左右臂各维护一份目标 (基座相对位置 + 3x3 姿态 + 累计旋转);
      - 摇杆/扳机/肩键只作用于当前激活臂, 切换后另一臂保持原位;
      - A 键 = 切换激活臂, B 键 = 夹爪开合切换.
    """

    # 与 joystick_yam.py 完全相同的轴映射
    RAW_AXES = SingleArmXboxController.RAW_AXES
    AXIS_NAMES = SingleArmXboxController.AXIS_NAMES
    AXIS_PROMPTS = SingleArmXboxController.AXIS_PROMPTS

    # 按键差异: A=切换机械臂, B=夹爪开合切换, 其余 X/Y/LB/RB/Start 同单臂版
    RAW_BUTTONS = {"switch": 0, "gripper": 1, "home": 2, "print": 3,
                   "pitch_neg": 4, "pitch_pos": 5, "quit": 7}
    LOGICAL_BUTTONS = ("switch", "gripper", "home", "print",
                       "pitch_neg", "pitch_pos", "quit")
    BUTTON_PROMPTS = {
        "switch": "按一下 A 键(切换当前机械臂)",
        "gripper": "按一下 B 键(夹爪开合切换)",
        "home": "按一下 X 键(回 home)",
        "print": "按一下 Y 键(打印位姿)",
        "pitch_neg": "按住 LB 左肩键",
        "pitch_pos": "按住 RB 右肩键",
        "quit": "按一下 Start 键(退出)",
    }

    def __init__(self, home_targets, home_rot, mapping_path=None,
                 force_raw=False, force_sdl=False):
        """Args: home_targets/home_rot 均为 {"left":..., "right":...}."""
        # 先用单臂版完整初始化手柄/映射/静止值/平滑状态
        super().__init__(np.asarray(home_targets["left"], dtype=float),
                         np.asarray(home_rot["left"], dtype=float),
                         mapping_path=mapping_path,
                         force_raw=force_raw,
                         force_sdl=force_sdl)

        # 双臂各自目标 (基座相对位置 + 姿态 + 累计旋转)
        self.targets = {}
        for arm in ARMS:
            self.targets[arm] = {
                "pos": np.asarray(home_targets[arm], dtype=float).copy(),
                "rot": np.asarray(home_rot[arm], dtype=float).reshape(3, 3).copy(),
                "rot_accum": np.zeros(3),
            }
        self.active_arm = "left"
        self.switch_requested = False
        # 双臂 home 位形下夹爪初始均为张开
        self.gripper_open = True

        # 双臂版有自己的工作空间/灵敏度常量 (覆盖单臂版的基类默认值)
        self.x_min, self.x_max = EE_X_MIN, EE_X_MAX
        self.y_min, self.y_max = EE_Y_MIN, EE_Y_MAX
        self.z_min, self.z_max = EE_Z_MIN, EE_Z_MAX
        self.pos_sensitivity = POS_SENSITIVITY
        self.rot_sensitivity = ROT_SENSITIVITY
        self.input_smoothing = INPUT_SMOOTHING
        self.deadzone = DEADZONE
        self._pos_vel = np.zeros(3)
        self._rot_vel = np.zeros(3)

    # -- 按键读取 (GC 模式需要双臂版按键表, raw 模式复用基类) --------------
    def _button(self, name: str) -> bool:
        """按逻辑名读取按钮状态 (SDL 标准映射优先)."""
        if self.gc is not None:
            const = getattr(pygame, {
                "switch": "CONTROLLER_BUTTON_A",
                "gripper": "CONTROLLER_BUTTON_B",
                "home": "CONTROLLER_BUTTON_X",
                "print": "CONTROLLER_BUTTON_Y",
                "pitch_neg": "CONTROLLER_BUTTON_LEFTSHOULDER",
                "pitch_pos": "CONTROLLER_BUTTON_RIGHTSHOULDER",
                "quit": "CONTROLLER_BUTTON_START",
            }[name])
            try:
                return bool(self.gc.get_button(const))
            except Exception:
                return False
        return super()._button(name)

    # -- 旋转增量 (作用于指定臂) -------------------------------------------
    def _apply_rot_delta(self, arm: str, delta_rot: np.ndarray):
        """在当前目标局部系中施加旋转增量, 并做累计旋转安全钳制."""
        t = self.targets[arm]
        new_accum = t["rot_accum"] + delta_rot
        if np.max(np.abs(new_accum)) > np.radians(MAX_ACCUM_ROT_DEG):
            now = time.time()
            if now - self._last_clamp_print > 1.0:
                print("[XboxController] 累计旋转达到安全上限, 已忽略本次旋转输入")
                self._last_clamp_print = now
            return
        t["rot_accum"] = new_accum
        t["rot"] = t["rot"] @ rodrigues(delta_rot)

    # -- 输入处理 (作用于当前激活臂) ----------------------------------------
    def handle_input(self, arm=None):
        """处理手柄输入并更新激活臂目标. 每帧调用一次."""
        if not self.is_connected():
            return

        pygame.event.pump()

        # 同一帧内只读一次
        vals = {name: self._axis(name) for name in self.AXIS_NAMES}
        btns = {name: self._button(name) for name in self.LOGICAL_BUTTONS}

        dx, dy = vals["lx"], vals["ly"]
        dz, dyaw = -vals["ry"], vals["rx"]
        droll = vals["rt"] - vals["lt"]
        dpitch = float(btns["pitch_pos"]) - float(btns["pitch_neg"])

        # 一阶平滑: 起步/停止渐进
        desired_pos = np.array([dx, dy, dz], dtype=float)
        desired_rot = np.array([droll, dpitch, dyaw], dtype=float)
        s = float(self.input_smoothing)
        self._pos_vel += s * (desired_pos - self._pos_vel)
        self._rot_vel += s * (desired_rot - self._rot_vel)
        step_pos = np.clip(self._pos_vel, -1.0, 1.0) * self.pos_sensitivity
        step_rot = np.clip(self._rot_vel, -1.0, 1.0) * self.rot_sensitivity

        # 增量累积激活臂位置目标 (基座相对)
        t = self.targets[self.active_arm]
        t["pos"][0] = float(np.clip(t["pos"][0] + step_pos[0],
                                    self.x_min, self.x_max))
        t["pos"][1] = float(np.clip(t["pos"][1] + step_pos[1],
                                    self.y_min, self.y_max))
        t["pos"][2] = float(np.clip(t["pos"][2] + step_pos[2],
                                    self.z_min, self.z_max))

        # 增量累积激活臂姿态目标 (末端局部系)
        self._apply_rot_delta(self.active_arm, step_rot)

        # 按钮 (边沿触发)
        for name, action in [
            ("switch", self._request_switch),         # A: 切换机械臂
            ("gripper", self._request_gripper_toggle),  # B: 夹爪开合切换
            ("home", self._request_home),             # X: 回 home
            ("print", self._request_print),           # Y: 打印位姿
            ("quit", self._request_quit),             # Start: 退出
        ]:
            pressed = btns[name]
            if pressed and not self._btn_states.get(name, False):
                action()
            self._btn_states[name] = pressed

        # 调试: 打印归一化输入 + raw 轴值
        if self.debug and time.time() - self._last_debug_print > 0.5:
            pressed_names = [n for n in self.LOGICAL_BUTTONS if btns[n]]
            extra = ""
            if self.gc is None:
                raw_axes = [f"{i}:{self.controller.get_axis(i):+.2f}"
                            for i in range(self.controller.get_numaxes())]
                raw_btns = [i for i in range(self.controller.get_numbuttons())
                            if self.controller.get_button(i)]
                extra = (f" raw_axes=[{', '.join(raw_axes)}] "
                         f"raw_btns={raw_btns}")
            print(f"[XboxController] active={self.active_arm} "
                  f"lx={dx:+.2f} ly={dy:+.2f} rx={dyaw:+.2f} ry={dz:+.2f} "
                  f"lt={vals['lt']:.2f} rt={vals['rt']:.2f} "
                  f"btns={pressed_names}{extra}")
            self._last_debug_print = time.time()

    # -- 按钮动作 ----------------------------------------------------------
    def _request_switch(self):
        self.switch_requested = True

    def _request_gripper_toggle(self):
        # 只发 toggle 请求; 实际开合状态由 BimanualTeleop 按臂维护
        self.gripper_cmd = "toggle"

    # -- 提供给控制循环的接口 ----------------------------------------------
    def get_active_arm(self):
        return self.active_arm

    def get_target(self, arm=None):
        if arm is None:
            arm = self.active_arm
        return self.targets[arm]["pos"]

    def get_target_rot(self, arm=None):
        if arm is None:
            arm = self.active_arm
        return self.targets[arm]["rot"]

    def consume_switch(self):
        """返回并清除切换请求; 同时切换激活臂并清零平滑状态."""
        if self.switch_requested:
            self.switch_requested = False
            self.active_arm = "right" if self.active_arm == "left" else "left"
            self.reset_motion_state()
            return True
        return False




# ---------------------------------------------------------------------------
# KeyboardController — 键盘回退控制器 (双臂版)
# ---------------------------------------------------------------------------
class KeyboardController:
    """pynput 全局键盘监听, 与 XboxController 提供相同的接口.

    按键 (同 MolmoSpaces viewer_keyboard_teleop.py):
      W/S: X±  A/D: Y±  R/F: Z±   (世界系平移, 作用于当前激活臂)
      Q/E: roll±(局部X)  Z/C: pitch±(局部Y)  B/N: yaw±(局部Z)
      ←/→: 偏航± (备用)
      Space: 夹爪开合切换   Tab: 切换机械臂   H: 回 home   P: 打印   Esc: 退出
    """

    def __init__(self, home_targets, home_rot):
        from pynput import keyboard

        self.targets = {}
        for arm in ARMS:
            self.targets[arm] = {
                "pos": np.array(home_targets[arm], dtype=float).copy(),
                "rot": np.array(home_rot[arm], dtype=float).reshape(3, 3).copy(),
                "rot_accum": np.zeros(3),
            }
        self.active_arm = "left"
        self.switch_requested = False
        self.gripper_cmd = None
        self.home_requested = False
        self.print_requested = False
        self.quit_requested = False

        self.pos_sensitivity = POS_SENSITIVITY
        self.rot_sensitivity = ROT_SENSITIVITY

        self._pressed = set()
        self._prev_keys = set()
        self._listener = keyboard.Listener(
            on_press=self._on_press, on_release=self._on_release)
        self._listener.start()
        print("[KeyboardController] 未检测到手柄, 已启用键盘控制回退.")

    def _on_press(self, key):
        try:
            self._pressed.add(key.char.lower())
        except AttributeError:
            self._pressed.add(key)

    def _on_release(self, key):
        try:
            self._pressed.discard(key.char.lower())
        except AttributeError:
            self._pressed.discard(key)

    def is_connected(self):
        return True

    def reset_motion_state(self):
        pass

    def handle_input(self, arm=None):
        from pynput import keyboard as kb

        pressed = self._pressed.copy()
        t = self.targets[self.active_arm]

        # 平移 (世界系, 作用于当前激活臂)
        if 'w' in pressed: t["pos"][0] += self.pos_sensitivity
        if 's' in pressed: t["pos"][0] -= self.pos_sensitivity
        if 'd' in pressed: t["pos"][1] += self.pos_sensitivity
        if 'a' in pressed: t["pos"][1] -= self.pos_sensitivity
        if 'r' in pressed: t["pos"][2] += self.pos_sensitivity
        if 'f' in pressed: t["pos"][2] -= self.pos_sensitivity

        # 旋转 (末端局部系): Q/E roll, Z/C pitch, B/N yaw, ←/→ yaw 备用
        droll = float('e' in pressed) - float('q' in pressed)
        dpitch = float('c' in pressed) - float('z' in pressed)
        dyaw = (float('n' in pressed) - float('b' in pressed)
                + float(kb.Key.right in pressed) - float(kb.Key.left in pressed))
        new_accum = t["rot_accum"] + np.array([droll, dpitch, dyaw]) * self.rot_sensitivity
        if np.max(np.abs(new_accum)) <= np.radians(MAX_ACCUM_ROT_DEG):
            t["rot_accum"] = new_accum
            t["rot"] = t["rot"] @ rodrigues(
                np.array([droll, dpitch, dyaw]) * self.rot_sensitivity)

        # 边沿触发
        if kb.Key.space in pressed and kb.Key.space not in self._prev_keys:
            self.gripper_cmd = "toggle"
        if kb.Key.tab in pressed and kb.Key.tab not in self._prev_keys:
            self.switch_requested = True
        if 'h' in pressed and 'h' not in self._prev_keys:
            self.home_requested = True
        if 'p' in pressed and 'p' not in self._prev_keys:
            self.print_requested = True
        if kb.Key.esc in pressed:
            self.quit_requested = True

        # 边界裁剪 (基座相对)
        t["pos"][0] = float(np.clip(t["pos"][0], EE_X_MIN, EE_X_MAX))
        t["pos"][1] = float(np.clip(t["pos"][1], EE_Y_MIN, EE_Y_MAX))
        t["pos"][2] = float(np.clip(t["pos"][2], EE_Z_MIN, EE_Z_MAX))
        self._prev_keys = pressed

    # -- 接口 (与 XboxController 一致) --------------------------------------
    def get_active_arm(self):
        return self.active_arm

    def get_target(self, arm=None):
        if arm is None:
            arm = self.active_arm
        return self.targets[arm]["pos"]

    def get_target_rot(self, arm=None):
        if arm is None:
            arm = self.active_arm
        return self.targets[arm]["rot"]

    def consume_switch(self):
        if self.switch_requested:
            self.switch_requested = False
            self.active_arm = "right" if self.active_arm == "left" else "left"
            return True
        return False

    def consume_gripper_cmd(self):
        cmd, self.gripper_cmd = self.gripper_cmd, None
        return cmd

    def consume_home(self):
        req, self.home_requested = self.home_requested, False
        return req

    def consume_print(self):
        req, self.print_requested = self.print_requested, False
        return req

    def cleanup(self):
        if self._listener.is_alive():
            self._listener.stop()


# ---------------------------------------------------------------------------
# BimanualTeleop — 双臂场景构建 + 控制循环
# ---------------------------------------------------------------------------
class BimanualTeleop:
    """双臂 YAM 手柄遥操作主控.

    每帧流程 (对应 joystick_yam.py 的 step):
        1. controller.handle_input()         处理输入, 更新激活臂目标
        2. 处理 A 键切换机械臂 / B 键夹爪切换 / X 键 home
        3. 对左右臂分别 DLS 雅可比 IK 求关节角 (每臂保持各自目标)
        4. 写入 qpos / ctrl, 仿真步进
    """

    def __init__(self, xml_path: str, controller):
        self.xml_path = xml_path
        self.controller = controller

        # ---- 构建场景: 双臂 yam 模型 + 简单地面/灯光 (调试窗口) ----
        spec = MjSpec.from_file(xml_path)
        spec.worldbody.add_geom(
            name="floor", type=mjtGeom.mjGEOM_PLANE,
            size=[2, 2, 0.05], rgba=[0.30, 0.32, 0.38, 1.0])
        spec.worldbody.add_light(
            pos=[0, 0, 5], dir=[0, 0, -1],
            diffuse=[0.45, 0.45, 0.45], specular=[0.3, 0.3, 0.3])
        spec.worldbody.add_light(
            pos=[-1, -1, 2], dir=[0.3, 0.3, -1],
            diffuse=[0.25, 0.25, 0.25])

        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        m = self.model

        # ---- 每臂关节 / 执行器 / 末端 site 索引 ----
        self.arm = {}
        for arm in ARMS:
            p = arm + "_"
            jnt_ids = [mujoco.mj_name2id(m, mjtObj.mjOBJ_JOINT, f"{p}joint{i}")
                       for i in range(1, 7)]
            act_ids = [mujoco.mj_name2id(m, mjtObj.mjOBJ_ACTUATOR, f"{p}joint{i}")
                       for i in range(1, 7)]
            gripper_act = mujoco.mj_name2id(m, mjtObj.mjOBJ_ACTUATOR, f"{p}gripper")
            ee_site = mujoco.mj_name2id(m, mjtObj.mjOBJ_SITE, f"{p}grasp_site")
            # qpos/qvel 下标 (单自由度关节: adr 即下标)
            qpos_idx = np.array([m.jnt_qposadr[j] for j in jnt_ids], dtype=int)
            dof_idx = np.array([m.jnt_dofadr[j] for j in jnt_ids], dtype=int)
            self.arm[arm] = {
                "jnt_ids": jnt_ids,
                "act_ids": act_ids,
                "gripper_act": gripper_act,
                "ee_site": ee_site,
                "qpos_idx": qpos_idx,
                "dof_idx": dof_idx,
                "jnt_range": m.jnt_range[jnt_ids],       # (6,2)
                "gripper_ctrl": GRIPPER_OPEN_CTRL,       # home 时夹爪张开
            }

        # ---- home 位形 & 每臂初始末端位姿 (基座相对) ----
        self.home_qpos = HOME_QPOS.copy()
        self._apply_qpos(self.home_qpos)
        for arm in ARMS:
            self.data.ctrl[self.arm[arm]["act_ids"]] = self.home_qpos[
                self.arm[arm]["qpos_idx"]]
            self.data.ctrl[self.arm[arm]["gripper_act"]] = GRIPPER_OPEN_CTRL
        mujoco.mj_forward(m, self.data)

        self.home_ee = {}
        for arm in ARMS:
            site_pos = self.data.site_xpos[self.arm[arm]["ee_site"]].copy()
            site_rot = self.data.site_xmat[self.arm[arm]["ee_site"]].reshape(3, 3).copy()
            self.home_ee[arm] = {
                "pos": site_pos - ARM_BASE_OFFSET[arm],   # 基座相对
                "rot": site_rot,
            }

        # ---- IK 参数 ----
        self.damp = 1e-2
        self.dq_clip = 0.2
        self.ik_iters = 5

        # ---- HUD ----
        self._last_print = 0.0
        self._last_active = None

    # -- 基础操作 ----------------------------------------------------------
    def _apply_qpos(self, qpos: np.ndarray):
        """写入全部关节位置并清空速度 (运动学遥操作)."""
        self.data.qpos[:] = qpos
        self.data.qvel[:] = 0.0

    def _reset_to_home(self):
        self._apply_qpos(self.home_qpos)
        for arm in ARMS:
            self.arm[arm]["gripper_ctrl"] = GRIPPER_OPEN_CTRL
            self.data.ctrl[self.arm[arm]["gripper_act"]] = GRIPPER_OPEN_CTRL
            self.data.ctrl[self.arm[arm]["act_ids"]] = self.home_qpos[
                self.arm[arm]["qpos_idx"]]
        mujoco.mj_forward(self.model, self.data)
        # 同步控制器目标到 home
        for arm in ARMS:
            t = self.controller.targets[arm]
            t["pos"] = self.home_ee[arm]["pos"].copy()
            t["rot"] = self.home_ee[arm]["rot"].copy()
            t["rot_accum"] = np.zeros(3)
        if hasattr(self.controller, "reset_motion_state"):
            self.controller.reset_motion_state()
        if hasattr(self.controller, "gripper_open"):
            self.controller.gripper_open = True
        print("[BimanualTeleop] 已回到 home 位形.")

    # -- 逆运动学 (DLS, 单臂) ----------------------------------------------
    def _dls_ik(self, arm: str, target_pos_world: np.ndarray,
                target_rot: np.ndarray) -> np.ndarray:
        """基于末端雅可比的最小二乘阻尼 IK, 返回该臂关节角 (6,)."""
        m, d = self.model, self.data
        info = self.arm[arm]
        q = d.qpos[info["qpos_idx"]].copy()

        Jp = np.zeros((3, m.nv))
        Jr = np.zeros((3, m.nv))

        for _ in range(self.ik_iters):
            mujoco.mj_forward(m, d)  # 更新当前末端位姿
            cur_pos = d.site_xpos[info["ee_site"]].copy()
            cur_rot = d.site_xmat[info["ee_site"]].reshape(3, 3).copy()

            err_pos = target_pos_world - cur_pos
            err_rot = rotmat_to_rotvec(target_rot @ cur_rot.T)
            err = np.concatenate([err_pos, err_rot])

            if np.linalg.norm(err_pos) < 1e-4 and np.linalg.norm(err_rot) < 1e-3:
                break

            mujoco.mj_jacSite(m, d, Jp, Jr, info["ee_site"])
            J = np.vstack([Jp[:, info["dof_idx"]], Jr[:, info["dof_idx"]]])  # (6,6)

            JJT = J @ J.T
            x = np.linalg.solve(JJT + self.damp * np.eye(6), err)
            dq = np.clip(J.T @ x, -self.dq_clip, self.dq_clip)

            q = np.clip(q + dq, info["jnt_range"][:, 0], info["jnt_range"][:, 1])
            d.qpos[info["qpos_idx"]] = q

        return q

    # -- Y 键详细状态打印 --------------------------------------------------
    def _print_state_detail(self):
        """Y 键触发: 打印双臂各自的六个关节角、末端位姿和夹爪开合."""
        mujoco.mj_forward(self.model, self.data)
        active = self.controller.get_active_arm()

        print("=" * 72)
        print(f"[BimanualTeleop] Y 键状态打印 (当前激活: {active.upper()})")
        for arm in ARMS:
            info = self.arm[arm]
            q_deg = np.degrees(self.data.qpos[info["qpos_idx"]])
            ee = self.data.site_xpos[info["ee_site"]]
            R = self.data.site_xmat[info["ee_site"]].reshape(3, 3)
            rv = np.degrees(rotmat_to_rotvec(self.home_ee[arm]["rot"].T @ R))
            gripper = "OPEN" if info["gripper_ctrl"] > 0.02 else "CLOSED"

            print(f"  ---- {arm.upper()} 臂 ----")
            print("    关节角度(deg): " + ", ".join(
                f"j{i+1}={q_deg[i]:+.2f}" for i in range(6)))
            print(f"    末端位置(m): [{ee[0]:.4f}, {ee[1]:.4f}, {ee[2]:.4f}]")
            print("    末端姿态 R(世界系):")
            for row in R:
                print(f"      [{row[0]:+.4f}, {row[1]:+.4f}, {row[2]:+.4f}]")
            print(f"    相对 home 旋转(deg): rx={rv[0]:+.2f}, ry={rv[1]:+.2f}, "
                  f"rz={rv[2]:+.2f}")
            print(f"    夹爪: {gripper} (ctrl={info['gripper_ctrl']:.3f})")
        print("=" * 72)

    # -- 主循环 ------------------------------------------------------------
    def step(self):
        """单帧控制: 输入 -> 切换/夹爪/home -> 双臂 IK -> 写入 -> 步进."""
        ctrl = self.controller

        # 1. 处理输入 (作用于当前激活臂)
        ctrl.handle_input()

        # 2. 按键事件
        if ctrl.consume_switch():
            active = ctrl.get_active_arm()
            print(f"[BimanualTeleop] 切换到 {active.upper()} 臂")
        if ctrl.consume_home():
            self._reset_to_home()
        if ctrl.consume_gripper_cmd() == "toggle":
            active = ctrl.get_active_arm()
            info = self.arm[active]
            new = (GRIPPER_CLOSE_CTRL if info["gripper_ctrl"] > 0.02
                   else GRIPPER_OPEN_CTRL)
            info["gripper_ctrl"] = new
            if hasattr(ctrl, "gripper_open"):
                ctrl.gripper_open = new > 0.02
            print(f"[BimanualTeleop] {active.upper()} 臂夹爪 "
                  f"{'张开' if new > 0.02 else '闭合'}")

        # 3. 双臂分别 IK 到各自目标
        for arm in ARMS:
            rel_pos = np.asarray(ctrl.get_target(arm), dtype=float)
            rot = np.asarray(ctrl.get_target_rot(arm), dtype=float).reshape(3, 3)
            target_pos = rel_pos + ARM_BASE_OFFSET[arm]

            q = self._dls_ik(arm, target_pos, rot)
            info = self.arm[arm]
            self.data.qpos[info["qpos_idx"]] = q
            self.data.qvel[info["dof_idx"]] = 0.0
            self.data.ctrl[info["act_ids"]] = q          # PD 目标跟随
            self.data.ctrl[info["gripper_act"]] = info["gripper_ctrl"]

        # 4a. Y 键: 详细状态打印 (双臂关节角 + 末端位姿 + 夹爪)
        if ctrl.consume_print():
            self._print_state_detail()
            self._last_print = time.time()

        # 4b. 周期性 HUD (位置 + 跟踪误差 + 夹爪摘要)
        '''
        elif time.time() - self._last_print > 0.5:
            active = ctrl.get_active_arm()
            trig = ""
            if hasattr(ctrl, "get_trigger_state"):
                lt, rt = ctrl.get_trigger_state()
                trig = f" | LT:{lt:.2f} RT:{rt:.2f}"
            parts = [f"ACTIVE={active.upper()}"]
            for arm in ARMS:
                ee = self.data.site_xpos[self.arm[arm]["ee_site"]]
                tpos = np.asarray(ctrl.get_target(arm)) + ARM_BASE_OFFSET[arm]
                err = np.linalg.norm(ee - tpos) * 1000.0
                g = ("OPEN" if self.arm[arm]["gripper_ctrl"] > 0.02 else "CLOSED")
                parts.append(f"{arm.upper()}:EE=({ee[0]:.3f},{ee[1]:.3f},{ee[2]:.3f})"
                             f" err={err:.0f}mm grp={g}")
            print("[BimanualTeleop] " + " | ".join(parts) + trig)
            self._last_print = time.time()
        '''

        # 5. 仿真步进
        mujoco.mj_forward(self.model, self.data)
        mujoco.mj_step(self.model, self.data)

    def run(self, headless_steps: int = 0, fps: int = 60):
        """启动遥操作循环. headless_steps>0 时不打开窗口, 仅步进自检."""
        m, d = self.model, self.data
        spf = 1.0 / fps

        if headless_steps > 0:
            print(f"[BimanualTeleop] headless 自检模式: {headless_steps} 步")
            t0 = time.time()
            for _ in range(headless_steps):
                self.step()
                time.sleep(min(spf, 0.005))
            print(f"[BimanualTeleop] headless 自检完成, 用时 {time.time() - t0:.2f}s")
            return

        viewer = mujoco.viewer.launch_passive(m, d)
        viewer.cam.distance = 1.8
        viewer.cam.azimuth = 120
        viewer.cam.elevation = -20
        viewer.cam.lookat = np.array([0.3, 0.0, 0.3])

        print("[BimanualTeleop] 遥操作开始. A=切换机械臂, B=夹爪开合, "
              "X=home, 关闭窗口/Start 退出.")
        try:
            while viewer.is_running():
                if getattr(self.controller, "quit_requested", False):
                    break
                self.step()
                viewer.sync()
                time.sleep(spf)
        except KeyboardInterrupt:
            pass
        finally:
            viewer.close()
            print("[BimanualTeleop] 已退出.")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def resolve_xml_path(xml_path: str) -> str:
    """解析模型路径: 默认工程内路径, 找不到时尝试 MolmoSpaces 缓存路径."""
    if os.path.isfile(xml_path):
        return xml_path
    raise FileNotFoundError(
        f"找不到双臂 YAM 模型文件: {xml_path}\n"
        f"请用 --xml 指定路径, 或先运行 molmospaces 的资源下载.")


def load_home_ee_pose(xml_path: str) -> dict:
    """加载模型并返回 home 位形下左右 grasp_site 的基座相对位置与姿态."""
    spec = MjSpec.from_file(xml_path)
    model = spec.compile()
    data = mujoco.MjData(model)
    data.qpos[:] = HOME_QPOS
    mujoco.mj_forward(model, data)
    out = {}
    for arm in ARMS:
        sid = mujoco.mj_name2id(model, mjtObj.mjOBJ_SITE, f"{arm}_grasp_site")
        out[arm] = {
            "pos": data.site_xpos[sid].copy() - ARM_BASE_OFFSET[arm],
            "rot": data.site_xmat[sid].reshape(3, 3).copy(),
        }
    return out


def main():
    parser = argparse.ArgumentParser(
        description="手柄遥操作 MuJoCo 仿真双臂 YAM 机械臂")
    parser.add_argument("--xml", type=str, default=DEFAULT_XML,
                        help="双臂 YAM 模型 MJCF 路径")
    parser.add_argument("--headless", type=int, default=0, metavar="STEPS",
                        help="无窗口自检模式, 运行 STEPS 步后退出")
    parser.add_argument("--fps", type=int, default=60, help="渲染帧率")
    parser.add_argument("--debug", action="store_true",
                        help="打印手柄归一化输入/原始轴值, 用于排查映射与静止值问题")
    parser.add_argument("--calibrate", action="store_true",
                        help="交互式检测手柄 raw 轴/按键映射并保存到 mapping 文件")
    parser.add_argument("--verify", action="store_true",
                        help="启动前实时打印各逻辑轴/按键检测值, 人工确认映射")
    parser.add_argument("--mapping", type=str, default=None,
                        help="手柄映射 JSON 路径 (默认与 joystick_yam.py 共用 "
                             "joystick_mapping.json)")
    parser.add_argument("--force-raw", action="store_true",
                        help="跳过 SDL GameController 标准映射, 强制使用 raw 映射")
    parser.add_argument("--force-sdl", action="store_true",
                        help="即使存在标定映射也强制使用 SDL GameController 映射")
    args = parser.parse_args()

    xml_path = resolve_xml_path(args.xml)
    print(f"[main] 加载模型: {xml_path}")

    # 先拿到 home 末端位置/姿态, 用于初始化控制器目标 (避免启动跳变)
    home = load_home_ee_pose(xml_path)
    home_targets = {arm: home[arm]["pos"] for arm in ARMS}
    home_rot = {arm: home[arm]["rot"] for arm in ARMS}

    # 手柄优先, 未连接则键盘回退
    controller = XboxController(home_targets, home_rot,
                                mapping_path=args.mapping,
                                force_raw=args.force_raw,
                                force_sdl=args.force_sdl)

    # 标定模式: 只检测映射并保存, 不启动机械臂
    if args.calibrate:
        if not controller.is_connected():
            print("[main] 未检测到手柄, 无法标定.")
            controller.cleanup()
            return 1
        ok = controller.calibrate_mapping(save=True)
        if ok and args.verify:
            controller.verify_mapping()
        controller.cleanup()
        return 0 if ok else 1

    if not controller.is_connected():
        controller = KeyboardController(home_targets, home_rot)
    controller.debug = args.debug

    if args.verify and hasattr(controller, "verify_mapping"):
        controller.verify_mapping(duration=8.0)

    robot = BimanualTeleop(xml_path, controller)
    try:
        robot.run(headless_steps=args.headless, fps=args.fps)
    finally:
        controller.cleanup()


if __name__ == "__main__":
    sys.exit(main())
