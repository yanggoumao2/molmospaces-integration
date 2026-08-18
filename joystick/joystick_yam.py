#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
joystick_yam.py — 北通/Xbox 手柄遥操作 MuJoCo 仿真 YAM 机械臂
================================================================
参照 joystick_so100.py 的结构编写：  https://github.com/LitchiCheng/mujoco-learning
    手柄输入 -> 末端目标位姿(增量累积) -> DLS 雅可比 IK -> 写入关节 -> 仿真步进

YAM 机械臂: I2RT Robotics 6-DOF 机械臂 + 耦合平行夹爪
  - 模型: model/yam/yam.xml (来自 MolmoSpaces, mesh 资源已拷贝到本工程)
  - 关节: joint1..joint6 (对应 qpos[0:6])
  - 夹爪: 单执行器 "gripper", ctrl 0.0=闭合 / 0.041=张开
          (right_finger 由 equality 约束与 left_finger 联动)

手柄按键映射 (Xbox 兼容手柄, SDL2 轴序):
  左摇杆 X (axis 0) ──> 末端世界 X 平移
  左摇杆 Y (axis 1) ──> 末端世界 Y 平移
  右摇杆 Y (axis 4) ──> 末端世界 Z 平移
  右摇杆 X (axis 3) ──> 末端偏航 yaw  (绕末端局部 Z 轴)
  LT(axis 2) / RT(axis 5) ──> 末端横滚 roll (绕末端局部 X 轴, 扳机差值)
  LB(4) / RB(5) (按住) ──> 末端俯仰 pitch (绕末端局部 Y 轴)
  B 键(1) ──> 夹爪开合切换 (张开/闭合都用 B)
  X 键(2) ──> 回到 home 位姿   Y 键(3) ──> 打印末端位姿
  Start(7) ──> 退出

  夹爪朝向采用 MolmoSpaces viewer_keyboard_teleop.py 的方式:
  在末端局部坐标系中施加三维旋转增量 R_target = R_current @ R_delta,
  因此夹爪朝向可以在空间中任意方向改变, 不限于平面角度.

未检测到手柄时自动回退为键盘控制 (与 viewer_keyboard_teleop.py 按键一致):
  W/S: X±   A/D: Y±   R/F: Z±
  Q/E: roll±(局部X)   Z/C: pitch±(局部Y)   B/N: yaw±(局部Z)  ←/→: 偏航±
  J: 夹爪开合切换   H: 回 home   P: 打印位姿   Esc: 退出

环境依赖：
    + pygame
  
运行方式:
  conda activate mujoco-learning            # 带有mujoco和pygame的环境均可
  python joystick_yam.py --calibrate        # 交互式检测手柄轴/按键映射并保存
  python joystick_yam.py --verify           # 启动前实时验证映射
  python joystick_yam.py

  python joystick_yam.py --debug            # 实时打印归一化输入/原始轴值
  python joystick_yam.py --headless 500     # 无窗口自检模式(调试用)
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pygame

import mujoco
import mujoco.viewer
from mujoco import MjSpec, mjtGeom, mjtObj

# ---------------------------------------------------------------------------
# 常量: 模型路径 / 初始位形 / 末端工作空间
# ---------------------------------------------------------------------------
DEFAULT_XML = "model/yam/yam.xml"

# YAM 初始关节角 
HOME_QPOS = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

# 末端位置目标的工作空间边界 (世界系, 软约束)
# 夹爪限位，通过调整对应轴的限位可以修改机械臂可到达的区域
EE_X_MIN, EE_X_MAX = 0.15, 0.50
EE_Y_MIN, EE_Y_MAX = -0.22, 0.22
EE_Z_MIN, EE_Z_MAX = 0.04, 0.35 # ! 现在0.04是测试过后紧贴桌面的值，不要再改了

# 手柄灵敏度 (满量程时: 平移约 0.06 m/s, 旋转约 10 deg/s @60fps)
POS_SENSITIVITY = 0.001    # 每单位摇杆位移对应的末端平移 (m)
ROT_SENSITIVITY = 0.003    # 每单位摇杆位移/按键对应的末端旋转 (rad)
INPUT_SMOOTHING = 0.25     # 输入速度一阶平滑系数 (越小起步越缓)
DEADZONE = 0.1             # 摇杆死区
MAX_ACCUM_ROT_DEG = 175.0  # 累计旋转安全上限 (度/轴), 防止异常输入导致机械臂缠绕

GRIPPER_OPEN_CTRL = 0.041  # 夹爪张开 ctrl
GRIPPER_CLOSE_CTRL = 0.0   # 夹爪闭合 ctrl

# 手柄映射标定文件 (--calibrate 生成, 正常运行时自动加载)
DEFAULT_MAPPING_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "joystick_mapping.json")
TRIGGER_DEADZONE = 0.02        # 扳机死区 (SDL 模式下扳机为 0..1)
STICK_REST_TOLERANCE = 0.35    # 摇杆静止值绝对值超过此值视为映射异常
AXIS_DETECT_THRESHOLD = 0.15   # 标定时判定轴变化的阈值


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
    """轴角旋转向量 -> 旋转矩阵 (Rodrigues 公式).

    与 viewer_keyboard_teleop.py 中 _ik_move 的局部旋转构造方式一致:
    R_delta = I + sin(θ)K + (1-cos(θ))K²
    """
    angle = np.linalg.norm(axis_angle)
    if angle < 1e-10:
        return np.eye(3)
    axis = axis_angle / angle
    K = np.array([[0.0, -axis[2], axis[1]],
                  [axis[2], 0.0, -axis[0]],
                  [-axis[1], axis[0], 0.0]])
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


# ---------------------------------------------------------------------------
# XboxController — 手柄控制器 (pygame 读取 /dev/input/jsX)
# ---------------------------------------------------------------------------
class XboxController:
    """北通/Xbox 手柄控制器类, 负责处理所有手柄输入.

    输入优先级:
      1. joystick_mapping.json 标定映射: 设备名匹配时优先使用, 保证你标定
         的轴序/方向/按钮编号一定生效 (--force-sdl 可改回 SDL 内置映射);
      2. SDL GameController 标准映射: 没有标定文件时, SDL 认出手柄则按
         物理键位读取, 扳机恒为 0..1(松开=0);
      3. 默认 raw Xbox 布局: 两者都没有时使用, 并给出 --calibrate 提示.

    运行 `python joystick_yam.py --calibrate` 可交互式检测手柄原始轴/按键
    编号并保存映射, 解决不同手柄/不同连接模式轴序不同的问题.
    raw 模式下会连续跟踪摇杆静止值和扳机松开值, 并在读取时减去, 避免
    "手柄没动但机械臂自己漂移"。
    """

    # 默认 raw 映射 (SDL 标准 Xbox 布局), 仅作为未标定时的初值
    RAW_AXES = {
        "lx": {"index": 0, "gain": 1.0, "kind": "stick"},
        "ly": {"index": 1, "gain": 1.0, "kind": "stick"},
        "lt": {"index": 2, "gain": 1.0, "kind": "trigger"},
        "rx": {"index": 3, "gain": 1.0, "kind": "stick"},
        "ry": {"index": 4, "gain": 1.0, "kind": "stick"},
        "rt": {"index": 5, "gain": 1.0, "kind": "trigger"},
    }
    # B 键同时负责夹爪张开/闭合 (单键切换); A 键在双臂版 joystick_bi_yam.py 中用于切换机械臂
    RAW_BUTTONS = {"gripper": 1, "home": 2, "print": 3,
                   "pitch_neg": 4, "pitch_pos": 5, "quit": 7}

    AXIS_NAMES = ("lx", "ly", "rx", "ry", "lt", "rt")
    LOGICAL_BUTTONS = ("gripper", "home", "print",
                       "pitch_neg", "pitch_pos", "quit")

    # 标定时每个逻辑轴对应的操作提示 (方向均为该逻辑轴的正方向)
    AXIS_PROMPTS = {
        "lx": "把左摇杆向右推到底并保持住",
        "ly": "把左摇杆向上推到底并保持住",
        "rx": "把右摇杆向右推到底并保持住",
        "ry": "把右摇杆向上推到底并保持住",
        "lt": "把左扳机 LT 按到底并保持住",
        "rt": "把右扳机 RT 按到底并保持住",
    }
    BUTTON_PROMPTS = {
        "gripper": "按一下 B 键(夹爪开合切换)",
        "home": "按一下 X 键(回 home)",
        "print": "按一下 Y 键(打印位姿)",
        "pitch_neg": "按住 LB 左肩键",
        "pitch_pos": "按住 RB 右肩键",
        "quit": "按一下 Start 键(退出)",
    }

    def __init__(self, home_pos: np.ndarray, home_rot: np.ndarray,
                 mapping_path: str = None, force_raw: bool = False,
                 force_sdl: bool = False):
        # 目标位置初始化为 home 处末端位置 (避免启动时跳变)
        self.x, self.y, self.z = home_pos
        # 目标姿态初始化为 home 处末端姿态 (3x3 旋转矩阵)
        self.rot = np.array(home_rot, dtype=float).reshape(3, 3).copy()
        self._home_rot = self.rot.copy()
        # 相对 home 姿态的累计旋转 (各分量有符号, 用于安全钳制防缠绕)
        self.rot_accum = np.zeros(3)

        # 位置/偏航限制
        self.x_min, self.x_max = EE_X_MIN, EE_X_MAX
        self.y_min, self.y_max = EE_Y_MIN, EE_Y_MAX
        self.z_min, self.z_max = EE_Z_MIN, EE_Z_MAX

        # 控制灵敏度与死区
        self.pos_sensitivity = POS_SENSITIVITY
        self.rot_sensitivity = ROT_SENSITIVITY
        self.deadzone = DEADZONE

        # 夹爪状态: None=保持, "open"=张开, "close"=闭合
        self.gripper_cmd = None
        self.gripper_open = False

        # 边沿触发标志
        self.home_requested = False
        self.print_requested = False
        self.quit_requested = False

        # 按钮按下状态 (避免重复触发)
        self._btn_states = {}

        # raw 映射状态
        self.force_raw = force_raw
        self.force_sdl = force_sdl
        self.mapping_path = mapping_path or DEFAULT_MAPPING_FILE
        self.axes = {k: dict(v) for k, v in self.RAW_AXES.items()}
        self.buttons = dict(self.RAW_BUTTONS)
        self._mapping_source = "default"      # default / file / calibrated
        self._rest = {}                       # 各 raw 轴的当前静止值估计
        self._gc_rest = {}                    # GC 模式各逻辑轴静止值估计
        self._axis_noise = {}                 # 静止采样时各轴噪声(std)
        self._axis_deadzone = {}              # 每个逻辑轴的实际死区
        self._shared_trigger_axes = set()     # 一根轴同时表示 LT/RT 的轴

        # GameController 标准映射句柄 (None = 使用 raw 映射)
        self.gc = None
        self.debug = False
        self._last_debug_print = 0.0
        self._last_clamp_print = 0.0

        # 输入速度一阶平滑状态 (让起步/停止都是渐进的, 不会猛冲)
        self.input_smoothing = INPUT_SMOOTHING
        self._pos_vel = np.zeros(3)
        self._rot_vel = np.zeros(3)

        # 初始化手柄
        self.controller = self.init_controller()
        if self.is_connected():
            # 先尝试加载已标定映射: 只要设备名匹配, 标定映射优先于 SDL 内置映射
            self._load_mapping()
            use_sdl = (not self.force_raw
                       and (self.force_sdl
                            or self._mapping_source == "default"))
            if use_sdl:
                self._init_gamecontroller()
            else:
                self.gc = None

            if self.gc is None:
                if self._mapping_source != "default":
                    print("[XboxController] 已加载标定映射, 将优先使用 raw 标定 "
                          "映射. 如需强制 SDL 内置映射, 请加 --force-sdl.")
                self._update_mapping_state()
                # 多帧采样静止值: 修正 SDL 首事件前读数不准、摇杆中心偏移
                self._capture_rest(duration=0.8)
                self._update_axis_deadzones()
                self._warn_mapping_sanity()
            else:
                self._update_axis_deadzones()
                # GameController 模式同样采样各逻辑轴静止值, 消除硬件中心偏移
                self._capture_gc_rest(duration=0.8)

    def init_controller(self):
        """初始化手柄 (通过 pygame 访问 js 设备)."""
        os.environ.setdefault("SDL_JOYSTICK_DEVICE", "/dev/input/js0")

        pygame.init()
        pygame.joystick.init()
        pygame.event.pump()

        if pygame.joystick.get_count() == 0:
            print("[XboxController] 未检测到任何游戏杆设备")
            return None

        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        pygame.event.pump()
        print(f"[XboxController] 检测到手柄: {joystick.get_name()}")
        print(f"[XboxController] 轴数量: {joystick.get_numaxes()}, "
              f"按钮数量: {joystick.get_numbuttons()}")
        return joystick

    def is_connected(self):
        return self.controller is not None

    # -- SDL GameController 标准映射 ---------------------------------------
    def _init_gamecontroller(self):
        """尝试启用 SDL GameController 标准映射.

        注意: pygame._sdl2.controller.is_controller() 只接受整数设备索引,
        传 Joystick 对象会抛异常; 这里取 get_id() 再判断, 并用
        Controller.from_joystick() 构造.
        """
        try:
            from pygame._sdl2 import controller as sdl2_controller
            sdl2_controller.init()
            if sdl2_controller.get_count() > 0:
                # is_controller() 接受的是设备索引 (0..get_count()-1),
                # 不是 SDL instance id
                device_index = self.controller.get_id()
                if sdl2_controller.is_controller(device_index):
                    try:
                        self.gc = sdl2_controller.Controller.from_joystick(
                            self.controller)
                    except AttributeError:
                        self.gc = sdl2_controller.Controller(self.controller)
        except Exception as exc:  # noqa: BLE001
            print(f"[XboxController] GameController 初始化失败: {exc}")
            self.gc = None

        if self.gc is not None:
            print("[XboxController] 已启用 SDL GameController 标准映射 "
                  "(扳机 0..1, 按键按物理键位)")
        else:
            print("[XboxController] GameController 映射不可用, 使用 raw 轴序. "
                  "如果轴/按键不对或机械臂自己漂移, 请运行 "
                  "`python joystick_yam.py --calibrate` 检测映射.")

    # -- 映射文件 -----------------------------------------------------------
    def _load_mapping(self):
        """加载 joystick_mapping.json (仅当设备名匹配时生效)."""
        if not os.path.isfile(self.mapping_path):
            return

        try:
            with open(self.mapping_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:  # noqa: BLE001
            print(f"[XboxController] 读取映射文件失败: {exc}")
            return

        if data.get("version") != 1:
            print("[XboxController] 映射文件版本不兼容, 已忽略")
            return

        saved_name = data.get("device_name", "")
        current_name = self.controller.get_name()
        if saved_name and saved_name != current_name:
            print(f"[XboxController] 映射文件属于 '{saved_name}', "
                  f"当前手柄为 '{current_name}', 已忽略映射文件. "
                  f"请重新运行 --calibrate.")
            return

        nax = self.controller.get_numaxes()
        nbtn = self.controller.get_numbuttons()
        for name, cfg in (data.get("axes") or {}).items():
            if name not in self.RAW_AXES:
                continue
            idx = int(cfg.get("index", -1))
            source = cfg.get("source", "axis")
            limit = nbtn if source == "button" else nax
            if not (0 <= idx < limit):
                continue
            gain = float(cfg.get("gain", 1.0))
            if abs(gain) < 1e-6:
                gain = 1.0
            kind = cfg.get("kind", "trigger" if name in ("lt", "rt") else "stick")
            rest = float(cfg.get("rest", 0.0))
            span = float(cfg.get("span", 1.0))
            if kind == "trigger" and source == "axis":
                # 兼容旧标定文件: 扳机没按到底时 span 过小, 会过度灵敏
                span = max(span, self._standard_trigger_span(rest))
            self.axes[name] = {
                "index": idx,
                "gain": gain,
                "kind": kind,
                "rest": rest,
                "span": span,
                "source": source,
            }

        saved_buttons = dict(data.get("buttons") or {})
        # 兼容旧标定文件: 以前 open=A / close=B, 现在统一为 gripper=B
        if "gripper" not in saved_buttons:
            for legacy in ("close", "open"):
                if legacy in saved_buttons:
                    saved_buttons["gripper"] = saved_buttons[legacy]
                    break
        for name, idx in saved_buttons.items():
            if name not in self.RAW_BUTTONS:
                continue
            idx = int(idx)
            if 0 <= idx < nbtn:
                self.buttons[name] = idx

        self._mapping_source = os.path.abspath(self.mapping_path)
        print(f"[XboxController] 已加载标定映射: {self._mapping_source}")

    def _update_mapping_state(self):
        """计算共享扳机轴等派生状态."""
        self._shared_trigger_axes = set()
        seen_sign = {}
        for cfg in self.axes.values():
            if cfg.get("kind") != "trigger":
                continue
            idx = int(cfg["index"])
            sign = 1 if float(cfg.get("gain", 1.0)) >= 0 else -1
            if idx in seen_sign and seen_sign[idx] != sign:
                self._shared_trigger_axes.add(idx)
            seen_sign[idx] = sign

    @staticmethod
    def _standard_trigger_span(rest: float) -> float:
        """按静止值推断扳机标准满量程.

        rest≈-1/+1: 典型 -1..+1 量程, 满量程 2
        rest≈0:     典型  0..+1 量程, 满量程 1
        标定时若扳机没按到底, 用标准量程兜底, 避免半按就被放大约一倍.
        """
        return 2.0 if abs(float(rest)) > 0.5 else 1.0

    def _save_mapping(self):
        """把标定结果保存为 JSON, 下次启动自动加载."""
        guid = ""
        try:
            guid = self.controller.get_guid()
        except Exception:
            pass

        data = {
            "version": 1,
            "device_name": self.controller.get_name(),
            "device_guid": guid,
            "num_axes": self.controller.get_numaxes(),
            "num_buttons": self.controller.get_numbuttons(),
            "axes": {name: {
                            k: (int(v) if k == "index" else
                                (float(v) if k in ("gain", "rest", "span") else v))
                            for k, v in cfg.items()}
                     for name, cfg in self.axes.items()},
            "buttons": {name: int(idx) for name, idx in self.buttons.items()},
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        try:
            with open(self.mapping_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"[XboxController] 映射已保存到: {self.mapping_path}")
        except Exception as exc:  # noqa: BLE001
            print(f"[XboxController] 映射保存失败: {exc}")

    # -- 静止值/死区 --------------------------------------------------------
    def _capture_rest(self, duration: float = 0.8, announce: bool = True):
        """多帧采样各 raw 轴的静止读数, 用于漂移抑制."""
        nax = self.controller.get_numaxes()
        if nax <= 0:
            return

        if announce:
            print(f"[XboxController] 正在记录各轴静止位置 ({duration:.1f} 秒), "
                  f"请松开所有摇杆和扳机...")
            time.sleep(0.2)

        n = max(10, int(duration / 0.02))
        rows = []
        for _ in range(n):
            pygame.event.pump()
            rows.append([float(self.controller.get_axis(i)) for i in range(nax)])
            time.sleep(0.02)

        rows = np.asarray(rows, dtype=float)
        med = np.median(rows, axis=0)
        noise = np.std(rows, axis=0)
        for i in range(nax):
            self._rest[i] = float(med[i])
            self._axis_noise[i] = float(noise[i])

        desc = ", ".join(f"axis{i}={med[i]:+.3f}" for i in range(nax))
        print(f"[XboxController] 轴静止读数: {desc}")

    def _capture_gc_rest(self, duration: float = 0.8):
        """GameController 模式下采样各逻辑轴静止值."""
        if self.gc is None:
            return
        print(f"[XboxController] 正在记录 GameController 各轴静止位置 "
              f"({duration:.1f} 秒), 请松开所有摇杆和扳机...")
        time.sleep(0.2)

        consts = {
            "lx": "CONTROLLER_AXIS_LEFTX", "ly": "CONTROLLER_AXIS_LEFTY",
            "rx": "CONTROLLER_AXIS_RIGHTX", "ry": "CONTROLLER_AXIS_RIGHTY",
            "lt": "CONTROLLER_AXIS_TRIGGERLEFT",
            "rt": "CONTROLLER_AXIS_TRIGGERRIGHT",
        }
        n = max(10, int(duration / 0.02))
        rows = {name: [] for name in consts}
        for _ in range(n):
            pygame.event.pump()
            for name, const_name in consts.items():
                try:
                    rows[name].append(float(self.gc.get_axis(
                        getattr(pygame, const_name))))
                except Exception:
                    rows[name].append(0.0)
            time.sleep(0.02)

        for name in consts:
            vals = np.asarray(rows[name], dtype=float)
            rest = float(np.median(vals))
            if name in ("lt", "rt"):
                rest = max(0.0, min(1.0, rest))
            self._gc_rest[name] = rest

        desc = ", ".join(f"{n}={self._gc_rest[n]:+.3f}" for n in consts)
        print(f"[XboxController] GameController 轴静止读数: {desc}")

    def _update_axis_deadzones(self):
        """根据静止采样噪声设置每个逻辑轴的实际死区."""
        if self.gc is not None:
            for name in self.AXIS_NAMES:
                self._axis_deadzone[name] = (
                    TRIGGER_DEADZONE if name in ("lt", "rt") else self.deadzone)
            return

        for name, cfg in self.axes.items():
            idx = int(cfg["index"])
            noise = float(self._axis_noise.get(idx, 0.0))
            if cfg.get("kind") == "trigger":
                dz = max(TRIGGER_DEADZONE, min(0.10, 3.0 * noise))
            else:
                dz = max(self.deadzone, min(0.30, 4.0 * noise))
            self._axis_deadzone[name] = float(dz)

    def _warn_mapping_sanity(self):
        """默认 raw 映射与手柄实际读数明显不符时给出警告.

        北通等手柄在不同模式下轴序差别很大, 默认 Xbox 轴序经常把扳机/摇杆
        接错; 静止读数异常就是最明显的特征.
        """
        if self.gc is not None or self._mapping_source != "default":
            return

        problems = []
        for name in ("lx", "ly", "rx", "ry"):
            cfg = self.axes[name]
            rest = float(self._rest.get(cfg["index"], cfg.get("rest", 0.0)))
            if abs(rest) > STICK_REST_TOLERANCE:
                problems.append(f"{name}(axis{cfg['index']}) 静止值={rest:+.2f}")
        for name in ("lt", "rt"):
            cfg = self.axes[name]
            if cfg.get("source", "axis") == "button":
                continue
            rest = float(self._rest.get(cfg["index"], cfg.get("rest", 0.0)))
            if 0.2 < abs(rest) < 0.8:
                problems.append(f"{name}(axis{cfg['index']}) 静止值={rest:+.2f}")

        if problems:
            print("[XboxController] 警告: 默认 Xbox 轴序与当前手柄读数不匹配 "
                  f"({' / '.join(problems)}).")
            print("[XboxController] 这通常就是机械臂自行漂移的原因, 请先运行:")
            print("    python joystick_yam.py --calibrate")

    # -- 轴/按键读取 --------------------------------------------------------
    def _read_raw_axis(self, name: str) -> float:
        """按标定/默认 raw 映射读取并去漂移.

        摇杆: val = (raw - rest) * gain, 再套死区; 静止时 rest 缓慢跟踪,
              小范围零点漂移不会输出.
        扳机: 输出归一化到 0..1. 只有标定过的映射才做运行最小值/最大值
              跟踪; 未标定时不跟踪, 避免"把摇杆误认成扳机后 rest 锁到 -1,
              松开后反而输出满值"的漂移陷阱.
        """
        if not self.is_connected():
            return 0.0

        cfg = self.axes.get(name)
        if cfg is None:
            return 0.0
        idx = int(cfg["index"])

        # 某些手柄的扳机是数字按钮 (SDL DB 里常见 lefttrigger:b8)
        if cfg.get("source", "axis") == "button":
            if not (0 <= idx < self.controller.get_numbuttons()):
                return 0.0
            try:
                return 1.0 if self.controller.get_button(idx) else 0.0
            except Exception:
                return 0.0

        if not (0 <= idx < self.controller.get_numaxes()):
            return 0.0

        try:
            raw = float(self.controller.get_axis(idx))
        except Exception:
            return 0.0

        gain = float(cfg.get("gain", 1.0) or 1.0)
        kind = cfg.get("kind", "stick")
        rest = float(self._rest.get(idx, float(cfg.get("rest", 0.0))))

        if kind == "trigger":
            if idx not in self._shared_trigger_axes and \
                    self._mapping_source != "default":
                if gain > 0.0:      # 按下使读数增大: 松开值=运行最小值
                    if raw < rest:
                        rest = raw
                else:               # 按下使读数减小: 松开值=运行最大值
                    if raw > rest:
                        rest = raw
                self._rest[idx] = rest

            span = float(cfg.get("span", 0.0) or 0.0)
            if span <= 0.0:
                span = 2.0 if rest < -0.5 else 1.0
            span = max(span, 0.25)
            out = max(0.0, min(1.0, (raw - rest) * gain / span))
            dz = float(self._axis_deadzone.get(name, TRIGGER_DEADZONE))
            return out if out > dz else 0.0

        # 摇杆
        delta = (raw - rest) * gain
        dz = float(self._axis_deadzone.get(name, self.deadzone))
        if abs(delta) <= dz:
            # 只在死区内缓慢修正静止值, 消除温度/电位器引起的慢漂移
            rest += 0.02 * (raw - rest)
            self._rest[idx] = rest
            return 0.0
        # 限幅到标准 ±1, 防止错误标定/异常量程导致目标一步跳变
        return float(np.clip(delta, -1.0, 1.0))

    def _axis(self, name: str) -> float:
        """按逻辑名读取轴值 (带死区/去漂移)."""
        if self.gc is not None:
            const = getattr(pygame, {
                "lx": "CONTROLLER_AXIS_LEFTX", "ly": "CONTROLLER_AXIS_LEFTY",
                "rx": "CONTROLLER_AXIS_RIGHTX", "ry": "CONTROLLER_AXIS_RIGHTY",
                "lt": "CONTROLLER_AXIS_TRIGGERLEFT",
                "rt": "CONTROLLER_AXIS_TRIGGERRIGHT",
            }[name])
            try:
                val = float(self.gc.get_axis(const))
            except Exception:
                return 0.0
            if name in ("lt", "rt"):
                val = max(0.0, min(1.0, val))
                dz = float(self._axis_deadzone.get(name, TRIGGER_DEADZONE))
                return val if val > dz else 0.0
            # 摇杆: 减去启动时采到的中心偏移, 并在死区内缓慢跟踪零点
            rest = float(self._gc_rest.get(name, 0.0))
            delta = val - rest
            dz = float(self._axis_deadzone.get(name, self.deadzone))
            if abs(delta) <= dz:
                rest += 0.02 * (val - rest)
                self._gc_rest[name] = rest
                return 0.0
            return float(np.clip(delta, -1.0, 1.0))
        return self._read_raw_axis(name)

    def _button(self, name: str) -> bool:
        """按逻辑名读取按钮状态 (SDL 标准映射优先)."""
        if self.gc is not None:
            const = getattr(pygame, {
                "gripper": "CONTROLLER_BUTTON_B",
                "home": "CONTROLLER_BUTTON_X", "print": "CONTROLLER_BUTTON_Y",
                "pitch_neg": "CONTROLLER_BUTTON_LEFTSHOULDER",
                "pitch_pos": "CONTROLLER_BUTTON_RIGHTSHOULDER",
                "quit": "CONTROLLER_BUTTON_START",
            }[name])
            try:
                return bool(self.gc.get_button(const))
            except Exception:
                return False

        idx = int(self.buttons.get(name, self.RAW_BUTTONS[name]))
        if idx < 0 or idx >= self.controller.get_numbuttons():
            return False
        try:
            return bool(self.controller.get_button(idx))
        except Exception:
            return False

    # -- 映射标定 -----------------------------------------------------------
    def _sample(self, duration: float = 1.0, interval: float = 0.02):
        """采样一段时间, 返回 (轴读数矩阵, 按钮读数矩阵)."""
        nax = self.controller.get_numaxes()
        nbtn = self.controller.get_numbuttons()
        axes_rows, btn_rows = [], []
        end = time.time() + duration
        while time.time() < end:
            pygame.event.pump()
            axes_rows.append([float(self.controller.get_axis(i))
                              for i in range(nax)])
            if nbtn:
                btn_rows.append([float(self.controller.get_button(i))
                                 for i in range(nbtn)])
            time.sleep(interval)
        ax = np.asarray(axes_rows, dtype=float).reshape(-1, nax)
        bt = np.asarray(btn_rows, dtype=float).reshape(-1, nbtn) if btn_rows \
            else np.zeros((1, 0))
        return ax, bt

    def _detect_axis(self, name: str, kind: str) -> dict:
        """让用户操作一个逻辑轴, 检测它对应哪根 raw 轴及方向.

        提示方向为该逻辑轴的正方向; 若读数反而减小, gain 记为 -1,
        这样无论手柄轴是否反相, 逻辑值符号都正确.
        """
        nax = self.controller.get_numaxes()
        base = {i: float(self._rest.get(i, 0.0)) for i in range(nax)}
        base_arr = np.array([base[i] for i in range(nax)], dtype=float)

        for attempt in range(3):
            print(f"\n[{name}] 请{self.AXIS_PROMPTS[name]}")
            print("    准备好后按 Enter 开始采样, 采样期间请保持操作...")
            try:
                input()
            except (EOFError, KeyboardInterrupt):
                time.sleep(1.0)

            ax, bt = self._sample(duration=1.0)
            med = np.median(ax, axis=0)
            dev = med - base_arr
            best = int(np.argmax(np.abs(dev)))
            delta = float(dev[best])

            if abs(delta) < AXIS_DETECT_THRESHOLD:
                # 扳机可能是数字按钮而不是轴: 检查按钮采样
                if kind == "trigger" and bt.size > 0:
                    counts = bt.sum(axis=0)
                    threshold = max(1.0, 0.35 * bt.shape[0])
                    pressed = [int(i) for i, c in enumerate(counts)
                               if c >= threshold]
                    if pressed:
                        best_btn = max(pressed, key=lambda i: counts[i])
                        cfg = {"index": best_btn, "gain": 1.0, "kind": "trigger",
                               "rest": 0.0, "span": 1.0, "source": "button"}
                        print(f"    未检测到轴变化, 但检测到数字扳机按钮 "
                              f"raw button {best_btn}.")
                        return cfg
                print(f"    未检测到明显轴变化 (最大 |Δ|={max(abs(dev)):.2f}), "
                      f"请重试.")
                continue

            gain = 1.0 if delta > 0.0 else -1.0
            if kind == "trigger":
                std_span = self._standard_trigger_span(base[best])
                span = float(max(abs(delta), 0.25, std_span))
                if abs(delta) < std_span * 0.9:
                    print(f"    提示: 检测到扳机行程只有 |Δ|={abs(delta):.2f}, "
                          f"疑似标定时未按到底. 已按标准满量程 {std_span:.1f} "
                          f"处理, 避免扳机过度灵敏; 若仍不对请重新标定并务必按到底.")
            else:
                span = 2.0
            cfg = {"index": best, "gain": gain, "kind": kind,
                   "rest": base[best], "span": span}
            print(f"    检测到: raw axis {best}, 静止值={base[best]:+.3f}, "
                  f"变化={delta:+.3f}, gain={gain:+.0f}, "
                  f"span={span:.3f}")
            return cfg

        print(f"    三次未检测到 {name}, 保留默认映射")
        return dict(self.RAW_AXES[name])

    def _detect_button(self, name: str) -> int:
        """让用户按一个逻辑按键, 检测它对应哪个 raw 按钮编号."""
        for attempt in range(3):
            print(f"\n[{name}] 请{self.BUTTON_PROMPTS[name]}")
            print("    准备好后按 Enter 开始采样...")
            try:
                input()
            except (EOFError, KeyboardInterrupt):
                time.sleep(1.0)

            _, bt = self._sample(duration=0.9)
            if bt.size == 0:
                continue
            counts = bt.sum(axis=0)
            threshold = max(1.0, 0.35 * bt.shape[0])
            pressed = [int(i) for i, c in enumerate(counts) if c >= threshold]
            if not pressed:
                print("    未检测到按钮按下, 请重试.")
                continue
            best = max(pressed, key=lambda i: counts[i])
            print(f"    检测到: raw button {best} (采样命中 {int(counts[best])} 次)")
            return best

        print(f"    三次未检测到 {name}, 保留默认映射")
        return int(self.RAW_BUTTONS.get(name, -1))

    def _print_mapping(self):
        print("\n======================== 标定结果 ========================")
        for name in self.AXIS_NAMES:
            cfg = self.axes[name]
            src = cfg.get("source", "axis")
            where = f"raw {src} {cfg['index']}"
            print(f"  {name:>3s}: {where}, "
                  f"gain={cfg['gain']:+.0f}, rest={cfg['rest']:+.3f}, "
                  f"kind={cfg['kind']}, span={cfg.get('span', 0.0):.3f}")
        for name in self.LOGICAL_BUTTONS:
            print(f"  {name:>10s}: raw button {self.buttons[name]}")
        print("=========================================================")

    def _check_axis_conflicts(self) -> bool:
        """检查是否有多个逻辑轴错误地映射到同一根 raw 轴."""
        stick_axes = set()
        trigger_axes = set()
        owners = {}
        problems = []
        for name, cfg in self.axes.items():
            if cfg.get("source", "axis") == "button":
                continue
            idx = int(cfg["index"])
            if name in ("lt", "rt"):
                trigger_axes.add(idx)
                continue
            if idx in owners:
                problems.append(f"{owners[idx]} 与 {name} 都映射到 axis {idx}")
            owners[idx] = name
            stick_axes.add(idx)

        # 摇杆和扳机撞到同一根轴 (LT/RT 共用一根轴除外)
        for idx in sorted(stick_axes & trigger_axes):
            problems.append(f"axis {idx} 同时被摇杆和扳机使用")

        if problems:
            print("[calibrate] 警告: " + "; ".join(problems) + ". "
                  "这会导致一个摇杆同时驱动多个自由度, 请重新标定.")
            return False
        return True

    def _check_released(self, duration: float = 1.2) -> bool:
        """标定后让用户松开所有输入, 确认静止时逻辑值全部为零."""
        print("\n[calibrate] 松手检查: 请完全松开所有摇杆/扳机, "
              f"保持 {duration:.0f} 秒...")
        time.sleep(1.0)
        n = int(duration / 0.02)
        worst = {name: 0.0 for name in self.AXIS_NAMES}
        for _ in range(n):
            pygame.event.pump()
            for name in self.AXIS_NAMES:
                worst[name] = max(worst[name], abs(self._read_raw_axis(name)))
            time.sleep(0.02)

        bad = {n: v for n, v in worst.items() if v > 1e-6}
        if bad:
            desc = ", ".join(f"{n}={v:.2f}" for n, v in bad.items())
            print("[calibrate] 警告: 松手后仍有非零输入 (" + desc + ").")
            print("[calibrate] 这会造成机械臂自行漂移, 建议重新标定, "
                  "并在采集静止值时务必松手.")
            return False
        print("[calibrate] 松手检查通过: 所有轴静止输出均为 0.")
        return True

    def reset_motion_state(self):
        """清除输入平滑状态 (回 home 时调用, 避免残余速度继续推动目标)."""
        self._pos_vel = np.zeros(3)
        self._rot_vel = np.zeros(3)

    def calibrate_mapping(self, save: bool = True) -> bool:
        """交互式检测手柄 raw 轴/按键映射并保存 (辅助遥操).

        步骤: 1) 采集静止值  2) 逐个轴检测编号/方向  3) 逐个按键检测编号
              4) 保存 joystick_mapping.json
        """
        if not self.is_connected():
            print("[calibrate] 未检测到手柄, 无法标定.")
            return False

        if self.gc is not None:
            print("[calibrate] 当前 SDL 已识别该手柄为标准 GameController, "
                  "正常运行时无需 raw 映射; 仍会标定并保存 raw 映射备用.")

        print(f"\n[calibrate] 开始标定: {self.controller.get_name()}")
        self._capture_rest(duration=1.0)
        self._update_axis_deadzones()

        new_axes = {}
        for name in self.AXIS_NAMES:
            kind = "trigger" if name in ("lt", "rt") else "stick"
            new_axes[name] = self._detect_axis(name, kind)
        self.axes = new_axes

        new_buttons = {}
        for name in self.LOGICAL_BUTTONS:
            new_buttons[name] = self._detect_button(name)
        self.buttons = new_buttons

        self._mapping_source = "calibrated"
        self._update_mapping_state()
        self._update_axis_deadzones()
        self._print_mapping()
        self._check_axis_conflicts()
        self._check_released()

        # 检查是否有两个逻辑按键撞到同一个 raw 编号
        used = {}
        for name, idx in self.buttons.items():
            if idx in used:
                print(f"[calibrate] 警告: '{used[idx]}' 与 '{name}' 都映射到 "
                      f"button {idx}, 请检查.")
            used[idx] = name

        if save:
            self._save_mapping()

        print("\n[calibrate] 完成. 可运行 `python joystick_yam.py --debug` "
              "或 `--verify` 验证映射.")
        return True

    def verify_mapping(self, duration: float = 8.0):
        """实时打印各逻辑轴/按键的检测值, 用于人工确认映射是否正确."""
        if not self.is_connected():
            print("[verify] 未检测到手柄.")
            return
        print(f"[verify] 实时验证 {duration:.0f} 秒, 请依次摇动摇杆、按扳机/肩键/A/B/X/Y/Start...")
        end = time.time() + duration
        while time.time() < end:
            pygame.event.pump()
            vals = {name: self._axis(name) for name in self.AXIS_NAMES}
            btns = [name for name in self.LOGICAL_BUTTONS if self._button(name)]
            raw = ""
            if self.gc is None:
                raw_vals = [f"{i}:{self.controller.get_axis(i):+.2f}"
                            for i in range(self.controller.get_numaxes())]
                raw = " raw_axes=[" + ", ".join(raw_vals) + "]"
            print(f"[verify] lx={vals['lx']:+.2f} ly={vals['ly']:+.2f} "
                  f"rx={vals['rx']:+.2f} ry={vals['ry']:+.2f} "
                  f"lt={vals['lt']:.2f} rt={vals['rt']:.2f} "
                  f"btns={btns}{raw}")
            time.sleep(0.2)

    # -- 旋转增量 ------------------------------------------------------------
    def _apply_rot_delta(self, delta_rot: np.ndarray):
        """在当前目标局部系中施加旋转增量, 并做累计旋转安全钳制.

        防止手柄异常输入导致目标姿态无限累积, 机械臂关节持续缠绕成"麻花".
        """
        new_accum = self.rot_accum + delta_rot
        if np.max(np.abs(new_accum)) > np.radians(MAX_ACCUM_ROT_DEG):
            now = time.time()
            if now - self._last_clamp_print > 1.0:  # 告警限频, 避免刷屏
                print("[XboxController] 累计旋转达到安全上限, 已忽略本次旋转输入")
                self._last_clamp_print = now
            return
        self.rot_accum = new_accum
        self.rot = self.rot @ rodrigues(delta_rot)

    def handle_input(self, arm=None):
        """处理手柄输入并更新目标位姿. 每帧调用一次."""
        if not self.is_connected():
            return

        pygame.event.pump()

        # 同一帧内只读一次, 避免 debug 打印重复调用造成静止值被重复更新
        vals = {name: self._axis(name) for name in self.AXIS_NAMES}
        btns = {name: self._button(name) for name in self.LOGICAL_BUTTONS}

        # 左摇杆: 世界 X/Y 平移; 右摇杆: Z 平移 + 局部 Z 旋转(偏航)
        dx = vals["lx"]
        dy = vals["ly"]
        dz = -vals["ry"]
        dyaw = vals["rx"]
        # 扳机: LT 负 / RT 正 -> 局部 X 旋转(横滚)
        droll = vals["rt"] - vals["lt"]
        # 肩键: LB 负 / RB 正, 按住连续 -> 局部 Y 旋转(俯仰)
        dpitch = float(btns["pitch_pos"]) - float(btns["pitch_neg"])

        # 一阶平滑: 起步/停止渐进, 避免满杆输入让目标一步猛冲
        desired_pos = np.array([dx, dy, dz], dtype=float)
        desired_rot = np.array([droll, dpitch, dyaw], dtype=float)
        s = float(self.input_smoothing)
        self._pos_vel += s * (desired_pos - self._pos_vel)
        self._rot_vel += s * (desired_rot - self._rot_vel)
        step_pos = np.clip(self._pos_vel, -1.0, 1.0) * self.pos_sensitivity
        step_rot = np.clip(self._rot_vel, -1.0, 1.0) * self.rot_sensitivity

        # 增量累积位置目标
        self.x = float(np.clip(self.x + step_pos[0],
                               self.x_min, self.x_max))
        self.y = float(np.clip(self.y + step_pos[1],
                               self.y_min, self.y_max))
        self.z = float(np.clip(self.z + step_pos[2],
                               self.z_min, self.z_max))

        # 增量累积姿态目标: 在当前目标局部系中旋转 (同 viewer_keyboard_teleop)
        self._apply_rot_delta(step_rot)

        # 按钮 (边沿触发)
        for name, action in [
            ("gripper", self._request_gripper_toggle),  # B: 夹爪开合切换
            ("home", self._request_home),               # X
            ("print", self._request_print),             # Y
            ("quit", self._request_quit),               # Start
        ]:
            pressed = btns[name]
            if pressed and not self._btn_states.get(name, False):
                action()
            self._btn_states[name] = pressed

        # 调试: 打印归一化后的输入值, 便于排查手柄映射/静止值问题
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
            print(f"[XboxController] lx={dx:+.2f} ly={dy:+.2f} rx={dyaw:+.2f} "
                  f"ry={dz:+.2f} lt={vals['lt']:.2f} rt={vals['rt']:.2f} "
                  f"btns={pressed_names}{extra}")
            self._last_debug_print = time.time()

    def _request_gripper_toggle(self):
        self.gripper_cmd = "toggle"
        self.gripper_open = not self.gripper_open

    def _request_home(self):
        self.home_requested = True

    def _request_print(self):
        self.print_requested = True

    def _request_quit(self):
        self.quit_requested = True

    # -- 提供给控制循环的接口 ----------------------------------------------
    def get_target(self):
        """返回 (x, y, z) 末端位置目标."""
        return self.x, self.y, self.z

    def get_target_rot(self):
        """返回 3x3 末端姿态目标矩阵."""
        return self.rot

    def get_trigger_state(self):
        """返回 (LT, RT) 扳机按下量 [0,1], 用于 HUD 显示/排查."""
        if not self.is_connected():
            return 0.0, 0.0
        return self._axis("lt"), self._axis("rt")

    def consume_gripper_cmd(self):
        """返回并清除夹爪指令."""
        cmd, self.gripper_cmd = self.gripper_cmd, None
        return cmd

    def consume_home(self):
        """返回并清除 home 复位请求."""
        req, self.home_requested = self.home_requested, False
        return req

    def consume_print(self):
        req, self.print_requested = self.print_requested, False
        return req

    def cleanup(self):
        pygame.quit()




# ---------------------------------------------------------------------------
# KeyboardController — 键盘回退控制器 (未检测到手柄时使用)
# ---------------------------------------------------------------------------
class KeyboardController:
    """pynput 全局键盘监听, 与 XboxController 提供相同的接口.

    按键与 MolmoSpaces viewer_keyboard_teleop.py 一致:
      W/S: X±  A/D: Y±  R/F: Z±            (世界系平移)
      Q/E: roll±(局部X)  Z/C: pitch±(局部Y)  B/N: yaw±(局部Z)  (夹爪朝向)
      ←/→: 偏航± (备用)
      J: 夹爪开合切换  H: 回 home  P: 打印位姿  Esc: 退出
    """

    def __init__(self, home_pos: np.ndarray, home_rot: np.ndarray):
        from pynput import keyboard

        self.x, self.y, self.z = home_pos
        self.rot = np.array(home_rot, dtype=float).reshape(3, 3).copy()
        self._home_rot = self.rot.copy()
        self.rot_accum = np.zeros(3)

        self.pos_sensitivity = POS_SENSITIVITY
        self.rot_sensitivity = ROT_SENSITIVITY

        self.gripper_cmd = None
        self.gripper_open = False
        self.home_requested = False
        self.print_requested = False
        self.quit_requested = False

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

    def handle_input(self, arm=None):
        from pynput import keyboard as kb

        pressed = self._pressed.copy()

        # 平移 (世界系)
        if 'w' in pressed: self.x += self.pos_sensitivity
        if 's' in pressed: self.x -= self.pos_sensitivity
        if 'd' in pressed: self.y += self.pos_sensitivity
        if 'a' in pressed: self.y -= self.pos_sensitivity
        if 'r' in pressed: self.z += self.pos_sensitivity
        if 'f' in pressed: self.z -= self.pos_sensitivity

        # 旋转 (末端局部系, 同 viewer_keyboard_teleop):
        #   Q/E: roll±(局部X)  Z/C: pitch±(局部Y)  B/N: yaw±(局部Z)
        #   ←/→: 偏航± (备用)
        droll = float('e' in pressed) - float('q' in pressed)
        dpitch = float('c' in pressed) - float('z' in pressed)
        dyaw = (float('n' in pressed) - float('b' in pressed)
                + float(kb.Key.right in pressed) - float(kb.Key.left in pressed))
        # 与 XboxController 相同的累计旋转安全钳制
        new_accum = self.rot_accum + np.array([droll, dpitch, dyaw]) * self.rot_sensitivity
        if np.max(np.abs(new_accum)) <= np.radians(MAX_ACCUM_ROT_DEG):
            self.rot_accum = new_accum
            self.rot = self.rot @ rodrigues(
                np.array([droll, dpitch, dyaw]) * self.rot_sensitivity)

        # 夹爪切换 (边沿)
        if 'j' in pressed and 'j' not in self._prev_keys:
            self.gripper_open = not self.gripper_open
            self.gripper_cmd = "open" if self.gripper_open else "close"
        if 'h' in pressed and 'h' not in self._prev_keys:
            self.home_requested = True
        if 'p' in pressed and 'p' not in self._prev_keys:
            self.print_requested = True
        if kb.Key.esc in pressed:
            self.quit_requested = True

        # 边界裁剪
        self.x = float(np.clip(self.x, EE_X_MIN, EE_X_MAX))
        self.y = float(np.clip(self.y, EE_Y_MIN, EE_Y_MAX))
        self.z = float(np.clip(self.z, EE_Z_MIN, EE_Z_MAX))
        self._prev_keys = pressed

    def get_target(self):
        return self.x, self.y, self.z

    def get_target_rot(self):
        return self.rot

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
# YamTeleop — 场景构建 + 控制循环
# ---------------------------------------------------------------------------
class YamTeleop:
    """YAM 机械臂手柄遥操作主控.

    每帧流程 (对应 joystick_so100.py 的 runFunc):
        1. controller.handle_input()          处理输入, 更新末端目标
        2. 读取目标 (x, y, z) + 姿态矩阵 R
        3. 构建目标位姿, DLS 雅可比 IK 求关节角
        4. 写入 qpos / ctrl, 仿真步进
    """

    def __init__(self, xml_path: str, controller):
        self.xml_path = xml_path
        self.controller = controller

        # ---- 构建场景: yam 模型 + 简单地面/灯光 (调试窗口) ----
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

        # ---- 关节 / 执行器 / 末端 site 索引 ----
        self.arm_jnt_ids = [mujoco.mj_name2id(m, mjtObj.mjOBJ_JOINT, f"joint{i}")
                            for i in range(1, 7)]
        self.arm_act_ids = [mujoco.mj_name2id(m, mjtObj.mjOBJ_ACTUATOR, f"joint{i}")
                            for i in range(1, 7)]
        self.gripper_act_id = mujoco.mj_name2id(m, mjtObj.mjOBJ_ACTUATOR, "gripper")
        self.ee_site_id = mujoco.mj_name2id(m, mjtObj.mjOBJ_SITE, "grasp_site")

        self.arm_qpos_idx = np.array(self.arm_jnt_ids, dtype=int)   # qpos 下标
        self.arm_dof_idx = np.array(self.arm_jnt_ids, dtype=int)    # qvel 下标 (joint 顺序一致)

        # ---- 关节范围 ----
        self.jnt_range = m.jnt_range[self.arm_jnt_ids]  # (6, 2)

        # ---- home 位形 & 初始末端位姿 ----
        self.home_qpos = HOME_QPOS.copy()
        self._apply_qpos(self.home_qpos)
        self.data.ctrl[self.arm_act_ids] = self.home_qpos[:6]
        self.data.ctrl[self.gripper_act_id] = GRIPPER_CLOSE_CTRL
        mujoco.mj_forward(m, self.data)

        self.home_ee_pos = self.data.site_xpos[self.ee_site_id].copy()
        self.home_ee_rot = self.data.site_xmat[self.ee_site_id].reshape(3, 3).copy()

        # ---- IK 参数 ----
        self.damp = 1e-2        # DLS 阻尼
        self.dq_clip = 0.2      # 单帧关节增量上限 (rad)
        self.ik_iters = 5       # 每帧 IK 迭代次数

        # ---- 夹爪 ----
        self.gripper_ctrl = GRIPPER_CLOSE_CTRL

        # ---- HUD ----
        self._last_print = 0.0

    # -- 基础操作 ----------------------------------------------------------

    def _apply_qpos(self, qpos: np.ndarray):
        """写入关节位置并清空对应速度 (运动学遥操作)."""
        self.data.qpos[:] = qpos
        self.data.qvel[:] = 0.0

    def _reset_to_home(self):
        self._apply_qpos(self.home_qpos)
        self.gripper_ctrl = GRIPPER_CLOSE_CTRL
        self.data.ctrl[self.gripper_act_id] = self.gripper_ctrl
        mujoco.mj_forward(self.model, self.data)
        # 同步控制器内部目标到 home 末端位姿
        self.controller.x, self.controller.y, self.controller.z = self.home_ee_pos
        self.controller.rot = self.home_ee_rot.copy()
        if hasattr(self.controller, "rot_accum"):
            self.controller.rot_accum = np.zeros(3)
        if hasattr(self.controller, "reset_motion_state"):
            self.controller.reset_motion_state()
        self.controller.gripper_open = False
        print("[YamTeleop] 已回到 home 位形.")

    # -- 逆运动学 (DLS) ----------------------------------------------------

    def _dls_ik(self, target_pos: np.ndarray, target_rot: np.ndarray):
        """基于末端雅可比的最小二乘阻尼 IK, 返回关节角 (6,).

        与 MolmoSpaces yam_control_demo 中 KeyboardTeleop._ik_move 同思路:
        dq = Jᵀ (J Jᵀ + λI)⁻¹ err, 迭代至收敛.
        """
        m, d = self.model, self.data
        q = self.data.qpos[self.arm_qpos_idx].copy()

        Jp = np.zeros((3, m.nv))
        Jr = np.zeros((3, m.nv))

        for _ in range(self.ik_iters):
            mujoco.mj_forward(m, d)  # 更新当前末端位姿
            cur_pos = d.site_xpos[self.ee_site_id].copy()
            cur_rot = d.site_xmat[self.ee_site_id].reshape(3, 3).copy()

            err_pos = target_pos - cur_pos
            err_rot = rotmat_to_rotvec(target_rot @ cur_rot.T)  # 世界系轴角
            err = np.concatenate([err_pos, err_rot])

            if np.linalg.norm(err_pos) < 1e-4 and np.linalg.norm(err_rot) < 1e-3:
                break

            mujoco.mj_jacSite(m, d, Jp, Jr, self.ee_site_id)
            J = np.vstack([Jp[:, self.arm_dof_idx], Jr[:, self.arm_dof_idx]])  # (6,6)

            JJT = J @ J.T
            x = np.linalg.solve(JJT + self.damp * np.eye(6), err)
            dq = np.clip(J.T @ x, -self.dq_clip, self.dq_clip)

            q = np.clip(q + dq, self.jnt_range[:, 0], self.jnt_range[:, 1])
            d.qpos[self.arm_qpos_idx] = q

        return q

    # -- Y 键详细状态打印 --------------------------------------------------
    def _print_state_detail(self):
        """Y 键触发: 打印六个关节角、末端位姿和夹爪开合."""
        mujoco.mj_forward(self.model, self.data)
        q_deg = np.degrees(self.data.qpos[self.arm_qpos_idx])
        ee = self.data.site_xpos[self.ee_site_id]
        R = self.data.site_xmat[self.ee_site_id].reshape(3, 3)
        rv = np.degrees(rotmat_to_rotvec(self.home_ee_rot.T @ R))
        gripper = "OPEN" if self.gripper_ctrl > 0.02 else "CLOSED"

        print("=" * 72)
        print("[YamTeleop] Y 键状态打印")
        print("  关节角度(deg): " + ", ".join(
            f"j{i+1}={q_deg[i]:+.2f}" for i in range(6)))
        print(f"  末端位置(m): [{ee[0]:.4f}, {ee[1]:.4f}, {ee[2]:.4f}]")
        print("  末端姿态 R(世界系):")
        for row in R:
            print(f"    [{row[0]:+.4f}, {row[1]:+.4f}, {row[2]:+.4f}]")
        print(f"  相对 home 旋转(deg): rx={rv[0]:+.2f}, ry={rv[1]:+.2f}, "
              f"rz={rv[2]:+.2f}")
        print(f"  夹爪: {gripper} (ctrl={self.gripper_ctrl:.3f})")
        print("=" * 72)

    # -- 主循环 ------------------------------------------------------------
    def step(self):
        """单帧控制: 输入 -> 目标 -> IK -> 写入 -> 步进."""
        ctrl = self.controller

        # 1. 处理输入
        ctrl.handle_input()

        # 2. 按键事件
        if ctrl.consume_home():
            self._reset_to_home()
        cmd = ctrl.consume_gripper_cmd()
        if cmd == "toggle":
            self.gripper_ctrl = (GRIPPER_OPEN_CTRL
                                 if self.gripper_ctrl <= 0.02
                                 else GRIPPER_CLOSE_CTRL)
            if hasattr(ctrl, "gripper_open"):
                ctrl.gripper_open = self.gripper_ctrl > 0.02
        elif cmd == "open":
            self.gripper_ctrl = GRIPPER_OPEN_CTRL
        elif cmd == "close":
            self.gripper_ctrl = GRIPPER_CLOSE_CTRL

        # 3. 目标位姿: 位置 (世界系) + 姿态 (控制器内按末端局部系累积)
        tx, ty, tz = ctrl.get_target()
        target_pos = np.array([tx, ty, tz])
        target_rot = np.asarray(ctrl.get_target_rot(), dtype=float).reshape(3, 3)

        # 4. IK 并写入
        q = self._dls_ik(target_pos, target_rot)
        self.data.qpos[self.arm_qpos_idx] = q
        self.data.qvel[self.arm_dof_idx] = 0.0
        self.data.ctrl[self.arm_act_ids] = q               # PD 目标跟随
        self.data.ctrl[self.gripper_act_id] = self.gripper_ctrl

        # 5a. Y 键: 详细状态打印 (关节角 + 末端位姿 + 夹爪)
        if ctrl.consume_print():
            self._print_state_detail()
            self._last_print = time.time()

        # 5b. 周期性 HUD (旋转量显示为相对 home 姿态的轴角, 并给出实际跟踪误差)
        elif time.time() - self._last_print > 0.5:
            ee = self.data.site_xpos[self.ee_site_id]
            R_cur = self.data.site_xmat[self.ee_site_id].reshape(3, 3)
            err_pos = np.linalg.norm(ee - target_pos) * 1000.0
            err_rot = np.degrees(np.linalg.norm(
                rotmat_to_rotvec(target_rot @ R_cur.T)))
            rv = rotmat_to_rotvec(self.home_ee_rot.T @ target_rot)
            trig = ""
            if hasattr(ctrl, "get_trigger_state"):
                lt, rt = ctrl.get_trigger_state()
                trig = f" | LT:{lt:.2f} RT:{rt:.2f}"
            '''
            print(f"[YamTeleop] EE pos: [{ee[0]:.3f}, {ee[1]:.3f}, {ee[2]:.3f}] "
                  f"| target: [{tx:.3f}, {ty:.3f}, {tz:.3f}] "
                  f"| rot: [{np.degrees(rv[0]):.1f}, {np.degrees(rv[1]):.1f}, "
                  f"{np.degrees(rv[2]):.1f}]deg "
                  f"| err: {err_pos:.1f}mm/{err_rot:.1f}deg"
                  f"{trig} "
                  f"| gripper: {'OPEN' if self.gripper_ctrl > 0.02 else 'CLOSED'}")
            '''
            self._last_print = time.time()

        # 6. 仿真步进
        mujoco.mj_forward(self.model, self.data)
        mujoco.mj_step(self.model, self.data)

    def run(self, headless_steps: int = 0, fps: int = 60):
        """启动遥操作循环. headless_steps>0 时不打开窗口, 仅步进自检."""
        m, d = self.model, self.data
        spf = 1.0 / fps

        if headless_steps > 0:
            print(f"[YamTeleop] headless 自检模式: {headless_steps} 步")
            t0 = time.time()
            for i in range(headless_steps):
                self.step()
                time.sleep(min(spf, 0.005))
            print(f"[YamTeleop] headless 自检完成, 用时 {time.time() - t0:.2f}s")
            return

        viewer = mujoco.viewer.launch_passive(m, d)
        viewer.cam.distance = 1.5
        viewer.cam.azimuth = 120
        viewer.cam.elevation = -20
        viewer.cam.lookat = np.array([0.35, 0.0, 0.4])

        print("[YamTeleop] 遥操作开始. 关闭窗口 / 按 Esc(Start) 退出.")
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
            print("[YamTeleop] 已退出.")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def resolve_xml_path(xml_path: str) -> str:
    """解析模型路径: 默认工程内路径, 找不到时尝试 MolmoSpaces 缓存路径."""
    if os.path.isfile(xml_path):
        return xml_path
    raise FileNotFoundError(
        f"找不到 YAM 模型文件: {xml_path}\n"
        f"请用 --xml 指定路径, 或先运行 molmospaces 的资源下载.")


def load_home_ee_pose(xml_path: str) -> tuple[np.ndarray, np.ndarray]:
    """加载模型并返回 home 位形下 grasp_site 的世界坐标与姿态矩阵."""
    spec = MjSpec.from_file(xml_path)
    model = spec.compile()
    data = mujoco.MjData(model)
    data.qpos[:] = HOME_QPOS
    mujoco.mj_forward(model, data)
    sid = mujoco.mj_name2id(model, mjtObj.mjOBJ_SITE, "grasp_site")
    pos = data.site_xpos[sid].copy()
    rot = data.site_xmat[sid].reshape(3, 3).copy()
    return pos, rot


def main():
    parser = argparse.ArgumentParser(description="手柄遥操作 MuJoCo 仿真 YAM 机械臂")
    parser.add_argument("--xml", type=str, default=DEFAULT_XML,
                        help="YAM 模型 MJCF 路径")
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
                        help="手柄映射 JSON 路径 (默认 joystick_mapping.json)")
    parser.add_argument("--force-raw", action="store_true",
                        help="跳过 SDL GameController 标准映射, 强制使用 raw 映射")
    parser.add_argument("--force-sdl", action="store_true",
                        help="即使存在标定映射也强制使用 SDL GameController 映射")
    args = parser.parse_args()

    xml_path = resolve_xml_path(args.xml)
    print(f"[main] 加载模型: {xml_path}")

    # 先拿到 home 末端位置/姿态, 用于初始化控制器目标 (避免启动跳变)
    home_pos, home_rot = load_home_ee_pose(xml_path)

    # 手柄优先, 未连接则键盘回退
    controller = XboxController(home_pos, home_rot,
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
        controller = KeyboardController(home_pos, home_rot)
    controller.debug = args.debug

    if args.verify and hasattr(controller, "verify_mapping"):
        controller.verify_mapping(duration=8.0)

    robot = YamTeleop(xml_path, controller)
    try:
        robot.run(headless_steps=args.headless, fps=args.fps)
    finally:
        controller.cleanup()


if __name__ == "__main__":
    sys.exit(main())
