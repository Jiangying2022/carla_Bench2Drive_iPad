#!/usr/bin/env python3
"""
PAD 端到端闭环控制 + 交通流 + 行人 + 可视化
参考: manual_control_with_traffic.py

功能:
- 连接 CARLA，使用当前地图或指定地图
- 生成交通流车辆和行人
- 生成 ego vehicle，规划全局路线
- PAD 端到端推理 + PID 控车
- CARLA debug 可视化: 全局路线(绿) + 模型预测轨迹(红) + 终点(黄)
- Spectator 跟随 ego vehicle
"""
import argparse
import collections
import collections.abc
import cv2
import glob
import logging
import math
import numpy as np
import os
import pygame
import random
import sys
import time
from datetime import datetime

if not hasattr(collections, 'Iterable'):
    collections.Iterable = collections.abc.Iterable

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
LOG = logging.getLogger(__name__)

REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
BENCH2DRIVE_ROOT = os.path.join(REPO_ROOT, 'Bench2Drive')
BENCH2DRIVE_ZOO_ROOT = os.path.join(REPO_ROOT, 'Bench2DriveZoo')
NUPLAN_DEVKIT_ROOT = os.path.join(REPO_ROOT, 'nuplan-devkit')


def _append_sys_path(path: str) -> None:
    if path not in sys.path:
        sys.path.append(path)


def _setup_python_paths() -> str:
    carla_root = os.environ.get('CARLA_ROOT')
    if not carla_root:
        raise RuntimeError('CARLA_ROOT is not set')

    carla_root = carla_root.rstrip('/')

    _append_sys_path(REPO_ROOT)
    _append_sys_path(BENCH2DRIVE_ZOO_ROOT)
    _append_sys_path(NUPLAN_DEVKIT_ROOT)
    _append_sys_path(BENCH2DRIVE_ROOT)
    _append_sys_path(os.path.join(BENCH2DRIVE_ROOT, 'leaderboard'))
    _append_sys_path(os.path.join(BENCH2DRIVE_ROOT, 'scenario_runner'))
    _append_sys_path(os.path.join(carla_root, 'PythonAPI'))
    _append_sys_path(os.path.join(carla_root, 'PythonAPI', 'carla'))

    carla_egg_paths = sorted(glob.glob(os.path.join(carla_root, 'PythonAPI', 'carla', 'dist', 'carla-0.9.16-*.egg')))
    if not carla_egg_paths:
        carla_egg_paths = sorted(glob.glob(os.path.join(carla_root, 'PythonAPI', 'carla', 'dist', 'carla-*.egg')))
    carla_whl_paths = sorted(glob.glob(os.path.join(carla_root, 'PythonAPI', 'carla', 'dist', 'carla-0.9.16-*.whl')))
    if not carla_whl_paths:
        carla_whl_paths = sorted(glob.glob(os.path.join(carla_root, 'PythonAPI', 'carla', 'dist', 'carla-*.whl')))
    carla_pkg_paths = carla_egg_paths if carla_egg_paths else carla_whl_paths
    if carla_pkg_paths:
        _append_sys_path(carla_pkg_paths[-1])

    os.environ.setdefault('Bench2Drive_ROOT', BENCH2DRIVE_ROOT)
    os.environ.setdefault('TEAM_AGENT', os.path.join(BENCH2DRIVE_ROOT, 'leaderboard', 'pad_team_code', 'pad_b2d_agent.py'))
    os.environ.setdefault('IS_BENCH2DRIVE', 'True')

    return carla_root


CARLA_ROOT = _setup_python_paths()

import carla
from leaderboard.autoagents.agent_wrapper import AgentWrapper
from leaderboard.utils.route_parser import RouteParser
from pad_team_code.pad_b2d_agent import padAgent
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.timer import GameTime
from srunner.tools.route_manipulation import _get_latlon_ref, _location_to_gps, interpolate_trajectory


DEFAULT_CHECKPOINT = '/home/pnc/Downloads/epoch=19-step=8780.ckpt'
DEFAULT_CONFIG = os.path.join(BENCH2DRIVE_ROOT, 'leaderboard', 'pad_team_code', 'pad_config.py')
DEFAULT_ROUTES = os.path.join(BENCH2DRIVE_ROOT, 'leaderboard', 'data', 'bench2drive220.xml')
DEFAULT_CARLA_PORT = int(os.environ.get('CARLA_PORT', '2000'))
ROUTE_COLOR = carla.Color(0, 255, 0)
TRAJECTORY_COLOR = carla.Color(255, 64, 64)
DESTINATION_COLOR = carla.Color(255, 255, 0)
START_COLOR = carla.Color(0, 128, 255)
NEAR_NODE_COLOR = carla.Color(200, 0, 255)


def _heading_deg_between_locations(start: carla.Location, end: carla.Location):
    dx = end.x - start.x
    dy = end.y - start.y
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return None
    return math.degrees(math.atan2(dy, dx))


def _normalize_angle_deg(angle_deg: float) -> float:
    return (angle_deg + 180.0) % 360.0 - 180.0


def _local_command_angle_deg(local_command_xy):
    if not local_command_xy or len(local_command_xy) < 2:
        return None
    x = float(local_command_xy[0])
    y = float(local_command_xy[1])
    if abs(x) < 1e-6 and abs(y) < 1e-6:
        return None
    return math.degrees(math.atan2(x, y))


def _trajectory_path_length(predicted_traj):
    if not predicted_traj:
        return 0.0
    length = 0.0
    prev_x = 0.0
    prev_y = 0.0
    for point in predicted_traj:
        x = float(point[0])
        y = float(point[1])
        length += math.hypot(x - prev_x, y - prev_y)
        prev_x = x
        prev_y = y
    return length


def _trajectory_terminal_heading_deg(predicted_traj):
    if not predicted_traj:
        return None
    if len(predicted_traj) >= 2:
        start = predicted_traj[-2]
        end = predicted_traj[-1]
        dx = float(end[0]) - float(start[0])
        dy = float(end[1]) - float(start[1])
        if abs(dx) >= 1e-6 or abs(dy) >= 1e-6:
            return math.degrees(math.atan2(dx, dy))
    end = predicted_traj[-1]
    x = float(end[0])
    y = float(end[1])
    if abs(x) < 1e-6 and abs(y) < 1e-6:
        return None
    return math.degrees(math.atan2(x, y))


def _format_sequence(values, precision=3):
    if values is None:
        return 'None'
    formatted = []
    for value in values:
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            formatted.append(f'({float(value[0]):.{precision}f}, {float(value[1]):.{precision}f})')
        else:
            formatted.append(f'{float(value):.{precision}f}')
    return '[' + ', '.join(formatted) + ']'


# ==================== 交通流辅助函数 ====================

def get_actor_blueprints(world, filter_str, generation):
    bps = world.get_blueprint_library().filter(filter_str)
    if generation.lower() == 'all':
        return bps
    if len(bps) == 1:
        return bps
    try:
        int_gen = int(generation)
        if int_gen in [1, 2, 3]:
            bps = [x for x in bps if int(x.get_attribute('generation')) == int_gen]
        return bps
    except Exception:
        return []


def spawn_traffic_vehicles(client, world, args, tm_port, exclude_location=None):
    """生成自动驾驶交通流车辆，返回车辆 actor id 列表"""
    vehicles_list = []
    blueprints = get_actor_blueprints(world, args.filterv, args.generationv)
    if not blueprints:
        LOG.warning('没有符合筛选条件的车辆蓝图')
        return vehicles_list

    if args.safe:
        blueprints = [x for x in blueprints if x.get_attribute('base_type') == 'car']
    blueprints = sorted(blueprints, key=lambda bp: bp.id)

    spawn_points = world.get_map().get_spawn_points()
    # 排除 ego 占用的 spawn point
    if exclude_location is not None:
        spawn_points = [sp for sp in spawn_points
                        if sp.location.distance(exclude_location) > 2.0]
    num_vehicles = min(args.number_of_vehicles, len(spawn_points))
    random.shuffle(spawn_points)

    SpawnActor = carla.command.SpawnActor
    SetAutopilot = carla.command.SetAutopilot
    FutureActor = carla.command.FutureActor

    batch = []
    for n, transform in enumerate(spawn_points):
        if n >= num_vehicles:
            break
        bp = random.choice(blueprints)
        if bp.has_attribute('color'):
            bp.set_attribute('color', random.choice(bp.get_attribute('color').recommended_values))
        if bp.has_attribute('driver_id'):
            bp.set_attribute('driver_id', random.choice(bp.get_attribute('driver_id').recommended_values))
        bp.set_attribute('role_name', 'autopilot')
        batch.append(SpawnActor(bp, transform).then(SetAutopilot(FutureActor, True, tm_port)))

    for response in client.apply_batch_sync(batch, True):
        if response.error:
            LOG.error(response.error)
        else:
            vehicles_list.append(response.actor_id)

    if args.car_lights_on and vehicles_list:
        tm = client.get_trafficmanager(tm_port)
        for actor in world.get_actors(vehicles_list):
            tm.update_vehicle_lights(actor, True)

    LOG.info(f'生成了 {len(vehicles_list)} 辆交通流车辆')
    return vehicles_list


def spawn_walkers(client, world, args):
    """生成行人和行人控制器，返回 (walkers_list, all_ids)"""
    walkers_list = []
    all_ids = []
    if args.number_of_walkers <= 0:
        return walkers_list, all_ids

    blueprints = get_actor_blueprints(world, args.filterw, args.generationw)
    if not blueprints:
        LOG.warning('没有符合筛选条件的行人蓝图')
        return walkers_list, all_ids

    if args.seedw:
        world.set_pedestrians_seed(args.seedw)
        random.seed(args.seedw)

    spawn_points = []
    for _ in range(args.number_of_walkers):
        sp = carla.Transform()
        loc = world.get_random_location_from_navigation()
        if loc is not None:
            sp.location = loc
            sp.location.z += 2.0
            spawn_points.append(sp)

    SpawnActor = carla.command.SpawnActor

    # 生成行人 actor
    batch = []
    walker_speed = []
    for sp in spawn_points:
        bp = random.choice(blueprints)
        if bp.has_attribute('is_invincible'):
            bp.set_attribute('is_invincible', 'false')
        if bp.has_attribute('speed'):
            if random.random() > args.percentage_pedestrians_running:
                walker_speed.append(bp.get_attribute('speed').recommended_values[1])
            else:
                walker_speed.append(bp.get_attribute('speed').recommended_values[2])
        else:
            walker_speed.append(0.0)
        batch.append(SpawnActor(bp, sp))

    results = client.apply_batch_sync(batch, True)
    walker_speed2 = []
    for i, result in enumerate(results):
        if result.error:
            LOG.error(result.error)
        else:
            walkers_list.append({'id': result.actor_id})
            walker_speed2.append(walker_speed[i])
    walker_speed = walker_speed2

    # 生成行人控制器
    batch = []
    walker_controller_bp = world.get_blueprint_library().find('controller.ai.walker')
    for walker in walkers_list:
        batch.append(SpawnActor(walker_controller_bp, carla.Transform(), walker['id']))
    results = client.apply_batch_sync(batch, True)
    for i, result in enumerate(results):
        if result.error:
            LOG.error(result.error)
        else:
            walkers_list[i]['con'] = result.actor_id

    for w in walkers_list:
        all_ids.append(w.get('con', 0))
        all_ids.append(w['id'])

    # tick 一帧确保控制器可用
    world.tick()

    # 启动控制器
    all_actors = world.get_actors(all_ids)
    world.set_pedestrians_cross_factor(args.percentage_pedestrians_crossing)
    for i in range(0, len(all_ids), 2):
        controller = all_actors[i]
        if controller is not None:
            controller.start()
            controller.go_to_location(world.get_random_location_from_navigation())
            controller.set_max_speed(float(walker_speed[i // 2]))

    LOG.info(f'生成了 {len(walkers_list)} 个行人')
    return walkers_list, all_ids


# ==================== 闭环控制主类 ====================

class ClosedLoopRunner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.client = None
        self.world = None
        self.traffic_manager = None
        self.spectator = None
        self.hero = None
        self.agent = None
        self.wrapper = None
        self.original_settings = None
        self.route_world_transforms = []
        self.destination_transform = None
        # 交通流
        self.traffic_vehicle_ids = []
        self.walkers_list = []
        self.walker_all_ids = []
        self.route_config = None
        self.precomputed_gps_route = None
        self.precomputed_route = None
        self.selected_spawn_index = None
        self.hud_window_name = 'PAD Closed Loop HUD'
        self.hud_enabled = not args.disable_hud
        self.hud_surface = None
        self.hud_font = None
        self.route_preview_index = 0
        self.record_camera = None
        self.record_latest_frame = None
        self.video_writer = None
        self.video_output_path = None

    def _load_route_config(self):
        if not self.args.routes or not self.args.route_id:
            return None
        route_configs = RouteParser.parse_routes_file(self.args.routes, self.args.route_id)
        if not route_configs:
            raise RuntimeError(f'No route found in {self.args.routes} for route id {self.args.route_id}')
        return route_configs[0]

    def _setup_hud(self) -> None:
        if not self.hud_enabled:
            return
        try:
            pygame.init()
            pygame.font.init()
            self.hud_surface = pygame.display.set_mode((2120, 900))
            pygame.display.set_caption(self.hud_window_name)
            self.hud_font = pygame.font.SysFont('monospace', 22)
        except Exception as exc:
            LOG.warning(f'Failed to initialize HUD window, disabling HUD: {exc}')
            self.hud_enabled = False
            self.hud_surface = None
            self.hud_font = None

    def _setup_video_recording(self) -> None:
        if not self.args.record_video:
            return
        raw_output_path = os.path.abspath(self.args.record_video)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if os.path.isdir(raw_output_path):
            output_path = os.path.join(raw_output_path, f'carla_view_{timestamp}.mp4')
        else:
            base_name = os.path.basename(raw_output_path)
            name, ext = os.path.splitext(base_name)
            if not ext:
                ext = '.mp4'
            parent_dir = os.path.dirname(raw_output_path)
            output_path = os.path.join(parent_dir, f'{name}_{timestamp}{ext}')
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        blueprint = self.world.get_blueprint_library().find('sensor.camera.rgb')
        blueprint.set_attribute('image_size_x', str(self.args.record_video_width))
        blueprint.set_attribute('image_size_y', str(self.args.record_video_height))
        blueprint.set_attribute('fov', str(self.args.record_video_fov))
        blueprint.set_attribute('sensor_tick', '0.0')

        camera_transform = carla.Transform(
            carla.Location(x=-self.args.record_camera_distance, y=0.0, z=self.args.record_camera_height),
            carla.Rotation(pitch=self.args.record_camera_pitch, yaw=0.0, roll=0.0),
        )
        self.record_camera = self.world.spawn_actor(
            blueprint,
            camera_transform,
            attach_to=self.hero,
            attachment_type=carla.AttachmentType.SpringArmGhost,
        )
        self.record_camera.listen(self._on_record_video_frame)

        fps = 1.0 / max(self.args.fixed_delta_seconds, 1e-6)
        file_ext = os.path.splitext(output_path)[1].lower()
        fourcc = cv2.VideoWriter_fourcc(*('mp4v' if file_ext == '.mp4' else 'XVID'))
        self.video_writer = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (self.args.record_video_width, self.args.record_video_height),
        )
        if not self.video_writer.isOpened():
            if self.record_camera is not None:
                self.record_camera.stop()
                self.record_camera.destroy()
                self.record_camera = None
            self.video_writer = None
            raise RuntimeError(f'Failed to open video writer: {output_path}')
        self.video_output_path = output_path
        LOG.info(f'录像已启用: {output_path}')

    def _on_record_video_frame(self, image: carla.Image) -> None:
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        self.record_latest_frame = array[:, :, :3].copy()

    def _build_grouped_overlay_lines(self, hud):
        predicted_traj = hud.get('predicted_traj') or []
        overlay_lines = [
            'INPUT:',
            f'  speed: {hud.get("speed", 0.0):.3f}',
            f'  acceleration_xy: {_format_sequence(hud.get("acceleration_xy") or [], precision=3)}',
            f'  local_command_xy: {_format_sequence(hud.get("local_command_xy") or [], precision=3)}',
            f'  command(raw/model): {hud.get("command_raw")} {hud.get("command_raw_name")} / {hud.get("command")} {hud.get("command_name")}',
            f'  command_onehot: {_format_sequence(hud.get("command_onehot") or [], precision=0)}',
            'OUTPUT:',
            f'  trajectory_point_count: {len(predicted_traj)}',
            f'  use_post_processor: {hud.get("use_trajectory_post_processor")}',
            'PID OUTPUT:',
            f'  desired_speed: {hud.get("desired_speed", 0.0):.3f}',
            f'  steer: {hud.get("steer", 0.0):.3f}',
            f'  throttle: {hud.get("throttle", 0.0):.3f}',
            f'  brake: {hud.get("brake", 0.0):.3f}',
        ]
        trajectory_lines = []
        for idx in range(0, len(predicted_traj), 2):
            point = predicted_traj[idx]
            line = f'  p{idx}: ({float(point[0]):.3f}, {float(point[1]):.3f})'
            if idx + 1 < len(predicted_traj):
                next_point = predicted_traj[idx + 1]
                line += f'    p{idx + 1}: ({float(next_point[0]):.3f}, {float(next_point[1]):.3f})'
            trajectory_lines.append(line)
        overlay_lines.extend(trajectory_lines)
        overlay_lines.extend([
            'DEBUG:',
            f'  target_angle/pred_angle/gap: {hud.get("local_command_angle_deg")} / {hud.get("pred_terminal_heading_deg")} / {hud.get("target_pred_angle_gap_deg")}',
            f'  pred_path_length: {hud.get("pred_path_length")}',
            f'  route_len: {hud.get("route_len")}',
            f'  near_node: {hud.get("near_node")}',
            f'  ego_yaw/route_heading/error: {hud.get("ego_yaw_deg")} / {hud.get("route_heading_deg")} / {hud.get("heading_error_deg")}',
            f'  near_node_distance: {hud.get("near_node_distance")}',
            f'  traffic_light: {hud.get("traffic_light_state")}',
        ])
        return overlay_lines

    @staticmethod
    def _compute_overlay_layout(total_lines, available_height, margin_top, margin_bottom, preferred_step, min_step):
        usable_height = max(available_height - margin_top - margin_bottom, min_step)
        if total_lines <= 0:
            return preferred_step
        return max(min_step, min(preferred_step, usable_height // total_lines))

    def _draw_video_overlay(self, frame: np.ndarray, hud) -> np.ndarray:
        if frame is None:
            return frame
        overlay_frame = frame.copy()
        lines = self._build_grouped_overlay_lines(hud)
        font = cv2.FONT_HERSHEY_SIMPLEX
        margin_x = 18
        margin_y = 28
        line_height = self._compute_overlay_layout(
            total_lines=len(lines),
            available_height=overlay_frame.shape[0],
            margin_top=margin_y,
            margin_bottom=16,
            preferred_step=22,
            min_step=14,
        )
        font_scale = max(0.36, min(0.55, line_height / 40.0))
        thickness = 1
        y = margin_y
        for line in lines:
            if y > overlay_frame.shape[0] - 12:
                break
            cv2.putText(overlay_frame, line, (margin_x + 1, y + 1), font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
            cv2.putText(overlay_frame, line, (margin_x, y), font, font_scale, (240, 240, 240), thickness, cv2.LINE_AA)
            y += line_height
        return overlay_frame

    def _write_video_frame(self) -> None:
        if self.video_writer is None or self.record_latest_frame is None:
            return
        hud = getattr(self.agent, 'latest_hud', {}) if self.agent is not None else {}
        frame = self._draw_video_overlay(self.record_latest_frame, hud)
        self.video_writer.write(frame)

    # ---------- 初始化 ----------
    def setup(self) -> None:
        self.client = carla.Client(self.args.host, self.args.port)
        self.client.set_timeout(self.args.timeout)
        self.route_config = self._load_route_config()

        # 地图
        if self.args.town:
            self.world = self.client.load_world(self.args.town)
        else:
            self.world = self.client.get_world()
        current_map_name = self.world.get_map().name.split('/')[-1]
        if self.route_config is not None and self.route_config.town != current_map_name:
            LOG.warning(
                "Route town %s does not match current map %s, falling back to current-map route generation",
                self.route_config.town,
                current_map_name,
            )
            self.route_config = None
        self.original_settings = self.world.get_settings()
        self.spectator = self.world.get_spectator()

        # 交通管理器
        self.traffic_manager = self.client.get_trafficmanager(self.args.tm_port)
        self.traffic_manager.set_global_distance_to_leading_vehicle(2.5)
        self.traffic_manager.set_synchronous_mode(True)
        if self.args.seed is not None:
            self.traffic_manager.set_random_device_seed(self.args.seed)
            random.seed(self.args.seed)

        # 同步模式
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.args.fixed_delta_seconds
        self.world.apply_settings(settings)

        CarlaDataProvider.set_client(self.client)
        CarlaDataProvider.set_world(self.world)
        GameTime.restart()

        if self.route_config is not None:
            self.precomputed_gps_route, self.precomputed_route = self._build_bench2drive_route(self.route_config)

        # 先生成 ego vehicle（确保 spawn point 不被交通流占用）
        self.hero = self._spawn_hero()
        CarlaDataProvider.register_actor(self.hero, self.hero.get_transform())

        # 生成交通流和行人（排除 ego 占用的 spawn point）
        LOG.info('开始生成交通流...')
        self.traffic_vehicle_ids = spawn_traffic_vehicles(
            self.client, self.world, self.args, self.args.tm_port,
            exclude_location=self.hero.get_transform().location)
        self.walkers_list, self.walker_all_ids = spawn_walkers(
            self.client, self.world, self.args)

        # 规划全局路线
        if self.precomputed_route is None:
            gps_route, route = self._build_route()
        else:
            gps_route, route = self.precomputed_gps_route, self.precomputed_route

        # 初始化 PAD agent
        self.agent = padAgent(self.args.host, self.args.port)
        config_string = f'{self.args.config}+{self.args.checkpoint}+standalone'
        self.agent.setup(config_string)
        self.agent.near_node_min_distance = self.args.near_node_distance
        self.agent.use_trajectory_post_processor = self.args.use_trajectory_post_processor
        self.agent.set_global_plan(gps_route, route)
        self.agent.hero_actor = self.hero
        self._setup_video_recording()

        # 挂载传感器
        self.wrapper = AgentWrapper(self.agent)
        self.wrapper.setup_sensors(self.hero)
        self._setup_hud()

        # 预热
        for _ in range(self.args.warmup_ticks):
            self._tick_world()

        LOG.info('初始化完成，开始闭环控制')

    # ---------- Hero ----------
    def _spawn_hero(self) -> carla.Actor:
        blueprint_library = self.world.get_blueprint_library()
        blueprint = blueprint_library.filter(self.args.blueprint)[0]
        blueprint.set_attribute('role_name', 'hero')
        if self.precomputed_route:
            start_transform = carla.Transform(
                self.precomputed_route[0][0].location + carla.Location(z=0.5),
                self.precomputed_route[0][0].rotation,
            )
            hero = self.world.try_spawn_actor(blueprint, start_transform)
            if hero is None:
                raise RuntimeError('Failed to spawn hero at Bench2Drive route start')
            hero.set_autopilot(False)
            return hero
        spawn_points = self.world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError('No spawn points found in current map')
        if self.args.spawn_index is None:
            self.selected_spawn_index = random.randrange(len(spawn_points))
        else:
            self.selected_spawn_index = self.args.spawn_index % len(spawn_points)
        start_transform = spawn_points[self.selected_spawn_index]
        hero = self.world.try_spawn_actor(blueprint, start_transform)
        if hero is None:
            raise RuntimeError(f'Failed to spawn hero at spawn index {self.selected_spawn_index}')
        hero.set_autopilot(False)
        LOG.info(f'ego spawn index: {self.selected_spawn_index}')
        return hero

    # ---------- 路线 ----------
    def _select_destination(self) -> carla.Transform:
        spawn_points = self.world.get_map().get_spawn_points()
        if len(spawn_points) < 2:
            raise RuntimeError('Need at least 2 spawn points to build a route')
        start_index = self.selected_spawn_index if self.selected_spawn_index is not None else 0
        start_transform = spawn_points[start_index]
        if self.args.destination_index is not None:
            destination = spawn_points[self.args.destination_index % len(spawn_points)]
            if destination.location.distance(start_transform.location) > 1.0:
                return destination

        candidate_transforms = [
            transform for idx, transform in enumerate(spawn_points)
            if idx != start_index and transform.location.distance(start_transform.location) > 30.0
        ]
        if not candidate_transforms:
            candidate_transforms = [
                transform for idx, transform in enumerate(spawn_points)
                if idx != start_index and transform.location.distance(start_transform.location) > 1.0
            ]
        if not candidate_transforms:
            raise RuntimeError('Could not select a destination spawn point')
        return random.choice(candidate_transforms)

    def _build_route(self):
        start_transform = self.hero.get_transform()
        destination_transform = self._select_destination()
        # 先将 ego 位置投射到当前车道中心，确保路线从 ego 所在车道开始
        ego_waypoint = self.world.get_map().get_waypoint(
            start_transform.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        destination_waypoint = self.world.get_map().get_waypoint(
            destination_transform.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        gps_route, route = interpolate_trajectory([
            ego_waypoint.transform.location,
            destination_waypoint.transform.location,
        ])
        if not route:
            raise RuntimeError('Failed to build interpolated route')
        first_option = route[0][1]
        ego_gps = _location_to_gps(*_get_latlon_ref(self.world), ego_waypoint.transform.location)
        route[0] = (ego_waypoint.transform, first_option)
        gps_route[0] = (ego_gps, first_option)
        self.destination_transform = destination_transform
        self.route_world_transforms = [wt for wt, _ in route]
        # 调试: 打印 ego 和 route 前几个点的精确坐标
        el = start_transform.location
        wl = ego_waypoint.transform.location
        LOG.info(f'[DEBUG] ego spawn     = ({el.x:.3f}, {el.y:.3f}, {el.z:.3f})')
        LOG.info(f'[DEBUG] ego waypoint  = ({wl.x:.3f}, {wl.y:.3f}, {wl.z:.3f})  lane_id={ego_waypoint.lane_id}  road_id={ego_waypoint.road_id}')
        for i, (wt, opt) in enumerate(route[:5]):
            rl = wt.location
            LOG.info(f'[DEBUG] route[{i}]      = ({rl.x:.3f}, {rl.y:.3f}, {rl.z:.3f})  option={opt}')
        LOG.info(f'全局路线: {len(route)} 个路点, '
                 f'起点=({ego_waypoint.transform.location.x:.1f}, {ego_waypoint.transform.location.y:.1f}), '
                 f'终点=({destination_transform.location.x:.1f}, {destination_transform.location.y:.1f})')
        return gps_route, route

    def _build_bench2drive_route(self, route_config):
        gps_route, route = interpolate_trajectory(route_config.keypoints)
        if not route:
            raise RuntimeError('Failed to build interpolated route from Bench2Drive keypoints')
        self.destination_transform = route[-1][0]
        self.route_world_transforms = [wt for wt, _ in route]
        LOG.info(f'Bench2Drive route loaded: {route_config.name} town={route_config.town} points={len(route)}')
        return gps_route, route

    # ---------- Tick ----------
    def _tick_world(self):
        self.world.tick()
        snapshot = self.world.get_snapshot()
        GameTime.on_carla_tick(snapshot.timestamp)
        CarlaDataProvider.on_carla_tick()
        time.sleep(self.args.sensor_wait)
        return snapshot

    # ---------- Spectator ----------
    def _update_spectator(self) -> None:
        if self.spectator is None or self.hero is None or not self.hero.is_alive:
            return
        ego_transform = self.hero.get_transform()
        forward = ego_transform.get_forward_vector()
        spectator_location = ego_transform.location + carla.Location(
            x=-6.5 * forward.x,
            y=-6.5 * forward.y,
            z=2.8,
        )
        spectator_rotation = carla.Rotation(
            pitch=-12.0,
            yaw=ego_transform.rotation.yaw,
            roll=0.0,
        )
        self.spectator.set_transform(carla.Transform(spectator_location, spectator_rotation))

    # ---------- 可视化 ----------
    def _get_forward_route_locations(self, max_distance: float = 60.0):
        if self.hero is None or not self.hero.is_alive:
            return []
        if not self.route_world_transforms:
            return []
        ego_transform = self.hero.get_transform()
        ego_location = ego_transform.location
        forward = ego_transform.get_forward_vector()
        best_index = self.route_preview_index
        best_distance = float('inf')
        start_search = max(0, self.route_preview_index - 20)
        for index in range(start_search, len(self.route_world_transforms)):
            location = self.route_world_transforms[index].location
            dx = location.x - ego_location.x
            dy = location.y - ego_location.y
            longitudinal = dx * forward.x + dy * forward.y
            distance = (dx * dx + dy * dy) ** 0.5
            if longitudinal < -5.0:
                continue
            if distance < best_distance:
                best_distance = distance
                best_index = index
        self.route_preview_index = best_index

        points = []
        cumulative_distance = 0.0
        previous = ego_location
        for transform in self.route_world_transforms[best_index:]:
            location = transform.location
            segment = location.distance(previous)
            cumulative_distance += segment
            points.append(carla.Location(x=location.x, y=location.y, z=ego_location.z + 0.35))
            previous = location
            if cumulative_distance >= max_distance:
                break
        return points

    def _get_future_route_points(self, count: int = 5):
        preview_points = self._get_forward_route_locations(max_distance=80.0)
        if not preview_points:
            return []
        if len(preview_points) <= count:
            return preview_points
        anchors = []
        anchor_spacing = 8.0
        target_distance = anchor_spacing
        cumulative_distance = 0.0
        previous = self.hero.get_transform().location
        for point in preview_points:
            cumulative_distance += point.distance(previous)
            if cumulative_distance >= target_distance:
                anchors.append(point)
                target_distance += anchor_spacing
                if len(anchors) >= count:
                    break
            previous = point
        if not anchors:
            anchors = preview_points[:count]
        return anchors[:count]

    def _densify_route_locations(self, locations, spacing: float = 0.75):
        if not locations:
            return []
        dense_points = [locations[0]]
        for start, end in zip(locations[:-1], locations[1:]):
            dx = end.x - start.x
            dy = end.y - start.y
            dz = end.z - start.z
            distance = max((dx * dx + dy * dy + dz * dz) ** 0.5, 1e-6)
            steps = max(int(distance / spacing), 1)
            for step in range(1, steps + 1):
                alpha = step / steps
                dense_points.append(
                    carla.Location(
                        x=start.x + dx * alpha,
                        y=start.y + dy * alpha,
                        z=start.z + dz * alpha,
                    )
                )
        return dense_points

    def _draw_route_preview(self) -> None:
        if self.hero is None or not self.hero.is_alive or self.world is None:
            return
        debug = self.world.debug
        ego_location = self.hero.get_transform().location + carla.Location(z=0.35)
        preview_points = self._get_forward_route_locations(40)
        dense_preview_points = self._densify_route_locations([ego_location] + preview_points)
        future_points = self._get_future_route_points(5)
        debug.draw_point(ego_location, size=0.16, color=START_COLOR, life_time=0.15, persistent_lines=False)
        debug.draw_string(ego_location + carla.Location(z=0.25), 'ego', False, START_COLOR, 0.15, False)
        for previous, point in zip(dense_preview_points[:-1], dense_preview_points[1:]):
            debug.draw_line(previous, point, thickness=0.16, color=ROUTE_COLOR, life_time=0.15, persistent_lines=False)
        for index, point in enumerate(future_points):
            debug.draw_point(point, size=0.12, color=ROUTE_COLOR, life_time=0.15, persistent_lines=False)
            debug.draw_string(point + carla.Location(z=0.2), f'wp{index + 1}', False, ROUTE_COLOR, 0.15, False)

    def _draw_predicted_trajectory(self) -> None:
        if self.hero is None or not self.hero.is_alive:
            return
        if not hasattr(self.agent, 'pid_metadata'):
            return
        local_plan = self.agent.pid_metadata.get('plan')
        if not local_plan:
            return
        ego_transform = self.hero.get_transform()
        ego_location = ego_transform.location
        forward = ego_transform.get_forward_vector()
        right = ego_transform.get_right_vector()
        world_points = []
        for lateral, longitudinal in local_plan:
            world_points.append(
                carla.Location(
                    x=ego_location.x + right.x * lateral + forward.x * longitudinal,
                    y=ego_location.y + right.y * lateral + forward.y * longitudinal,
                    z=ego_location.z + 0.35,
                )
            )
        debug = self.world.debug
        debug.draw_point(world_points[0], size=0.08, color=TRAJECTORY_COLOR, life_time=0.15, persistent_lines=False)
        for a, b in zip(world_points[:-1], world_points[1:]):
            debug.draw_line(a, b, thickness=0.18, color=TRAJECTORY_COLOR, life_time=0.15, persistent_lines=False)
            debug.draw_point(b, size=0.07, color=TRAJECTORY_COLOR, life_time=0.15, persistent_lines=False)

    def _draw_near_node_link(self) -> None:
        if self.hero is None or not self.hero.is_alive or self.world is None or self.agent is None:
            return
        near_node = getattr(self.agent, 'latest_hud', {}).get('near_node')
        if not near_node or len(near_node) < 2:
            return
        ego_location = self.hero.get_transform().location
        near_location = carla.Location(x=float(near_node[0]), y=float(near_node[1]), z=ego_location.z + 0.35)
        ego_draw = ego_location + carla.Location(z=0.35)
        debug = self.world.debug
        debug.draw_line(ego_draw, near_location, thickness=0.16, color=NEAR_NODE_COLOR, life_time=0.15, persistent_lines=False)
        debug.draw_point(near_location, size=0.10, color=NEAR_NODE_COLOR, life_time=0.15, persistent_lines=False)
        debug.draw_string(near_location + carla.Location(z=0.2), 'near_node', False, NEAR_NODE_COLOR, 0.15, False)

    def _render_hud(self) -> None:
        if not self.hud_enabled or self.agent is None:
            return
        hud = getattr(self.agent, 'latest_hud', {})
        if self.hud_surface is None or self.hud_font is None:
            return
        try:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.hud_enabled = False
                    pygame.display.quit()
                    self.hud_surface = None
                    return
            self.hud_surface.fill((8, 8, 8))
            bev_rect = pygame.Rect(0, 0, 1600, 900)
            panel_rect = pygame.Rect(1600, 0, 520, 900)
            pygame.draw.rect(self.hud_surface, (12, 12, 12), bev_rect)
            pygame.draw.rect(self.hud_surface, (20, 20, 20), panel_rect)
            ego_transform = self.hero.get_transform() if self.hero is not None and self.hero.is_alive else None
            center_x = bev_rect.width * 0.5
            center_y = bev_rect.height * 0.62
            scale = 14.0
            if ego_transform is not None:
                forward = ego_transform.get_forward_vector()
                right = ego_transform.get_right_vector()

                def world_to_bev(location):
                    dx = location.x - ego_transform.location.x
                    dy = location.y - ego_transform.location.y
                    longitudinal = dx * forward.x + dy * forward.y
                    lateral = dx * right.x + dy * right.y
                    return (int(center_x + lateral * scale), int(center_y - longitudinal * scale))

                pygame.draw.circle(self.hud_surface, (0, 128, 255), (int(center_x), int(center_y)), 10)
                pygame.draw.polygon(
                    self.hud_surface,
                    (0, 128, 255),
                    [
                        (int(center_x), int(center_y - 18)),
                        (int(center_x - 10), int(center_y + 12)),
                        (int(center_x + 10), int(center_y + 12)),
                    ],
                )

                preview_points = self._get_forward_route_locations(40)
                dense_preview_points = self._densify_route_locations([ego_transform.location] + preview_points)
                future_points = self._get_future_route_points(5)
                preview_polyline = [(int(center_x), int(center_y))]
                for point in dense_preview_points[1:]:
                    preview_polyline.append(world_to_bev(point))
                if len(preview_polyline) >= 2:
                    pygame.draw.lines(self.hud_surface, (0, 255, 0), False, preview_polyline, 3)
                for index, point in enumerate(future_points):
                    pt = world_to_bev(point)
                    pygame.draw.circle(self.hud_surface, (0, 255, 0), pt, 6)
                    label = self.hud_font.render(f'wp{index + 1}', True, (0, 255, 0))
                    self.hud_surface.blit(label, (pt[0] + 8, pt[1] - 8))

                near_node = hud.get('near_node')
                if near_node and len(near_node) >= 2:
                    near_pt = world_to_bev(carla.Location(x=float(near_node[0]), y=float(near_node[1]), z=ego_transform.location.z))
                    pygame.draw.line(self.hud_surface, (200, 0, 255), (int(center_x), int(center_y)), near_pt, 2)
                    pygame.draw.circle(self.hud_surface, (200, 0, 255), near_pt, 7)

                predicted_traj = hud.get('predicted_traj') or []
                if predicted_traj:
                    traj_points = []
                    for lateral, longitudinal in predicted_traj:
                        traj_points.append((int(center_x + lateral * scale), int(center_y - longitudinal * scale)))
                    if traj_points:
                        pygame.draw.lines(self.hud_surface, (255, 64, 64), False, [(int(center_x), int(center_y))] + traj_points, 3)
                        for point in traj_points:
                            pygame.draw.circle(self.hud_surface, (255, 64, 64), point, 4)

                command_xy = hud.get('local_command_xy')
                if command_xy and len(command_xy) >= 2:
                    cmd_pt = (int(center_x + command_xy[0] * scale), int(center_y - command_xy[1] * scale))
                    pygame.draw.line(self.hud_surface, (255, 255, 0), (int(center_x), int(center_y)), cmd_pt, 2)
                    pygame.draw.circle(self.hud_surface, (255, 255, 0), cmd_pt, 6)

                for distance in [5, 10, 20, 30]:
                    radius = int(distance * scale)
                    pygame.draw.circle(self.hud_surface, (45, 45, 45), (int(center_x), int(center_y)), radius, 1)

            lines = self._build_grouped_overlay_lines(hud)
            line_step = self._compute_overlay_layout(
                total_lines=len(lines),
                available_height=panel_rect.height,
                margin_top=30,
                margin_bottom=20,
                preferred_step=28,
                min_step=18,
            )
            font_size = max(14, min(22, line_step - 4))
            panel_font = self.hud_font if font_size == 22 else pygame.font.SysFont('monospace', font_size)
            y = 30
            for line in lines:
                if y > panel_rect.bottom - line_step:
                    break
                text_surface = panel_font.render(line, True, (255, 255, 255))
                self.hud_surface.blit(text_surface, (1612, y))
                y += line_step
            pygame.display.flip()
        except Exception as exc:
            LOG.warning(f'HUD render failed, disabling HUD: {exc}')
            self.hud_enabled = False
            try:
                pygame.display.quit()
            except Exception:
                pass
            self.hud_surface = None
            self.hud_font = None

    # ---------- 主循环 ----------
    def run(self) -> None:
        start_time = time.time()
        for step in range(self.args.max_steps):
            self._tick_world()
            self._update_spectator()
            if self.agent.hero_actor is None or not self.agent.hero_actor.is_alive:
                self.agent.hero_actor = self.hero
            control = self.wrapper()
            self._draw_route_preview()
            self._draw_near_node_link()
            self._draw_predicted_trajectory()
            self._render_hud()
            self._write_video_frame()
            self.hero.apply_control(control)

            if step % self.args.log_every == 0:
                hero_transform = self.hero.get_transform()
                location = hero_transform.location
                velocity = self.hero.get_velocity()
                speed = (velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2) ** 0.5
                remaining = location.distance(self.destination_transform.location)
                nav_debug = getattr(self.agent, 'nav_debug', {})
                pid_debug = getattr(self.agent, 'pid_metadata', {})
                hud_debug = getattr(self.agent, 'latest_hud', {})
                future_points = [
                    (round(point.x, 2), round(point.y, 2))
                    for point in self._get_future_route_points(5)
                ]
                near_node = nav_debug.get('near_node')
                near_node_distance = None
                if near_node and len(near_node) >= 2:
                    near_location = carla.Location(x=float(near_node[0]), y=float(near_node[1]), z=location.z)
                    near_node_distance = round(location.distance(near_location), 3)
                route_heading_deg = None
                if len(future_points) >= 2:
                    route_heading_deg = _heading_deg_between_locations(
                        carla.Location(x=float(future_points[0][0]), y=float(future_points[0][1]), z=location.z),
                        carla.Location(x=float(future_points[1][0]), y=float(future_points[1][1]), z=location.z),
                    )
                ego_yaw_deg = float(hero_transform.rotation.yaw)
                heading_error_deg = None if route_heading_deg is None else _normalize_angle_deg(route_heading_deg - ego_yaw_deg)
                traffic_light_state = 'NONE'
                try:
                    if self.hero.is_at_traffic_light():
                        traffic_light_state = self.hero.get_traffic_light_state().name
                except Exception:
                    pass
                predicted_traj = pid_debug.get('plan') or []
                predicted_points = [
                    (round(point[0], 2), round(point[1], 2))
                    for point in predicted_traj
                ]
                local_command_xy = nav_debug.get('local_command_xy')
                local_command_angle_deg = _local_command_angle_deg(local_command_xy)
                pred_terminal_heading_deg = _trajectory_terminal_heading_deg(predicted_traj)
                pred_path_length = round(_trajectory_path_length(predicted_traj), 3)
                target_pred_angle_gap_deg = None
                if local_command_angle_deg is not None and pred_terminal_heading_deg is not None:
                    target_pred_angle_gap_deg = round(
                        _normalize_angle_deg(pred_terminal_heading_deg - local_command_angle_deg),
                        2,
                    )
                print(
                    f'step={step} location=({location.x:.2f}, {location.y:.2f}) '
                    f'speed={speed:.2f} remaining={remaining:.2f} '
                    f'control=({control.steer:.3f}, {control.throttle:.3f}, {control.brake:.3f}) '
                    f'command_raw={nav_debug.get("near_command")}:{nav_debug.get("near_command_name")} '
                    f'command_model={nav_debug.get("model_command")}:{nav_debug.get("model_command_name")} '
                    f'gps2loc={nav_debug.get("gps_to_location")} '
                    f'hero_loc={nav_debug.get("hero_location")} '
                    f'ego_yaw_deg={ego_yaw_deg:.2f} '
                    f'route_heading_deg={None if route_heading_deg is None else round(route_heading_deg, 2)} '
                    f'heading_error_deg={None if heading_error_deg is None else round(heading_error_deg, 2)} '
                    f'near_node={near_node} '
                    f'near_node_distance={near_node_distance} '
                    f'local_cmd={local_command_xy} '
                    f'local_cmd_angle_deg={None if local_command_angle_deg is None else round(local_command_angle_deg, 2)} '
                    f'route_len={nav_debug.get("route_len")} '
                    f'traffic_light={traffic_light_state} '
                    f'desired_speed={pid_debug.get("desired_speed")} '
                    f'speed_ratio={pid_debug.get("speed_ratio")} '
                    f'brake_by_speed={pid_debug.get("brake_by_speed")} '
                    f'brake_by_ratio={pid_debug.get("brake_by_ratio")} '
                    f'pred_point_count={len(predicted_traj)} '
                    f'pred_path_length={pred_path_length} '
                    f'pred_final_heading_deg={None if pred_terminal_heading_deg is None else round(pred_terminal_heading_deg, 2)} '
                    f'target_pred_angle_gap_deg={target_pred_angle_gap_deg} '
                    f'pred_traj={predicted_points} '
                    f'future_wps={future_points}',
                    flush=True,
                )
                hud_debug['ego_yaw_deg'] = round(ego_yaw_deg, 2)
                hud_debug['route_heading_deg'] = None if route_heading_deg is None else round(route_heading_deg, 2)
                hud_debug['heading_error_deg'] = None if heading_error_deg is None else round(heading_error_deg, 2)
                hud_debug['near_node_distance'] = near_node_distance
                hud_debug['traffic_light_state'] = traffic_light_state
                hud_debug['local_command_angle_deg'] = None if local_command_angle_deg is None else round(local_command_angle_deg, 2)
                hud_debug['pred_terminal_heading_deg'] = None if pred_terminal_heading_deg is None else round(pred_terminal_heading_deg, 2)
                hud_debug['pred_path_length'] = pred_path_length
                hud_debug['target_pred_angle_gap_deg'] = target_pred_angle_gap_deg

            if self.hero.get_transform().location.distance(self.destination_transform.location) <= self.args.stop_distance:
                LOG.info('destination reached')
                break

            if time.time() - start_time >= self.args.max_runtime_seconds:
                LOG.info('max runtime reached')
                break

    # ---------- 清理 ----------
    def cleanup(self) -> None:
        LOG.info('清理中...')
        if self.wrapper is not None:
            try:
                self.wrapper.cleanup()
            except Exception:
                pass
        if self.record_camera is not None:
            try:
                self.record_camera.stop()
            except Exception:
                pass
            try:
                self.record_camera.destroy()
            except Exception:
                pass
            self.record_camera = None
        if self.agent is not None:
            try:
                self.agent.destroy()
            except Exception:
                pass
        if self.walker_all_ids:
            try:
                all_actors = self.world.get_actors(self.walker_all_ids)
                for i in range(0, len(self.walker_all_ids), 2):
                    ctrl = all_actors[i]
                    if ctrl and ctrl.is_alive:
                        ctrl.stop()
                self.client.apply_batch([carla.command.DestroyActor(x) for x in self.walker_all_ids])
            except Exception:
                pass
        if self.traffic_vehicle_ids:
            try:
                self.client.apply_batch([carla.command.DestroyActor(x) for x in self.traffic_vehicle_ids])
            except Exception:
                pass
        if self.hero is not None:
            try:
                self.hero.destroy()
            except Exception:
                pass
        if self.world is not None and self.original_settings is not None:
            try:
                self.world.apply_settings(self.original_settings)
            except Exception:
                pass
        if self.hud_enabled:
            try:
                pygame.display.quit()
                pygame.quit()
            except Exception:
                pass
        if self.video_writer is not None:
            try:
                self.video_writer.release()
                if self.video_output_path:
                    LOG.info(f'录像已保存: {self.video_output_path}')
            except Exception:
                pass
            self.video_writer = None
        LOG.info('清理完成')

# ==================== 参数解析 ====================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='PAD 端到端闭环控制 + 交通流 + 可视化')

    # CARLA 连接
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=DEFAULT_CARLA_PORT)
    parser.add_argument('--town', default=None, help='指定地图，默认使用当前地图')
    parser.add_argument('--timeout', type=float, default=60.0)
    parser.add_argument('--routes', default=DEFAULT_ROUTES, help='Bench2Drive routes xml')
    parser.add_argument('--route-id', default=None, help='Bench2Drive route id')

    # PAD 模型
    parser.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT)
    parser.add_argument('--config', default=DEFAULT_CONFIG)

    # ego vehicle
    parser.add_argument('--blueprint', default='vehicle.tesla.model3')
    parser.add_argument('--spawn-index', type=int, default=None)
    parser.add_argument('--destination-index', type=int, default=None)
    parser.add_argument('--disable-hud', action='store_true', help='禁用 OpenCV 实时 HUD 窗口')
    parser.add_argument('--record-video', default=None, help='保存第三人称 CARLA 视角视频，例如 output/run.mp4')
    parser.add_argument('--record-video-width', type=int, default=1280)
    parser.add_argument('--record-video-height', type=int, default=720)
    parser.add_argument('--record-video-fov', type=float, default=110.0)
    parser.add_argument('--record-camera-distance', type=float, default=7.5)
    parser.add_argument('--record-camera-height', type=float, default=3.2)
    parser.add_argument('--record-camera-pitch', type=float, default=-10.0)

    # 交通流
    parser.add_argument('-n', '--number-of-vehicles', type=int, default=50, help='交通流车辆数量')
    parser.add_argument('-w', '--number-of-walkers', type=int, default=30, help='行人数量')
    parser.add_argument('--safe', action='store_true', help='只生成不易事故的小汽车')
    parser.add_argument('--filterv', default='vehicle.*', help='车辆蓝图过滤')
    parser.add_argument('--generationv', default='All', help='车辆代次 (1,2,3,All)')
    parser.add_argument('--filterw', default='walker.pedestrian.*', help='行人蓝图过滤')
    parser.add_argument('--generationw', default='All', help='行人代次')
    parser.add_argument('--car-lights-on', action='store_true', help='自动管理车灯')
    parser.add_argument('--tm-port', type=int, default=8000, help='交通管理器端口')
    parser.add_argument('--seed', type=int, default=None, help='随机种子')
    parser.add_argument('--seedw', type=int, default=0, help='行人随机种子')
    parser.add_argument('--percentage-pedestrians-running', type=float, default=0.0)
    parser.add_argument('--percentage-pedestrians-crossing', type=float, default=0.0)

    # 仿真参数
    parser.add_argument('--fixed-delta-seconds', type=float, default=0.05)
    parser.add_argument('--sensor-wait', type=float, default=0.02)
    parser.add_argument('--warmup-ticks', type=int, default=3)
    parser.add_argument('--max-steps', type=int, default=4000)
    parser.add_argument('--max-runtime-seconds', type=float, default=1800.0)
    parser.add_argument('--stop-distance', type=float, default=15.0)
    parser.add_argument('--near-node-distance', type=float, default=10.0)
    parser.add_argument('--disable-trajectory-post-processor', action='store_false', dest='use_trajectory_post_processor', help='关闭轨迹后处理，直接将 PAD 原始输出送入 PID')
    parser.set_defaults(use_trajectory_post_processor=True)
    parser.add_argument('--log-every', type=int, default=20)

    return parser.parse_args()


# ==================== 入口 ====================

def main() -> None:
    args = parse_args()
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f'checkpoint not found: {args.checkpoint}')
    if not os.path.exists(args.config):
        raise FileNotFoundError(f'config not found: {args.config}')
    if args.route_id is not None and not os.path.exists(args.routes):
        raise FileNotFoundError(f'routes xml not found: {args.routes}')

    runner = ClosedLoopRunner(args)
    try:
        runner.setup()
        runner.run()
    except KeyboardInterrupt:
        LOG.info('用户中断')
    finally:
        runner.cleanup()


if __name__ == '__main__':
    main()
