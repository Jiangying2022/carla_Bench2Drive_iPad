import os
import sys
import importlib
import importlib.machinery
import importlib.util
import math
import site
import uuid

import cv2
import numpy as np

ORIGINAL_SYS_PATH = list(sys.path)
ENV_SITE_PACKAGES = os.path.abspath(
    os.path.join(
        os.path.dirname(sys.executable),
        '..',
        'lib',
        f'python{sys.version_info.major}.{sys.version_info.minor}',
        'site-packages',
    )
)
if ENV_SITE_PACKAGES not in sys.path:
    sys.path.insert(0, ENV_SITE_PACKAGES)
os.environ['PYTHONNOUSERSITE'] = '1'
USER_SITE_MARKERS = {
    os.path.realpath(site.getusersitepackages()),
    os.path.realpath(os.path.join(os.path.expanduser('~'), '.local')),
}
CURRENT_WORKDIR = os.getcwd()


def _is_user_site_path(path):
    if not path:
        return False
    real_path = os.path.realpath(path)
    return any(real_path.startswith(marker) for marker in USER_SITE_MARKERS if marker)


sys.path = [
    path for path in sys.path
    if path not in ('', CURRENT_WORKDIR) and not _is_user_site_path(path)
]
for module_name in list(sys.modules):
    if module_name == 'mmcv' or module_name.startswith('mmcv.'):
        sys.modules.pop(module_name, None)

importlib.invalidate_caches()


def _load_module_from_search_path(module_name, search_path):
    spec = importlib.machinery.PathFinder.find_spec(module_name, search_path)
    if spec is None or spec.loader is None:
        raise ModuleNotFoundError(f'Unable to locate {module_name} from search path: {search_path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


mmcv_module = _load_module_from_search_path('mmcv', [ENV_SITE_PACKAGES])
_load_module_from_search_path('mmcv._ext', mmcv_module.__path__)

from bench2drive.leaderboard.team_code.hipad_b2d_agent import SparseAgent

sys.path = ORIGINAL_SYS_PATH


RAW_COMMAND_NAMES = {
    -1: 'VOID',
    1: 'LEFT',
    2: 'RIGHT',
    3: 'STRAIGHT',
    4: 'LANEFOLLOW',
    5: 'CHANGELANELEFT',
    6: 'CHANGELANERIGHT',
}

MODEL_COMMAND_NAMES = {
    0: 'LEFT',
    1: 'RIGHT',
    2: 'STRAIGHT',
    3: 'LANEFOLLOW',
    4: 'CHANGELANELEFT',
    5: 'CHANGELANERIGHT',
}


DEFAULT_HIPAD_ROOT = '/home/pnc/HiP-AD'
DEFAULT_HIPAD_CONFIG = os.path.join(DEFAULT_HIPAD_ROOT, 'projects', 'configs', 'hipad_b2d_stage2.py')
DEFAULT_HIPAD_CHECKPOINT = os.path.join(DEFAULT_HIPAD_ROOT, 'ckpts', 'hipad_stage2.pth')


def get_entry_point():
    return 'hipadAdapterAgent'


def _heading_deg(start_xy, end_xy):
    delta = np.asarray(end_xy, dtype=np.float32) - np.asarray(start_xy, dtype=np.float32)
    return float(math.degrees(math.atan2(float(delta[1]), float(delta[0]))))


def _angle_deg(vector_xy):
    vector_xy = np.asarray(vector_xy, dtype=np.float32)
    return float(math.degrees(math.atan2(float(vector_xy[1]), float(vector_xy[0]))))


def _round_pair_list(points, limit=5):
    rounded = []
    for point in list(points)[:limit]:
        if point is None or len(point) < 2:
            continue
        rounded.append([round(float(point[0]), 3), round(float(point[1]), 3)])
    return rounded


def _filter_distinct_points(points, limit=None, min_distance=1e-3):
    filtered = []
    for point in list(points):
        if point is None or len(point) < 2:
            continue
        candidate = [float(point[0]), float(point[1])]
        if not filtered:
            filtered.append(candidate)
        else:
            previous = np.asarray(filtered[-1], dtype=np.float32)
            current = np.asarray(candidate, dtype=np.float32)
            if float(np.linalg.norm(current - previous)) > float(min_distance):
                filtered.append(candidate)
        if limit is not None and len(filtered) >= limit:
            break
    return filtered


def _pairwise_distances(points, limit=4):
    distances = []
    clipped = list(points)[:limit + 1]
    for idx in range(len(clipped) - 1):
        start_xy = np.asarray(clipped[idx], dtype=np.float32)
        end_xy = np.asarray(clipped[idx + 1], dtype=np.float32)
        distance = float(np.linalg.norm(end_xy - start_xy))
        if distance <= 1e-3:
            continue
        distances.append(round(distance, 3))
    return distances


def _heading_gap_deg(heading_a, heading_b):
    if heading_a is None or heading_b is None:
        return None
    delta = float(heading_b) - float(heading_a)
    while delta > 180.0:
        delta -= 360.0
    while delta < -180.0:
        delta += 360.0
    return round(delta, 3)


class hipadAdapterAgent(SparseAgent):
    def setup(self, path_to_conf_file):
        save_root = os.environ.get('SAVE_PATH', '/tmp/hipad_adapter_outputs')
        os.environ['SAVE_PATH'] = os.path.join(save_root, uuid.uuid4().hex)
        os.environ.setdefault('ROUTES', 'hipad_standalone.xml')
        self.latest_front_image = None
        self.latest_hud = {}
        self.nav_debug = {}
        self.hero_actor = None
        self.near_node_min_distance = 4.0
        self.use_trajectory_post_processor = False
        self._last_nav_debug_print_step = -999999
        self._global_plan_dense = None
        self._global_plan_world_coord_dense = None
        super().setup(path_to_conf_file)

    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        self._global_plan_dense = list(global_plan_gps) if global_plan_gps is not None else None
        self._global_plan_world_coord_dense = list(global_plan_world_coord) if global_plan_world_coord is not None else None
        super().set_global_plan(global_plan_gps, global_plan_world_coord)

    def init(self):
        super().init()
        if self.save_name != 'standalone':
            return
        route_world = getattr(self, '_global_plan_world_coord_dense', None)
        if route_world is None:
            route_world = getattr(self, '_global_plan_world_coord', None)
        if not route_world:
            return
        route_planner_cls = self._route_planner.__class__
        self.lat_ref, self.lon_ref = 0.0, 0.0
        self._route_planner = route_planner_cls(
            self.near_node_min_distance,
            50.0,
            lat_ref=self.lat_ref,
            lon_ref=self.lon_ref,
        )
        self._route_planner.set_route(route_world, False)

    def tick(self, input_data):
        self.step += 1
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 20]
        imgs = {}
        for cam in self.lidar2img.keys():
            img = cv2.cvtColor(input_data[cam][1][:, :, :3], cv2.COLOR_BGR2RGB)
            _, img = cv2.imencode('.jpg', img, encode_param)
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)
            imgs[cam] = img
        bev = cv2.cvtColor(input_data['bev'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        gps = input_data['GPS'][1][:2]
        speed = input_data['SPEED'][1]['speed']
        compass = input_data['IMU'][1][-1]
        acceleration = input_data['IMU'][1][:3]
        angular_velocity = input_data['IMU'][1][3:6]

        hero_location = None
        if self.hero_actor is not None and self.hero_actor.is_alive:
            hero_transform = self.hero_actor.get_transform()
            hero_location = np.array([hero_transform.location.x, hero_transform.location.y], dtype=np.float32)
        if self.save_name == 'standalone' and hero_location is not None:
            pos = hero_location
        else:
            pos = self.gps_to_location(gps).astype(np.float32)

        waypoint_routes = self._route_planner.run_step(pos)
        planner_route_world = []
        for route_entry in list(waypoint_routes):
            route_point = route_entry[0]
            planner_route_world.append([
                float(route_point[0]),
                float(route_point[1]),
            ])
        near_node_xy = waypoint_routes[0][0]
        near_command = waypoint_routes[0][1]
        if len(waypoint_routes) >= 3:
            target_xy = waypoint_routes[1][0]
            target_xy_next = waypoint_routes[2][0]
            command = waypoint_routes[0][1]
            command_next = waypoint_routes[1][1]
        elif len(waypoint_routes) == 2:
            target_xy = target_xy_next = waypoint_routes[1][0]
            command = command_next = waypoint_routes[0][1]
        else:
            target_xy = target_xy_next = waypoint_routes[0][0]
            command = command_next = waypoint_routes[0][1]

        near_command_raw_value = int(getattr(near_command, 'value', near_command))
        command_raw_value = int(getattr(command, 'value', command))
        if np.isnan(compass):
            compass = 0.0
            acceleration = np.zeros(3, dtype=np.float32)
            angular_velocity = np.zeros(3, dtype=np.float32)

        ego_pos = np.array([pos[0], -pos[1]], dtype=np.float32)
        target_xy_flipped = np.array([target_xy[0], -target_xy[1]], dtype=np.float32)
        target_xy_next_flipped = np.array([target_xy_next[0], -target_xy_next[1]], dtype=np.float32)
        command_delta_xy = target_xy_flipped - ego_pos
        command_delta_xy_next = target_xy_next_flipped - ego_pos
        rotation_matrix = np.array(
            [[np.cos(compass), -np.sin(compass)], [np.sin(compass), np.cos(compass)]],
            dtype=np.float32,
        )
        local_command_xy = rotation_matrix @ command_delta_xy
        local_command_xy_next = rotation_matrix @ command_delta_xy_next
        model_command = command_raw_value
        if model_command < 0:
            model_command = 4
        model_command -= 1
        model_command_name = MODEL_COMMAND_NAMES.get(int(model_command), f'UNKNOWN({int(model_command)})')
        route_head_world = _round_pair_list(planner_route_world, limit=5)
        route_head_world_distinct = _filter_distinct_points(route_head_world, limit=3)
        route_head_distances = _pairwise_distances(route_head_world, limit=4)
        route_heading_01 = None
        route_heading_12 = None
        if len(route_head_world_distinct) >= 2:
            route_heading_01 = round(_heading_deg(route_head_world_distinct[0], route_head_world_distinct[1]), 3)
        if len(route_head_world_distinct) >= 3:
            route_heading_12 = round(_heading_deg(route_head_world_distinct[1], route_head_world_distinct[2]), 3)
        route_heading_gap = _heading_gap_deg(route_heading_01, route_heading_12)

        self.nav_debug = {
            'gps': gps.tolist(),
            'gps_to_location': pos.tolist() if hasattr(pos, 'tolist') else list(pos),
            'hero_location': None if hero_location is None else hero_location.tolist(),
            'near_node': near_node_xy.tolist() if hasattr(near_node_xy, 'tolist') else list(near_node_xy),
            'near_command': near_command_raw_value,
            'near_command_name': RAW_COMMAND_NAMES.get(near_command_raw_value, f'UNKNOWN({near_command_raw_value})'),
            'route_len': len(getattr(self._route_planner, 'route', [])),
            'planner_route_world': planner_route_world,
            'local_command_xy': local_command_xy.tolist(),
            'local_command_xy_next': local_command_xy_next.tolist(),
            'model_command': int(model_command),
            'model_command_name': model_command_name,
            'command_near_distance': float(np.linalg.norm(command_delta_xy)),
            'raw_theta': float(compass),
            'route_head_world': route_head_world,
            'route_head_distances': route_head_distances,
            'route_heading_01_deg': route_heading_01,
            'route_heading_12_deg': route_heading_12,
            'route_heading_gap_deg': route_heading_gap,
            'target_point_angle_deg': round(_angle_deg(local_command_xy), 3),
        }

        front = imgs.get('CAM_FRONT')
        self.latest_front_image = None if front is None else front.copy()
        tick_data = {
            'imgs': imgs,
            'gps': gps,
            'pos': pos,
            'bev': bev,
            'speed': speed,
            'compass': compass,
            'acceleration': acceleration,
            'angular_velocity': angular_velocity,
            'target_xy': target_xy,
            'target_xy_next': target_xy_next,
            'command': command,
            'command_next': command_next,
        }
        self._latest_tick_data = tick_data
        return tick_data

    def run_step(self, input_data, timestamp):
        control = super().run_step(input_data, timestamp)
        tick_data = getattr(self, '_latest_tick_data', {})
        command_raw_value = self.nav_debug.get('near_command', -1)
        command_model = self.nav_debug.get('model_command')
        command_onehot = [0.0] * 6
        if command_model is not None and 0 <= int(command_model) < len(command_onehot):
            command_onehot[int(command_model)] = 1.0
        display_plan = []
        if hasattr(self, 'pid_metadata'):
            display_plan = self.pid_metadata.get('plan_spat') or self.pid_metadata.get('plan_temp') or self.pid_metadata.get('plan') or []
            self.pid_metadata.setdefault('plan', display_plan)
            self.pid_metadata['use_trajectory_post_processor'] = False
        target_point = self.nav_debug.get('local_command_xy') or []
        target_point_next = self.nav_debug.get('local_command_xy_next') or []
        near_node = self.nav_debug.get('near_node')
        pred_spat_head = _round_pair_list(display_plan, limit=5)
        route_heading_gap = self.nav_debug.get('route_heading_gap_deg')
        agent_profile = dict(getattr(self, 'run_step_profile', {}) or {})
        agent_init_profile = dict(getattr(self, 'init_profile', {}) or {})
        self.latest_hud = {
            'command': None if command_model is None else int(command_model),
            'command_name': self.nav_debug.get('model_command_name'),
            'command_raw': command_raw_value,
            'command_raw_name': self.nav_debug.get('near_command_name'),
            'command_onehot': command_onehot,
            'local_command_xy': [float(x) for x in target_point] if target_point else [],
            'local_command_xy_next': [float(x) for x in target_point_next] if target_point_next else [],
            'predicted_traj': [[float(point[0]), float(point[1])] for point in display_plan],
            'predicted_traj_pad_xy': None,
            'predicted_traj_post_processor_xy': None,
            'use_trajectory_post_processor': False,
            'speed': float(tick_data.get('speed', 0.0)),
            'acceleration_xy': [float(x) for x in np.asarray(tick_data.get('acceleration', [0.0, 0.0]))[:2]],
            'steer': float(control.steer),
            'throttle': float(control.throttle),
            'brake': float(control.brake),
            'desired_speed': float(self.pid_metadata.get('desired_speed', 0.0)) if hasattr(self, 'pid_metadata') else 0.0,
            'near_node': near_node,
            'near_command_name': self.nav_debug.get('near_command_name'),
            'route_len': self.nav_debug.get('route_len'),
            'planner_route_world': self.nav_debug.get('planner_route_world') or [],
            'planner_route_source': 'agent_route',
            'route_head_world': self.nav_debug.get('route_head_world') or [],
            'route_head_distances': self.nav_debug.get('route_head_distances') or [],
            'target_point_angle_deg': self.nav_debug.get('target_point_angle_deg'),
            'route_heading_01_deg': self.nav_debug.get('route_heading_01_deg'),
            'route_heading_12_deg': self.nav_debug.get('route_heading_12_deg'),
            'route_heading_gap_deg': route_heading_gap,
            'pred_spat_head': pred_spat_head,
            'agent_profile': agent_profile,
            'agent_init_profile': agent_init_profile,
        }
        should_log_bend = route_heading_gap is not None and abs(float(route_heading_gap)) >= 8.0
        if should_log_bend and self.step - self._last_nav_debug_print_step >= 5:
            self._last_nav_debug_print_step = self.step
            print(
                '[HiPAD NAV DEBUG] '
                f'step={self.step} '
                f'route_head_world={self.latest_hud.get("route_head_world") or []} '
                f'route_head_distances={self.latest_hud.get("route_head_distances") or []} '
                f'target_point={self.latest_hud.get("local_command_xy") or []} '
                f'target_point_next={self.latest_hud.get("local_command_xy_next") or []} '
                f'target_point_angle_deg={self.latest_hud.get("target_point_angle_deg")} '
                f'route_heading_01_deg={self.latest_hud.get("route_heading_01_deg")} '
                f'route_heading_12_deg={self.latest_hud.get("route_heading_12_deg")} '
                f'route_heading_gap_deg={self.latest_hud.get("route_heading_gap_deg")} '
                f'pred_spat_head={self.latest_hud.get("pred_spat_head") or []}',
                flush=True,
            )
        return control


padAgent = hipadAdapterAgent
