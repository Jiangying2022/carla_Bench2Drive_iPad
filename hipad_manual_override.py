#!/usr/bin/env python3
import hipad_closed_loop as base


class KeyboardOverrideController:
    def __init__(self):
        self.control = base.base.carla.VehicleControl()
        self._steer_cache = 0.0
        self.auto_mode = True
        self._manual_keys = {
            base.base.pygame.K_UP,
            base.base.pygame.K_DOWN,
            base.base.pygame.K_LEFT,
            base.base.pygame.K_RIGHT,
            base.base.pygame.K_w,
            base.base.pygame.K_a,
            base.base.pygame.K_s,
            base.base.pygame.K_d,
            base.base.pygame.K_SPACE,
            base.base.pygame.K_q,
        }

    def _is_quit_shortcut(self, key: int) -> bool:
        return key == base.base.pygame.K_ESCAPE or (key == base.base.pygame.K_q and base.base.pygame.key.get_mods() & base.base.pygame.KMOD_CTRL)

    def _set_auto_mode(self, enabled: bool) -> None:
        self.auto_mode = enabled
        if enabled:
            self.control = base.base.carla.VehicleControl()
            self._steer_cache = 0.0
            base.base.LOG.info('切换为自动驾驶模式 (HiP-AD + PID)')
        else:
            base.base.LOG.info('切换为手动驾驶模式 (Keyboard)')

    def parse_events(self, milliseconds: int):
        should_quit = False
        for event in base.base.pygame.event.get():
            if event.type == base.base.pygame.QUIT:
                should_quit = True
            elif event.type == base.base.pygame.KEYDOWN:
                if self._is_quit_shortcut(event.key):
                    should_quit = True
                elif event.key in self._manual_keys and self.auto_mode:
                    self._set_auto_mode(False)
            elif event.type == base.base.pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    should_quit = True
                elif event.key == base.base.pygame.K_p:
                    self._set_auto_mode(not self.auto_mode)
                elif event.key == base.base.pygame.K_q and not (base.base.pygame.key.get_mods() & base.base.pygame.KMOD_CTRL):
                    self.control.gear = 1 if self.control.reverse else -1

        manual_control = None
        if not self.auto_mode:
            self._parse_vehicle_keys(base.base.pygame.key.get_pressed(), milliseconds)
            self.control.reverse = self.control.gear < 0
            manual_control = self.control
        return should_quit, manual_control

    def _parse_vehicle_keys(self, keys, milliseconds: int) -> None:
        if keys[base.base.pygame.K_UP] or keys[base.base.pygame.K_w]:
            self.control.throttle = min(self.control.throttle + 0.1, 1.0)
        else:
            self.control.throttle = 0.0

        if keys[base.base.pygame.K_DOWN] or keys[base.base.pygame.K_s]:
            self.control.brake = min(self.control.brake + 0.2, 1.0)
        else:
            self.control.brake = 0.0

        steer_increment = 5e-4 * milliseconds
        if keys[base.base.pygame.K_LEFT] or keys[base.base.pygame.K_a]:
            if self._steer_cache > 0:
                self._steer_cache = 0.0
            else:
                self._steer_cache -= steer_increment
        elif keys[base.base.pygame.K_RIGHT] or keys[base.base.pygame.K_d]:
            if self._steer_cache < 0:
                self._steer_cache = 0.0
            else:
                self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0

        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
        self.control.steer = round(self._steer_cache, 1)
        self.control.hand_brake = bool(keys[base.base.pygame.K_SPACE])


class ManualOverrideRunner(base.base.ClosedLoopRunner):
    def __init__(self, args):
        super().__init__(args)
        self.keyboard_controller = KeyboardOverrideController()
        self._manual_dt_ms = max(int(round(self.args.fixed_delta_seconds * 1000.0)), 1)
        self.display_camera = None
        self.display_latest_frame = None
        self.display_width = 1600
        self.display_height = 900
        self.bev_window_name = 'HiP-AD Manual Override BEV'
        self.bev_size = 900
        self.bev_window_enabled = True
        self.frame_profile = {}
        self._frame_profile_history = []

    @staticmethod
    def _route_draw_color():
        return (56, 140, 72)

    @staticmethod
    def _target_draw_color():
        return (136, 84, 156)

    @staticmethod
    def _trajectory_draw_color():
        return (220, 92, 92)

    def _get_debug_predicted_traj(self, hud):
        predicted_traj = []
        if hasattr(self.agent, 'pid_metadata'):
            predicted_traj = self.agent.pid_metadata.get('plan_spat') or self.agent.pid_metadata.get('plan_temp') or self.agent.pid_metadata.get('plan') or []
        if not predicted_traj:
            predicted_traj = hud.get('predicted_traj') or []
        return predicted_traj

    def _get_camera_intrinsic(self, image_width, image_height):
        focal = image_width / (2.0 * base.base.np.tan(base.base.np.deg2rad(self.args.record_video_fov) / 2.0))
        intrinsic = base.base.np.identity(3, dtype=base.base.np.float32)
        intrinsic[0, 0] = focal
        intrinsic[1, 1] = focal
        intrinsic[0, 2] = image_width / 2.0
        intrinsic[1, 2] = image_height / 2.0
        return intrinsic

    def _get_active_display_camera(self):
        if self.display_camera is not None and self.display_camera.is_alive:
            return self.display_camera
        if self.record_camera is not None and self.record_camera.is_alive:
            return self.record_camera
        return None

    def _get_display_source_size(self):
        active_camera = self._get_active_display_camera()
        if active_camera is self.record_camera and self.record_camera is not None:
            if self.record_latest_frame is not None:
                return int(self.record_latest_frame.shape[1]), int(self.record_latest_frame.shape[0])
            return int(self.args.record_video_width), int(self.args.record_video_height)
        if self.display_latest_frame is not None:
            return int(self.display_latest_frame.shape[1]), int(self.display_latest_frame.shape[0])
        return int(self.display_width), int(self.display_height)

    def _get_display_camera_intrinsic(self):
        source_width, source_height = self._get_display_source_size()
        return self._get_camera_intrinsic(source_width, source_height)

    def _project_world_to_image(self, location, camera_actor, image_width, image_height):
        if camera_actor is None or not camera_actor.is_alive:
            return None
        world_to_camera = base.base.np.array(camera_actor.get_transform().get_inverse_matrix(), dtype=base.base.np.float32)
        world_point = base.base.np.array([location.x, location.y, location.z, 1.0], dtype=base.base.np.float32)
        camera_point = world_to_camera @ world_point
        camera_point = base.base.np.array([camera_point[1], -camera_point[2], camera_point[0]], dtype=base.base.np.float32)
        depth = float(camera_point[2])
        if depth <= 0.1:
            return None
        intrinsic = self._get_camera_intrinsic(image_width, image_height)
        image_point = intrinsic @ camera_point
        pixel_x = float(image_point[0] / image_point[2])
        pixel_y = float(image_point[1] / image_point[2])
        if not base.base.np.isfinite(pixel_x) or not base.base.np.isfinite(pixel_y):
            return None
        if pixel_x < 0 or pixel_x >= image_width or pixel_y < 0 or pixel_y >= image_height:
            return None
        return (int(round(pixel_x)), int(round(pixel_y)))

    def _project_world_to_display(self, location):
        camera_actor = self._get_active_display_camera()
        if camera_actor is None:
            return None
        source_width, source_height = self._get_display_source_size()
        projected = self._project_world_to_image(location, camera_actor, source_width, source_height)
        if projected is None:
            return None
        if source_width == self.display_width and source_height == self.display_height:
            return projected
        scale_x = float(self.display_width) / max(float(source_width), 1.0)
        scale_y = float(self.display_height) / max(float(source_height), 1.0)
        return (
            int(round(projected[0] * scale_x)),
            int(round(projected[1] * scale_y)),
        )

    def _local_to_world_location(self, lateral, longitudinal, z_offset: float = 0.35):
        ego_transform = self.hero.get_transform()
        ego_location = ego_transform.location
        forward = ego_transform.get_forward_vector()
        right = ego_transform.get_right_vector()
        return base.base.carla.Location(
            x=ego_location.x + right.x * float(lateral) + forward.x * float(longitudinal),
            y=ego_location.y + right.y * float(lateral) + forward.y * float(longitudinal),
            z=ego_location.z + z_offset,
        )

    def _draw_projected_world_polyline(self, world_points, color, width: int = 3, point_radius: int = 0, point_limit: int | None = None) -> None:
        if self.hud_surface is None:
            return
        visible_points = []
        for index, location in enumerate(world_points):
            projected = self._project_world_to_display(location)
            visible_points.append(projected)
            if point_radius > 0 and projected is not None and (point_limit is None or index < point_limit):
                base.base.pygame.draw.circle(self.hud_surface, color, projected, point_radius)
        current_segment = []
        for projected in visible_points:
            if projected is None:
                if len(current_segment) >= 2:
                    base.base.pygame.draw.lines(self.hud_surface, color, False, current_segment, width)
                current_segment = []
                continue
            current_segment.append(projected)
        if len(current_segment) >= 2:
            base.base.pygame.draw.lines(self.hud_surface, color, False, current_segment, width)

    def _draw_projected_world_polyline_on_frame(self, frame, camera_actor, world_points, color_bgr, width: int = 3, point_radius: int = 0, point_limit: int | None = None):
        if frame is None:
            return frame
        image_height, image_width = frame.shape[:2]
        visible_points = []
        for index, location in enumerate(world_points):
            projected = self._project_world_to_image(location, camera_actor, image_width, image_height)
            visible_points.append(projected)
            if point_radius > 0 and projected is not None and (point_limit is None or index < point_limit):
                base.base.cv2.circle(frame, projected, point_radius, color_bgr, -1, base.base.cv2.LINE_AA)
        current_segment = []
        for projected in visible_points:
            if projected is None:
                if len(current_segment) >= 2:
                    base.base.cv2.polylines(frame, [base.base.np.array(current_segment, dtype=base.base.np.int32)], False, color_bgr, width, base.base.cv2.LINE_AA)
                current_segment = []
                continue
            current_segment.append(projected)
        if len(current_segment) >= 2:
            base.base.cv2.polylines(frame, [base.base.np.array(current_segment, dtype=base.base.np.int32)], False, color_bgr, width, base.base.cv2.LINE_AA)
        return frame

    def _draw_video_debug(self, frame, hud):
        if frame is None or self.hero is None or not self.hero.is_alive or self.record_camera is None or not self.record_camera.is_alive:
            return frame
        ego_location = self.hero.get_transform().location
        route_color = self._route_draw_color()
        target_color = self._target_draw_color()
        trajectory_color = self._trajectory_draw_color()
        route_color_bgr = (route_color[2], route_color[1], route_color[0])
        target_color_bgr = (target_color[2], target_color[1], target_color[0])
        trajectory_color_bgr = (trajectory_color[2], trajectory_color[1], trajectory_color[0])

        route_points = self._get_debug_route_points(ego_location, max_distance=45.0)
        dense_route_points = self._densify_route_locations([ego_location] + route_points) if route_points else []
        route_world_points = [
            base.base.carla.Location(x=point.x, y=point.y, z=point.z + 0.35)
            for point in dense_route_points
        ]
        self._draw_projected_world_polyline_on_frame(frame, self.record_camera, route_world_points, route_color_bgr, width=3)
        route_marker_points = [
            base.base.carla.Location(x=point.x, y=point.y, z=point.z + 0.35)
            for point in route_points[:12]
        ]
        self._draw_projected_world_polyline_on_frame(frame, self.record_camera, route_marker_points, route_color_bgr, width=1, point_radius=4, point_limit=12)

        command_xy = hud.get('local_command_xy')
        if command_xy and len(command_xy) >= 2:
            target_world_point = self._local_to_world_location(command_xy[0], command_xy[1], z_offset=0.35)
            target_anchor = self._local_to_world_location(0.0, 0.0, z_offset=0.35)
            self._draw_projected_world_polyline_on_frame(frame, self.record_camera, [target_anchor, target_world_point], target_color_bgr, width=3)
            projected_target = self._project_world_to_image(target_world_point, self.record_camera, frame.shape[1], frame.shape[0])
            if projected_target is not None:
                base.base.cv2.circle(frame, projected_target, 6, target_color_bgr, -1, base.base.cv2.LINE_AA)

        predicted_traj = self._get_debug_predicted_traj(hud)
        if predicted_traj:
            trajectory_world_points = [self._local_to_world_location(0.0, 0.0, z_offset=0.35)]
            for point in predicted_traj:
                trajectory_world_points.append(self._local_to_world_location(point[0], point[1], z_offset=0.35))
            self._draw_projected_world_polyline_on_frame(frame, self.record_camera, trajectory_world_points, trajectory_color_bgr, width=3, point_radius=4)
        return frame

    def _record_frame_profile(self, profile):
        sanitized = {key: round(float(value), 2) for key, value in profile.items()}
        self.frame_profile = sanitized
        self._frame_profile_history.append(sanitized)
        if len(self._frame_profile_history) > 30:
            self._frame_profile_history = self._frame_profile_history[-30:]

    def _get_average_frame_profile(self):
        if not self._frame_profile_history:
            return {}
        keys = self._frame_profile_history[0].keys()
        averaged = {}
        history_count = float(len(self._frame_profile_history))
        for key in keys:
            averaged[key] = round(sum(float(entry.get(key, 0.0)) for entry in self._frame_profile_history) / history_count, 2)
        return averaged

    def _build_grouped_overlay_lines(self, hud):
        lines = super()._build_grouped_overlay_lines(hud)
        mode_label = 'AUTO(HiP-AD+PID)' if self.keyboard_controller.auto_mode else 'MANUAL(KEYBOARD)'
        traffic_light_state = hud.get('traffic_light_state', 'NONE')
        frame_profile = hud.get('frame_profile') or {}
        frame_profile_avg = hud.get('frame_profile_avg') or {}
        agent_profile = hud.get('agent_profile') or {}
        agent_init_profile = hud.get('agent_init_profile') or {}
        lines = [line for line in lines if line != f'  traffic_light: {traffic_light_state}']
        debug_index = lines.index('DEBUG:') + 1 if 'DEBUG:' in lines else len(lines)
        debug_lines = [
            f'  mode: {mode_label}',
            f'  current_lane_tl: {traffic_light_state}',
            f'  route_head_world: {hud.get("route_head_world") or []}',
            f'  route_head_distances: {hud.get("route_head_distances") or []}',
            f'  target_point_next: {hud.get("local_command_xy_next") or []}',
            f'  target_point_angle_deg: {hud.get("target_point_angle_deg")}',
            f'  route_heading_01/12: {hud.get("route_heading_01_deg")} / {hud.get("route_heading_12_deg")}',
            f'  pred_spat_head: {hud.get("pred_spat_head") or []}',
            f'  frame_ms: {frame_profile.get("frame_ms")} avg30={frame_profile_avg.get("frame_ms")}',
            f'  tick/wrap/input: {frame_profile.get("tick_world_ms")} / {frame_profile.get("wrapper_ms")} / {frame_profile.get("input_ms")}',
            f'  hud/bev/video: {frame_profile.get("render_hud_ms")} / {frame_profile.get("render_bev_ms")} / {frame_profile.get("write_video_ms")}',
            f'  debug/apply/spec: {frame_profile.get("draw_debug_ms")} / {frame_profile.get("apply_control_ms")} / {frame_profile.get("spectator_ms")}',
            f'  agent tick/build/pipe: {agent_profile.get("tick_ms")} / {agent_profile.get("build_inputs_ms")} / {agent_profile.get("pipeline_ms")}',
            f'  agent collate/to/fw: {agent_profile.get("collate_unpack_ms")} / {agent_profile.get("to_device_ms")} / {agent_profile.get("forward_ms")}',
            f'  agent pid/io/vis: {agent_profile.get("pid_ms")} / {agent_profile.get("metric_io_ms")} / {agent_profile.get("visualize_ms")}',
            f'  agent init/fsolve: {agent_profile.get("init_ms")} / {agent_init_profile.get("fsolve_ms")}',
            '  hotkeys: WASD/ARROWS/SPACE/Q -> manual, P -> auto toggle, ESC -> quit',
        ]
        return lines[:debug_index] + debug_lines + lines[debug_index:]

    def _on_display_camera_frame(self, image: base.base.carla.Image) -> None:
        array = base.base.np.frombuffer(image.raw_data, dtype=base.base.np.uint8)
        array = array.reshape((image.height, image.width, 4))
        self.display_latest_frame = array[:, :, :3][:, :, ::-1].copy()

    def _on_record_video_frame(self, image: base.base.carla.Image) -> None:
        super()._on_record_video_frame(image)
        if self.record_latest_frame is None:
            return
        self.display_latest_frame = self.record_latest_frame[:, :, ::-1].copy()

    def _setup_display_camera(self) -> None:
        blueprint = self.world.get_blueprint_library().find('sensor.camera.rgb')
        blueprint.set_attribute('image_size_x', str(self.display_width))
        blueprint.set_attribute('image_size_y', str(self.display_height))
        blueprint.set_attribute('fov', str(self.args.record_video_fov))
        blueprint.set_attribute('sensor_tick', '0.0')

        camera_transform = base.base.carla.Transform(
            base.base.carla.Location(
                x=-self.args.record_camera_distance,
                y=0.0,
                z=self.args.record_camera_height,
            ),
            base.base.carla.Rotation(
                pitch=self.args.record_camera_pitch,
                yaw=0.0,
                roll=0.0,
            ),
        )
        self.display_camera = self.world.spawn_actor(
            blueprint,
            camera_transform,
            attach_to=self.hero,
            attachment_type=base.base.carla.AttachmentType.SpringArmGhost,
        )
        self.display_camera.listen(self._on_display_camera_frame)

    def setup(self) -> None:
        super().setup()
        if self.record_camera is not None and self.record_camera.is_alive:
            self.display_camera = self.record_camera
        else:
            self._setup_display_camera()
        try:
            base.base.cv2.namedWindow(self.bev_window_name, base.base.cv2.WINDOW_NORMAL)
            base.base.cv2.resizeWindow(self.bev_window_name, self.bev_size, self.bev_size)
        except Exception as exc:
            self.bev_window_enabled = False
            base.base.LOG.warning(f'OpenCV BEV window disabled: {exc}')

    def _get_debug_route_points(self, ego_location, max_distance: float = 45.0, max_points: int = 80):
        hud = getattr(self.agent, 'latest_hud', {}) if self.agent is not None else {}
        planner_route_world = hud.get('planner_route_world') or []
        route_points = []
        if planner_route_world:
            cumulative_distance = 0.0
            previous = ego_location
            for point in planner_route_world:
                if point is None or len(point) < 2:
                    continue
                location = base.base.carla.Location(
                    x=float(point[0]),
                    y=float(point[1]),
                    z=ego_location.z,
                )
                dx = location.x - ego_location.x
                dy = location.y - ego_location.y
                forward = self.hero.get_transform().get_forward_vector()
                longitudinal = dx * forward.x + dy * forward.y
                if longitudinal < -2.0:
                    continue
                cumulative_distance += location.distance(previous)
                route_points.append(location)
                previous = location
                if cumulative_distance >= max_distance or len(route_points) >= max_points:
                    break
            if route_points:
                return route_points
        return self._get_forward_route_locations(max_distance)

    def _draw_main_window_debug(self, hud) -> None:
        if self.hud_surface is None or self.hero is None or not self.hero.is_alive:
            return
        ego_location = self.hero.get_transform().location
        route_color = self._route_draw_color()
        target_color = self._target_draw_color()
        trajectory_color = self._trajectory_draw_color()

        route_points = self._get_debug_route_points(ego_location, max_distance=45.0)
        dense_route_points = self._densify_route_locations([ego_location] + route_points) if route_points else []
        route_world_points = [
            base.base.carla.Location(x=point.x, y=point.y, z=point.z + 0.35)
            for point in dense_route_points
        ]
        self._draw_projected_world_polyline(route_world_points, route_color, width=3)
        route_marker_points = [
            base.base.carla.Location(x=point.x, y=point.y, z=point.z + 0.35)
            for point in route_points[:12]
        ]
        self._draw_projected_world_polyline(route_marker_points, route_color, width=0, point_radius=4, point_limit=12)

        command_xy = hud.get('local_command_xy')
        if command_xy and len(command_xy) >= 2:
            target_world_point = self._local_to_world_location(command_xy[0], command_xy[1], z_offset=0.35)
            target_anchor = self._local_to_world_location(0.0, 0.0, z_offset=0.35)
            self._draw_projected_world_polyline([target_anchor, target_world_point], target_color, width=3)
            projected_target = self._project_world_to_display(target_world_point)
            if projected_target is not None:
                base.base.pygame.draw.circle(self.hud_surface, target_color, projected_target, 6)

        predicted_traj = self._get_debug_predicted_traj(hud)
        if predicted_traj:
            trajectory_world_points = [self._local_to_world_location(0.0, 0.0, z_offset=0.35)]
            for point in predicted_traj:
                trajectory_world_points.append(self._local_to_world_location(point[0], point[1], z_offset=0.35))
            self._draw_projected_world_polyline(trajectory_world_points, trajectory_color, width=3, point_radius=4)

    def _render_bev_window(self) -> None:
        if not self.bev_window_enabled:
            return
        if self.hero is None or not self.hero.is_alive:
            return
        hud = getattr(self.agent, 'latest_hud', {}) if self.agent is not None else {}
        image = base.base.np.full((self.bev_size, self.bev_size, 3), 18, dtype=base.base.np.uint8)
        center_x = self.bev_size // 2
        center_y = int(self.bev_size * 0.62)
        scale = 14.0

        ego_transform = self.hero.get_transform()
        ego_location = ego_transform.location
        forward = ego_transform.get_forward_vector()
        right = ego_transform.get_right_vector()

        def world_to_bev(location):
            dx = location.x - ego_location.x
            dy = location.y - ego_location.y
            longitudinal = dx * forward.x + dy * forward.y
            lateral = dx * right.x + dy * right.y
            return (int(center_x + lateral * scale), int(center_y - longitudinal * scale))

        def local_to_bev(lateral, longitudinal):
            return (int(center_x + float(lateral) * scale), int(center_y - float(longitudinal) * scale))

        route_color = self._route_draw_color()
        target_color = self._target_draw_color()
        trajectory_color = self._trajectory_draw_color()
        route_color_bgr = (route_color[2], route_color[1], route_color[0])
        target_color_bgr = (target_color[2], target_color[1], target_color[0])
        trajectory_color_bgr = (trajectory_color[2], trajectory_color[1], trajectory_color[0])

        for distance in [5, 10, 20, 30]:
            radius = int(distance * scale)
            base.base.cv2.circle(image, (center_x, center_y), radius, (45, 45, 45), 1, base.base.cv2.LINE_AA)

        route_points = self._get_debug_route_points(ego_location, max_distance=45.0)
        dense_route_points = self._densify_route_locations([ego_location] + route_points) if route_points else []
        preview_polyline = [world_to_bev(point) for point in dense_route_points]
        if len(preview_polyline) >= 2:
            base.base.cv2.polylines(image, [base.base.np.array(preview_polyline, dtype=base.base.np.int32)], False, route_color_bgr, 2, base.base.cv2.LINE_AA)
        for index, point in enumerate(route_points[:12]):
            bev_point = world_to_bev(point)
            base.base.cv2.circle(image, bev_point, 4 if index == 0 else 3, route_color_bgr, -1, base.base.cv2.LINE_AA)

        predicted_traj = self._get_debug_predicted_traj(hud)
        if predicted_traj:
            traj_points = [(center_x, center_y)]
            for point in predicted_traj:
                traj_points.append(local_to_bev(point[0], point[1]))
            if len(traj_points) >= 2:
                base.base.cv2.polylines(image, [base.base.np.array(traj_points, dtype=base.base.np.int32)], False, trajectory_color_bgr, 3, base.base.cv2.LINE_AA)
            for point in traj_points[1:]:
                base.base.cv2.circle(image, point, 4, trajectory_color_bgr, -1, base.base.cv2.LINE_AA)

        command_xy = hud.get('local_command_xy')
        if command_xy and len(command_xy) >= 2:
            cmd_point = local_to_bev(command_xy[0], command_xy[1])
            base.base.cv2.line(image, (center_x, center_y), cmd_point, target_color_bgr, 2, base.base.cv2.LINE_AA)
            base.base.cv2.circle(image, cmd_point, 6, target_color_bgr, -1, base.base.cv2.LINE_AA)
            base.base.cv2.putText(image, hud.get('command_name', 'cmd'), (cmd_point[0] + 8, cmd_point[1] - 8), base.base.cv2.FONT_HERSHEY_SIMPLEX, 0.45, target_color_bgr, 1, base.base.cv2.LINE_AA)

        base.base.cv2.circle(image, (center_x, center_y), 10, (0, 128, 255), -1, base.base.cv2.LINE_AA)
        ego_triangle = base.base.np.array([
            (center_x, center_y - 18),
            (center_x - 10, center_y + 12),
            (center_x + 10, center_y + 12),
        ], dtype=base.base.np.int32)
        base.base.cv2.fillConvexPoly(image, ego_triangle, (0, 128, 255), lineType=base.base.cv2.LINE_AA)

        mode_label = 'AUTO(HiP-AD+PID)' if self.keyboard_controller.auto_mode else 'MANUAL(KEYBOARD)'
        base.base.cv2.putText(image, mode_label, (20, 36), base.base.cv2.FONT_HERSHEY_SIMPLEX, 0.8, (230, 230, 230), 2, base.base.cv2.LINE_AA)
        base.base.cv2.putText(image, 'route deque=green  target_point=purple  model traj=red', (20, 70), base.base.cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 180, 180), 1, base.base.cv2.LINE_AA)

        try:
            base.base.cv2.imshow(self.bev_window_name, image)
            base.base.cv2.waitKey(1)
        except Exception as exc:
            self.bev_window_enabled = False
            base.base.LOG.warning(f'OpenCV BEV window disabled during rendering: {exc}')

    def _draw_predicted_trajectory(self) -> None:
        if self.hero is None or not self.hero.is_alive or self.world is None or self.agent is None:
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
        debug = self.world.debug
        trajectory_color = base.base.carla.Color(255, 0, 0)
        for index, (lateral, longitudinal) in enumerate(local_plan):
            world_point = base.base.carla.Location(
                x=ego_location.x + right.x * lateral + forward.x * longitudinal,
                y=ego_location.y + right.y * lateral + forward.y * longitudinal,
                z=ego_location.z + 0.35,
            )
            point_size = max(0.055, 0.085 - index * 0.005)
            debug.draw_point(
                world_point,
                size=point_size,
                color=trajectory_color,
                life_time=0.06,
                persistent_lines=False,
            )
            debug.draw_point(
                world_point + base.base.carla.Location(z=0.02),
                size=max(0.04, point_size * 0.70),
                color=trajectory_color,
                life_time=0.06,
                persistent_lines=False,
            )

    def _draw_video_overlay(self, frame, hud):
        overlay_frame = super()._draw_video_overlay(frame, hud)
        return self._draw_video_debug(overlay_frame, hud)

    def _render_hud(self) -> None:
        if not self.hud_enabled or self.agent is None:
            return
        hud = getattr(self.agent, 'latest_hud', {})
        if self.hud_surface is None or self.hud_font is None:
            return
        try:
            self.hud_surface.fill((8, 8, 8))
            camera_rect = base.base.pygame.Rect(0, 0, self.display_width, self.display_height)
            panel_rect = base.base.pygame.Rect(self.display_width, 0, 520, self.display_height)
            base.base.pygame.draw.rect(self.hud_surface, (12, 12, 12), camera_rect)
            base.base.pygame.draw.rect(self.hud_surface, (20, 20, 20), panel_rect)

            if self.display_latest_frame is not None:
                frame_surface = base.base.pygame.surfarray.make_surface(base.base.np.swapaxes(self.display_latest_frame, 0, 1))
                if frame_surface.get_size() != (self.display_width, self.display_height):
                    frame_surface = base.base.pygame.transform.smoothscale(frame_surface, (self.display_width, self.display_height))
                self.hud_surface.blit(frame_surface, (0, 0))
                self._draw_main_window_debug(hud)
            else:
                waiting_surface = self.hud_font.render('Waiting for third-person camera frame...', True, (255, 255, 255))
                self.hud_surface.blit(waiting_surface, (40, 40))

            lines = self._build_grouped_overlay_lines(hud)
            y = 30
            for line in lines:
                text_surface = self.hud_font.render(line, True, (255, 255, 255))
                self.hud_surface.blit(text_surface, (self.display_width + 12, y))
                y += 28
            base.base.pygame.display.flip()
        except Exception as exc:
            base.base.LOG.warning(f'HUD render failed, disabling HUD: {exc}')
            self.hud_enabled = False
            try:
                base.base.pygame.display.quit()
            except Exception:
                pass
            self.hud_surface = None
            self.hud_font = None

    def _refresh_global_route(self, gps_route, route) -> None:
        self.destination_transform = route[-1][0]
        self.route_world_transforms = [wt for wt, _ in route]
        self.route_preview_index = 0
        self.agent.set_global_plan(gps_route, route)
        if getattr(self.agent, 'initialized', False) and hasattr(self.agent, '_route_planner'):
            if getattr(self.agent, 'save_name', None) == 'standalone':
                route_world = self.agent._global_plan_world_coord_dense if getattr(self.agent, '_global_plan_world_coord_dense', None) is not None else self.agent._global_plan_world_coord
                self.agent._route_planner.set_route(route_world, False)
            else:
                self.agent._route_planner.set_route(self.agent._global_plan, True)

    def _extend_destination_along_route(self) -> bool:
        if self.hero is None or not self.hero.is_alive or self.world is None or self.destination_transform is None:
            return False
        world_map = self.world.get_map()
        hero_transform = self.hero.get_transform()
        hero_waypoint = world_map.get_waypoint(
            hero_transform.location,
            project_to_road=True,
            lane_type=base.base.carla.LaneType.Driving,
        )
        destination_waypoint = world_map.get_waypoint(
            self.destination_transform.location,
            project_to_road=True,
            lane_type=base.base.carla.LaneType.Driving,
        )
        if hero_waypoint is None or destination_waypoint is None:
            return False
        extension_distance = max(self.args.stop_distance * 4.0, 30.0)
        next_waypoints = destination_waypoint.next(extension_distance)
        target_waypoint = next_waypoints[0] if next_waypoints else None
        if target_waypoint is None:
            hero_location = hero_transform.location
            destination_location = destination_waypoint.transform.location
            dx = destination_location.x - hero_location.x
            dy = destination_location.y - hero_location.y
            norm = (dx * dx + dy * dy) ** 0.5
            if norm < 1e-6:
                forward = destination_waypoint.transform.get_forward_vector()
                dx = forward.x
                dy = forward.y
                norm = max((dx * dx + dy * dy) ** 0.5, 1e-6)
            target_location = base.base.carla.Location(
                x=destination_location.x + dx / norm * extension_distance,
                y=destination_location.y + dy / norm * extension_distance,
                z=destination_location.z,
            )
            target_waypoint = world_map.get_waypoint(
                target_location,
                project_to_road=True,
                lane_type=base.base.carla.LaneType.Driving,
            )
        if target_waypoint is None:
            return False
        gps_route, route = base.base.interpolate_trajectory([
            hero_waypoint.transform.location,
            target_waypoint.transform.location,
        ])
        if not route:
            return False
        first_option = route[0][1]
        ego_gps = base.base._location_to_gps(*base.base._get_latlon_ref(self.world), hero_waypoint.transform.location)
        route[0] = (hero_waypoint.transform, first_option)
        gps_route[0] = (ego_gps, first_option)
        self._refresh_global_route(gps_route, route)
        base.base.LOG.info(
            'destination extended to (%.2f, %.2f), route points=%d',
            self.destination_transform.location.x,
            self.destination_transform.location.y,
            len(route),
        )
        return True

    def run(self) -> None:
        start_time = base.base.time.time()
        for step in range(self.args.max_steps):
            frame_start = base.base.time.perf_counter()
            frame_profile = {}

            section_start = base.base.time.perf_counter()
            self._tick_world()
            frame_profile['tick_world_ms'] = (base.base.time.perf_counter() - section_start) * 1000.0

            section_start = base.base.time.perf_counter()
            self._update_spectator()
            frame_profile['spectator_ms'] = (base.base.time.perf_counter() - section_start) * 1000.0
            if self.agent.hero_actor is None or not self.agent.hero_actor.is_alive:
                self.agent.hero_actor = self.hero

            section_start = base.base.time.perf_counter()
            should_quit, manual_control = self.keyboard_controller.parse_events(self._manual_dt_ms)
            frame_profile['input_ms'] = (base.base.time.perf_counter() - section_start) * 1000.0
            if should_quit:
                base.base.LOG.info('用户退出')
                break

            section_start = base.base.time.perf_counter()
            auto_control = self.wrapper()
            frame_profile['wrapper_ms'] = (base.base.time.perf_counter() - section_start) * 1000.0
            control = auto_control if self.keyboard_controller.auto_mode else manual_control
            if control is None:
                control = auto_control

            hud_debug = getattr(self.agent, 'latest_hud', {})
            hud_debug['control_mode'] = 'AUTO(HiP-AD+PID)' if self.keyboard_controller.auto_mode else 'MANUAL(KEYBOARD)'
            hud_debug['steer'] = float(control.steer)
            hud_debug['throttle'] = float(control.throttle)
            hud_debug['brake'] = float(control.brake)

            traffic_light_state, traffic_light_actor_id = self._get_current_lane_traffic_light_state()
            hud_debug['traffic_light_state'] = traffic_light_state
            hud_debug['traffic_light_actor_id'] = traffic_light_actor_id

            section_start = base.base.time.perf_counter()
            self._draw_predicted_trajectory()
            frame_profile['draw_debug_ms'] = (base.base.time.perf_counter() - section_start) * 1000.0

            section_start = base.base.time.perf_counter()
            self._render_hud()
            frame_profile['render_hud_ms'] = (base.base.time.perf_counter() - section_start) * 1000.0

            section_start = base.base.time.perf_counter()
            self._render_bev_window()
            frame_profile['render_bev_ms'] = (base.base.time.perf_counter() - section_start) * 1000.0

            section_start = base.base.time.perf_counter()
            self._write_video_frame()
            frame_profile['write_video_ms'] = (base.base.time.perf_counter() - section_start) * 1000.0

            section_start = base.base.time.perf_counter()
            self.hero.apply_control(control)
            frame_profile['apply_control_ms'] = (base.base.time.perf_counter() - section_start) * 1000.0
            frame_profile['frame_ms'] = (base.base.time.perf_counter() - frame_start) * 1000.0
            self._record_frame_profile(frame_profile)
            hud_debug['frame_profile'] = self.frame_profile
            hud_debug['frame_profile_avg'] = self._get_average_frame_profile()

            if step % self.args.log_every == 0:
                hero_transform = self.hero.get_transform()
                location = hero_transform.location
                velocity = self.hero.get_velocity()
                speed = (velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2) ** 0.5
                remaining = self.destination_transform.location.distance(location)
                profile_avg = self._get_average_frame_profile()
                agent_profile = hud_debug.get('agent_profile') or {}
                agent_init_profile = hud_debug.get('agent_init_profile') or {}
                base.base.LOG.info(
                    'step=%d mode=%s location=(%.2f, %.2f) speed=%.2f remaining=%.2f control=(%.3f, %.3f, %.3f)',
                    step,
                    hud_debug.get('control_mode'),
                    location.x,
                    location.y,
                    speed,
                    remaining,
                    control.steer,
                    control.throttle,
                    control.brake,
                )
                base.base.LOG.info(
                    'profile_ms step=%d frame=%.2f avg30=%.2f tick=%.2f spectator=%.2f input=%.2f wrapper=%.2f draw=%.2f hud=%.2f bev=%.2f video=%.2f apply=%.2f',
                    step,
                    self.frame_profile.get('frame_ms', 0.0),
                    profile_avg.get('frame_ms', 0.0),
                    self.frame_profile.get('tick_world_ms', 0.0),
                    self.frame_profile.get('spectator_ms', 0.0),
                    self.frame_profile.get('input_ms', 0.0),
                    self.frame_profile.get('wrapper_ms', 0.0),
                    self.frame_profile.get('draw_debug_ms', 0.0),
                    self.frame_profile.get('render_hud_ms', 0.0),
                    self.frame_profile.get('render_bev_ms', 0.0),
                    self.frame_profile.get('write_video_ms', 0.0),
                    self.frame_profile.get('apply_control_ms', 0.0),
                )
                base.base.LOG.info(
                    'agent_profile_ms step=%d total=%.2f tick=%.2f build=%.2f pipeline=%.2f collate=%.2f to_device=%.2f forward=%.2f pid=%.2f metric_io=%.2f visualize=%.2f init=%.2f fsolve=%.2f',
                    step,
                    agent_profile.get('run_step_total_ms', 0.0),
                    agent_profile.get('tick_ms', 0.0),
                    agent_profile.get('build_inputs_ms', 0.0),
                    agent_profile.get('pipeline_ms', 0.0),
                    agent_profile.get('collate_unpack_ms', 0.0),
                    agent_profile.get('to_device_ms', 0.0),
                    agent_profile.get('forward_ms', 0.0),
                    agent_profile.get('pid_ms', 0.0),
                    agent_profile.get('metric_io_ms', 0.0),
                    agent_profile.get('visualize_ms', 0.0),
                    agent_profile.get('init_ms', 0.0),
                    agent_init_profile.get('fsolve_ms', 0.0),
                )

            if self.hero.get_transform().location.distance(self.destination_transform.location) <= self.args.stop_distance:
                base.base.LOG.info('destination reached, extending route forward')
                if self._extend_destination_along_route():
                    continue
                base.base.LOG.warning('failed to extend destination, keeping current route')

            if base.base.time.time() - start_time >= self.args.max_runtime_seconds:
                base.base.LOG.info('max runtime reached')
                break

    def _get_current_lane_traffic_light_state(self):
        traffic_light_state = 'NONE'
        traffic_light_actor_id = None
        try:
            if self.hero is not None and self.hero.is_alive and self.hero.is_at_traffic_light():
                traffic_light = self.hero.get_traffic_light()
                if traffic_light is not None:
                    traffic_light_actor_id = traffic_light.id
                    traffic_light_state = traffic_light.get_state().name
                else:
                    traffic_light_state = self.hero.get_traffic_light_state().name
        except Exception:
            pass
        return traffic_light_state, traffic_light_actor_id

    def cleanup(self) -> None:
        if self.display_camera is not None and self.display_camera is not self.record_camera:
            try:
                self.display_camera.stop()
            except Exception:
                pass
            try:
                self.display_camera.destroy()
            except Exception:
                pass
        self.display_camera = None
        self.display_latest_frame = None
        try:
            base.base.cv2.destroyWindow(self.bev_window_name)
        except Exception:
            pass
        super().cleanup()


def main() -> None:
    args = base.parse_args()
    if args.disable_hud:
        raise RuntimeError('hipad_manual_override.py 依赖 pygame HUD 接收键盘输入，请不要使用 --disable-hud')
    if not base.base.os.path.exists(args.checkpoint):
        raise FileNotFoundError(f'checkpoint not found: {args.checkpoint}')
    if not base.base.os.path.exists(args.config):
        raise FileNotFoundError(f'config not found: {args.config}')
    if args.route_id is not None and not base.base.os.path.exists(args.routes):
        raise FileNotFoundError(f'routes xml not found: {args.routes}')

    runner = ManualOverrideRunner(args)
    try:
        runner.setup()
        runner.run()
    except KeyboardInterrupt:
        base.base.LOG.info('用户中断')
    finally:
        runner.cleanup()


if __name__ == '__main__':
    main()
