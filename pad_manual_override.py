#!/usr/bin/env python3
import pad_closed_loop as base


class KeyboardOverrideController:
    def __init__(self):
        self.control = base.carla.VehicleControl()
        self._steer_cache = 0.0
        self.auto_mode = True
        self._manual_keys = {
            base.pygame.K_UP,
            base.pygame.K_DOWN,
            base.pygame.K_LEFT,
            base.pygame.K_RIGHT,
            base.pygame.K_w,
            base.pygame.K_a,
            base.pygame.K_s,
            base.pygame.K_d,
            base.pygame.K_SPACE,
            base.pygame.K_q,
        }

    def _is_quit_shortcut(self, key: int) -> bool:
        return key == base.pygame.K_ESCAPE or (key == base.pygame.K_q and base.pygame.key.get_mods() & base.pygame.KMOD_CTRL)

    def _set_auto_mode(self, enabled: bool) -> None:
        self.auto_mode = enabled
        if enabled:
            self.control = base.carla.VehicleControl()
            self._steer_cache = 0.0
            base.LOG.info('切换为自动驾驶模式 (PAD + PID)')
        else:
            base.LOG.info('切换为手动驾驶模式 (Keyboard)')

    def parse_events(self, milliseconds: int):
        should_quit = False
        for event in base.pygame.event.get():
            if event.type == base.pygame.QUIT:
                should_quit = True
            elif event.type == base.pygame.KEYDOWN:
                if self._is_quit_shortcut(event.key):
                    should_quit = True
                elif event.key in self._manual_keys and self.auto_mode:
                    self._set_auto_mode(False)
            elif event.type == base.pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    should_quit = True
                elif event.key == base.pygame.K_p:
                    self._set_auto_mode(not self.auto_mode)
                elif event.key == base.pygame.K_q and not (base.pygame.key.get_mods() & base.pygame.KMOD_CTRL):
                    self.control.gear = 1 if self.control.reverse else -1

        manual_control = None
        if not self.auto_mode:
            self._parse_vehicle_keys(base.pygame.key.get_pressed(), milliseconds)
            self.control.reverse = self.control.gear < 0
            manual_control = self.control
        return should_quit, manual_control

    def _parse_vehicle_keys(self, keys, milliseconds: int) -> None:
        if keys[base.pygame.K_UP] or keys[base.pygame.K_w]:
            self.control.throttle = min(self.control.throttle + 0.1, 1.0)
        else:
            self.control.throttle = 0.0

        if keys[base.pygame.K_DOWN] or keys[base.pygame.K_s]:
            self.control.brake = min(self.control.brake + 0.2, 1.0)
        else:
            self.control.brake = 0.0

        steer_increment = 5e-4 * milliseconds
        if keys[base.pygame.K_LEFT] or keys[base.pygame.K_a]:
            if self._steer_cache > 0:
                self._steer_cache = 0.0
            else:
                self._steer_cache -= steer_increment
        elif keys[base.pygame.K_RIGHT] or keys[base.pygame.K_d]:
            if self._steer_cache < 0:
                self._steer_cache = 0.0
            else:
                self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0

        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
        self.control.steer = round(self._steer_cache, 1)
        self.control.hand_brake = bool(keys[base.pygame.K_SPACE])


class ManualOverrideRunner(base.ClosedLoopRunner):
    def __init__(self, args):
        super().__init__(args)
        self.keyboard_controller = KeyboardOverrideController()
        self._manual_dt_ms = max(int(round(self.args.fixed_delta_seconds * 1000.0)), 1)
        self.display_camera = None
        self.display_latest_frame = None
        self.display_width = 1600
        self.display_height = 900
        self.bev_window_name = 'PAD Manual Override BEV'
        self.bev_size = 900
        self.bev_window_enabled = True

    def _build_grouped_overlay_lines(self, hud):
        lines = super()._build_grouped_overlay_lines(hud)
        mode_label = 'AUTO(PAD+PID)' if self.keyboard_controller.auto_mode else 'MANUAL(KEYBOARD)'
        traffic_light_state = hud.get('traffic_light_state', 'NONE')
        lines = [line for line in lines if line != f'  traffic_light: {traffic_light_state}']
        debug_index = lines.index('DEBUG:') + 1 if 'DEBUG:' in lines else len(lines)
        debug_lines = [
            f'  mode: {mode_label}',
            f'  current_lane_tl: {traffic_light_state}',
            '  hotkeys: WASD/ARROWS/SPACE/Q -> manual, P -> auto toggle, ESC -> quit',
        ]
        return lines[:debug_index] + debug_lines + lines[debug_index:]

    def _on_display_camera_frame(self, image: base.carla.Image) -> None:
        array = base.np.frombuffer(image.raw_data, dtype=base.np.uint8)
        array = array.reshape((image.height, image.width, 4))
        self.display_latest_frame = array[:, :, :3][:, :, ::-1].copy()

    def _setup_display_camera(self) -> None:
        blueprint = self.world.get_blueprint_library().find('sensor.camera.rgb')
        blueprint.set_attribute('image_size_x', str(self.display_width))
        blueprint.set_attribute('image_size_y', str(self.display_height))
        blueprint.set_attribute('fov', str(self.args.record_video_fov))
        blueprint.set_attribute('sensor_tick', '0.0')

        camera_transform = base.carla.Transform(
            base.carla.Location(
                x=-self.args.record_camera_distance,
                y=0.0,
                z=self.args.record_camera_height,
            ),
            base.carla.Rotation(
                pitch=self.args.record_camera_pitch,
                yaw=0.0,
                roll=0.0,
            ),
        )
        self.display_camera = self.world.spawn_actor(
            blueprint,
            camera_transform,
            attach_to=self.hero,
            attachment_type=base.carla.AttachmentType.SpringArmGhost,
        )
        self.display_camera.listen(self._on_display_camera_frame)

    def setup(self) -> None:
        super().setup()
        self._setup_display_camera()
        try:
            base.cv2.namedWindow(self.bev_window_name, base.cv2.WINDOW_NORMAL)
            base.cv2.resizeWindow(self.bev_window_name, self.bev_size, self.bev_size)
        except Exception as exc:
            self.bev_window_enabled = False
            base.LOG.warning(f'OpenCV BEV window disabled: {exc}')

    def _render_bev_window(self) -> None:
        if not self.bev_window_enabled:
            return
        if self.hero is None or not self.hero.is_alive:
            return
        hud = getattr(self.agent, 'latest_hud', {}) if self.agent is not None else {}
        image = base.np.full((self.bev_size, self.bev_size, 3), 18, dtype=base.np.uint8)
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

        for distance in [5, 10, 20, 30]:
            radius = int(distance * scale)
            base.cv2.circle(image, (center_x, center_y), radius, (45, 45, 45), 1, base.cv2.LINE_AA)

        preview_points = self._get_forward_route_locations(40)
        dense_preview_points = self._densify_route_locations([ego_location] + preview_points)
        preview_polyline = [world_to_bev(point) for point in dense_preview_points]
        if len(preview_polyline) >= 2:
            base.cv2.polylines(image, [base.np.array(preview_polyline, dtype=base.np.int32)], False, (0, 255, 0), 3, base.cv2.LINE_AA)

        future_points = self._get_future_route_points(5)
        for index, point in enumerate(future_points):
            bev_point = world_to_bev(point)
            base.cv2.circle(image, bev_point, 5, (0, 255, 0), -1, base.cv2.LINE_AA)
            base.cv2.putText(image, f'wp{index + 1}', (bev_point[0] + 8, bev_point[1] - 8), base.cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, base.cv2.LINE_AA)

        near_node = hud.get('near_node')
        if near_node and len(near_node) >= 2:
            near_point = world_to_bev(base.carla.Location(x=float(near_node[0]), y=float(near_node[1]), z=ego_location.z))
            base.cv2.line(image, (center_x, center_y), near_point, (200, 0, 255), 2, base.cv2.LINE_AA)
            base.cv2.circle(image, near_point, 6, (200, 0, 255), -1, base.cv2.LINE_AA)

        predicted_traj = hud.get('predicted_traj') or []
        if predicted_traj:
            traj_points = [(center_x, center_y)]
            for point in predicted_traj:
                traj_points.append(local_to_bev(point[0], point[1]))
            if len(traj_points) >= 2:
                base.cv2.polylines(image, [base.np.array(traj_points, dtype=base.np.int32)], False, (255, 64, 64), 3, base.cv2.LINE_AA)
            for point in traj_points[1:]:
                base.cv2.circle(image, point, 4, (255, 64, 64), -1, base.cv2.LINE_AA)

        command_xy = hud.get('local_command_xy')
        if command_xy and len(command_xy) >= 2:
            cmd_point = local_to_bev(command_xy[0], command_xy[1])
            base.cv2.line(image, (center_x, center_y), cmd_point, (255, 255, 0), 2, base.cv2.LINE_AA)
            base.cv2.circle(image, cmd_point, 6, (255, 255, 0), -1, base.cv2.LINE_AA)

        base.cv2.circle(image, (center_x, center_y), 10, (0, 128, 255), -1, base.cv2.LINE_AA)
        ego_triangle = base.np.array([
            (center_x, center_y - 18),
            (center_x - 10, center_y + 12),
            (center_x + 10, center_y + 12),
        ], dtype=base.np.int32)
        base.cv2.fillConvexPoly(image, ego_triangle, (0, 128, 255), lineType=base.cv2.LINE_AA)

        mode_label = 'AUTO(PAD+PID)' if self.keyboard_controller.auto_mode else 'MANUAL(KEYBOARD)'
        base.cv2.putText(image, mode_label, (20, 36), base.cv2.FONT_HERSHEY_SIMPLEX, 0.8, (230, 230, 230), 2, base.cv2.LINE_AA)
        base.cv2.putText(image, 'BEV Debug Window', (20, 70), base.cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, base.cv2.LINE_AA)

        try:
            base.cv2.imshow(self.bev_window_name, image)
            base.cv2.waitKey(1)
        except Exception as exc:
            self.bev_window_enabled = False
            base.LOG.warning(f'OpenCV BEV window disabled during rendering: {exc}')

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
        trajectory_color = base.carla.Color(255, 0, 0)
        for index, (lateral, longitudinal) in enumerate(local_plan):
            world_point = base.carla.Location(
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
                world_point + base.carla.Location(z=0.02),
                size=max(0.04, point_size * 0.70),
                color=trajectory_color,
                life_time=0.06,
                persistent_lines=False,
            )

    def _render_hud(self) -> None:
        if not self.hud_enabled or self.agent is None:
            return
        hud = getattr(self.agent, 'latest_hud', {})
        if self.hud_surface is None or self.hud_font is None:
            return
        try:
            self.hud_surface.fill((8, 8, 8))
            camera_rect = base.pygame.Rect(0, 0, self.display_width, self.display_height)
            panel_rect = base.pygame.Rect(self.display_width, 0, 520, self.display_height)
            base.pygame.draw.rect(self.hud_surface, (12, 12, 12), camera_rect)
            base.pygame.draw.rect(self.hud_surface, (20, 20, 20), panel_rect)

            if self.display_latest_frame is not None:
                frame_surface = base.pygame.surfarray.make_surface(base.np.swapaxes(self.display_latest_frame, 0, 1))
                if frame_surface.get_size() != (self.display_width, self.display_height):
                    frame_surface = base.pygame.transform.smoothscale(frame_surface, (self.display_width, self.display_height))
                self.hud_surface.blit(frame_surface, (0, 0))
            else:
                waiting_surface = self.hud_font.render('Waiting for third-person camera frame...', True, (255, 255, 255))
                self.hud_surface.blit(waiting_surface, (40, 40))

            lines = self._build_grouped_overlay_lines(hud)
            y = 30
            for line in lines:
                text_surface = self.hud_font.render(line, True, (255, 255, 255))
                self.hud_surface.blit(text_surface, (self.display_width + 12, y))
                y += 28
            base.pygame.display.flip()
        except Exception as exc:
            base.LOG.warning(f'HUD render failed, disabling HUD: {exc}')
            self.hud_enabled = False
            try:
                base.pygame.display.quit()
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
            if getattr(self.agent, 'standalone_mode', False):
                route_world = self.agent._global_plan_world_coord_dense if self.agent._global_plan_world_coord_dense is not None else self.agent._global_plan_world_coord
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
            lane_type=base.carla.LaneType.Driving,
        )
        destination_waypoint = world_map.get_waypoint(
            self.destination_transform.location,
            project_to_road=True,
            lane_type=base.carla.LaneType.Driving,
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
            target_location = base.carla.Location(
                x=destination_location.x + dx / norm * extension_distance,
                y=destination_location.y + dy / norm * extension_distance,
                z=destination_location.z,
            )
            target_waypoint = world_map.get_waypoint(
                target_location,
                project_to_road=True,
                lane_type=base.carla.LaneType.Driving,
            )
        if target_waypoint is None:
            return False
        gps_route, route = base.interpolate_trajectory([
            hero_waypoint.transform.location,
            target_waypoint.transform.location,
        ])
        if not route:
            return False
        first_option = route[0][1]
        ego_gps = base._location_to_gps(*base._get_latlon_ref(self.world), hero_waypoint.transform.location)
        route[0] = (hero_waypoint.transform, first_option)
        gps_route[0] = (ego_gps, first_option)
        self._refresh_global_route(gps_route, route)
        base.LOG.info(
            'destination extended to (%.2f, %.2f), route points=%d',
            self.destination_transform.location.x,
            self.destination_transform.location.y,
            len(route),
        )
        return True
    def run(self) -> None:
        start_time = base.time.time()
        for step in range(self.args.max_steps):
            self._tick_world()
            self._update_spectator()
            if self.agent.hero_actor is None or not self.agent.hero_actor.is_alive:
                self.agent.hero_actor = self.hero

            should_quit, manual_control = self.keyboard_controller.parse_events(self._manual_dt_ms)
            if should_quit:
                base.LOG.info('用户退出')
                break

            auto_control = self.wrapper()
            control = auto_control if self.keyboard_controller.auto_mode else manual_control
            if control is None:
                control = auto_control

            hud_debug = getattr(self.agent, 'latest_hud', {})
            hud_debug['control_mode'] = 'AUTO(PAD+PID)' if self.keyboard_controller.auto_mode else 'MANUAL(KEYBOARD)'
            hud_debug['steer'] = float(control.steer)
            hud_debug['throttle'] = float(control.throttle)
            hud_debug['brake'] = float(control.brake)

            traffic_light_state, traffic_light_actor_id = self._get_current_lane_traffic_light_state()
            hud_debug['traffic_light_state'] = traffic_light_state
            hud_debug['traffic_light_actor_id'] = traffic_light_actor_id

            self._draw_predicted_trajectory()
            self._render_hud()
            self._render_bev_window()
            self._write_video_frame()
            self.hero.apply_control(control)

            if step % self.args.log_every == 0:
                hero_transform = self.hero.get_transform()
                location = hero_transform.location
                velocity = self.hero.get_velocity()
                speed = (velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2) ** 0.5
                remaining = self.destination_transform.location.distance(location)
                base.LOG.info(
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

            if self.hero.get_transform().location.distance(self.destination_transform.location) <= self.args.stop_distance:
                base.LOG.info('destination reached, extending route forward')
                if self._extend_destination_along_route():
                    continue
                base.LOG.warning('failed to extend destination, keeping current route')

            if base.time.time() - start_time >= self.args.max_runtime_seconds:
                base.LOG.info('max runtime reached')
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
        if self.display_camera is not None:
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
            base.cv2.destroyWindow(self.bev_window_name)
        except Exception:
            pass
        super().cleanup()


def main() -> None:
    args = base.parse_args()
    if args.disable_hud:
        raise RuntimeError('pad_manual_override.py 依赖 pygame HUD 接收键盘输入，请不要使用 --disable-hud')
    if not base.os.path.exists(args.checkpoint):
        raise FileNotFoundError(f'checkpoint not found: {args.checkpoint}')
    if not base.os.path.exists(args.config):
        raise FileNotFoundError(f'config not found: {args.config}')
    if args.route_id is not None and not base.os.path.exists(args.routes):
        raise FileNotFoundError(f'routes xml not found: {args.routes}')

    runner = ManualOverrideRunner(args)
    try:
        runner.setup()
        runner.run()
    except KeyboardInterrupt:
        base.LOG.info('用户中断')
    finally:
        runner.cleanup()


if __name__ == '__main__':
    main()
