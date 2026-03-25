from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


# 本文件是面向 PAD 未来 4 秒轨迹的简化后处理器。
# 默认输入已经在外部完成坐标变换：
# PAD 原始输出为 (横向, 纵向)，传入本文件前应先变成 (纵向, 横向)。
# 本文件内部不再做重采样，输入几个点，输出就保持几个点。
DEFAULT_TIME_INTERVAL = 0.5


@dataclass(slots=True)
class TrajectoryPoint:
    # 注意：本文件统一采用 x=纵向、y=横向 的坐标定义。
    # 这与 PAD -> PID 接入时的局部轨迹解释保持一致。
    x: float = 0.0
    y: float = 0.0
    velocity: float = 0.0
    acc: float = 0.0
    cur: float = 0.0
    yaw: float = 0.0
    timestamp: float = 0.0

    def as_tuple(self) -> tuple[float, float, float, float, float, float, float]:
        return (
            self.x,
            self.y,
            self.velocity,
            self.acc,
            self.cur,
            self.yaw,
            self.timestamp,
        )


class TrajectoryPostProcessor:
    # 原始输入轨迹为 PAD 的未来离散点：
    #   P0 ... PN, shape=(N, 2), 单点格式为 (x, y)
    # 其中 x=纵向、y=横向。
    # 本类不负责坐标交换，只处理已经变换好的轨迹。
    def __init__(self, raw_trajectory_points: Sequence[Sequence[float]]) -> None:
        self.raw_trajectory_points_ = self._prepare_input_points(raw_trajectory_points)

    def compute_path_profile(self) -> list[TrajectoryPoint]:
        if len(self.raw_trajectory_points_) < 2:
            return []

        # Step 1. 计算累计里程 s。
        # 这里的 s 是沿轨迹前进方向累积的弧长参数，后续会作为路径拟合的自变量。
        raw_distances = self._calculate_cumulative_distance(self.raw_trajectory_points_)

        # Step 2. 拟合 s(t)，并通过求导得到 velocity / acc。
        # 这里的核心思想是：
        #   先把“走了多远”看成时间 t 的函数，再由 ds/dt、d²s/dt² 得到速度和加速度。
        timestamps = self._build_timestamps(len(self.raw_trajectory_points_))
        distances, velocity, acc = self._fit_arc_length_profile(
            raw_distances,
            timestamps,
        )

        # Step 3. 基于累计里程 s 分别拟合 x(s)、y(s)。
        # 由于这里采用 x=纵向、y=横向，因此拟合后的路径切线方向也会基于这个坐标定义计算。
        fit_x_s, fit_y_s = self._fit_path_polynomials(raw_distances, self.raw_trajectory_points_)

        x_values = self._evaluate_poly(fit_x_s, distances)
        y_values = self._evaluate_poly(fit_y_s, distances)

        # Step 4. 由 x(s)、y(s) 的一阶/二阶导数计算曲率 curvature。
        curvature = self._calculate_curvature(fit_x_s, fit_y_s, distances)

        # Step 5. 根据平滑后路径切线方向计算 yaw。
        # 这里不再直接插值原始 yaw，而是让 yaw 反映后处理后轨迹本身的朝向。
        yaw = self._calculate_yaw_from_path(fit_x_s, fit_y_s, distances)

        # Step 6. 组装输出 TrajectoryPoint 列表。
        # 输出字段固定为：x, y, velocity, acc, cur, yaw, timestamp。
        return [
            TrajectoryPoint(
                x=float(x_values[i]),
                y=float(y_values[i]),
                velocity=float(velocity[i]),
                acc=float(acc[i]),
                cur=float(curvature[i]),
                yaw=float(yaw[i]),
                timestamp=float(timestamps[i]),
            )
            for i in range(len(self.raw_trajectory_points_))
        ]

    def ComputePathProfile(self) -> list[TrajectoryPoint]:
        return self.compute_path_profile()

    @staticmethod
    def _coerce_point(point: Sequence[float]) -> TrajectoryPoint:
        values = list(point)
        if len(values) == 2:
            # 输入只接受已经变换好的 (x, y)=(纵向, 横向)。
            # yaw 和动力学字段在后处理阶段重新计算。
            return TrajectoryPoint(x=float(values[0]), y=float(values[1]))
        raise ValueError("Trajectory point sequence must contain exactly 2 elements: (x, y)")

    def _prepare_input_points(self, points: Sequence[Sequence[float]]) -> list[TrajectoryPoint]:
        # 统一把输入点整理成 TrajectoryPoint，便于后续各步骤按统一字段访问。
        prepared = [self._coerce_point(point) for point in points]
        if not prepared:
            return []

        if len(prepared) < 2:
            raise ValueError("PAD trajectory must contain at least 2 points")

        timestamps = self._build_timestamps(len(prepared))
        for index, point in enumerate(prepared):
            point.timestamp = float(timestamps[index])

        return prepared

    @staticmethod
    def _build_timestamps(point_count: int) -> np.ndarray:
        # PAD 轨迹按固定 0.5 秒间隔组织。
        # 本文件不重采样，因此输入多少个点，就生成多少个时间戳。
        return np.asarray(
            [(index + 1) * DEFAULT_TIME_INTERVAL for index in range(point_count)],
            dtype=float,
        )

    @staticmethod
    def _calculate_cumulative_distance(points: Sequence[TrajectoryPoint]) -> np.ndarray:
        # 逐点累计相邻点之间的欧氏距离，构成弧长参数 s。
        # 注意这里仍然使用平面距离 hypot(dx, dy)，
        # 只不过 dx 对应纵向变化、dy 对应横向变化。
        if not points:
            return np.zeros(0, dtype=float)

        distances = np.zeros(len(points), dtype=float)
        for index in range(1, len(points)):
            dx = float(points[index].x - points[index - 1].x)
            dy = float(points[index].y - points[index - 1].y)
            distances[index] = distances[index - 1] + math.hypot(dx, dy)
        return distances

    def _fit_arc_length_profile(
        self,
        raw_distances: np.ndarray,
        timestamps: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # 这一步实现“拟合 s(t)，并求 velocity / acc”。
        # 输入是固定 P0-P7 对应的 (timestamp, s) 对；输出仍然是 P0-P7：
        #   distances = s(t)
        #   velocity = ds/dt
        #   acc = d²s/dt²
        relative_times = timestamps - timestamps[0]
        degree = min(3, len(raw_distances) - 1)
        coeffs = np.polyfit(relative_times, raw_distances, deg=degree)

        distances = np.polyval(coeffs, relative_times)
        distances[0] = max(0.0, distances[0])
        distances = np.maximum.accumulate(distances)
        for index in range(1, len(distances)):
            distances[index] = max(distances[index], distances[index - 1] + 1e-4)

        velocity = np.polyval(np.polyder(coeffs, 1), relative_times)
        velocity = np.maximum(velocity, 0.0)
        acc = np.polyval(np.polyder(coeffs, 2), relative_times)
        return distances, velocity, acc

    @staticmethod
    def _fit_path_polynomials(
        distances: np.ndarray,
        points: Sequence[TrajectoryPoint],
    ) -> tuple[np.ndarray, np.ndarray]:
        # 用累计里程 s 作为自变量，分别拟合：
        #   x = x(s)
        #   y = y(s)
        # 这是后续计算平滑路径、曲率和 yaw 的基础。
        if len(points) == 0:
            return np.asarray([0.0], dtype=float), np.asarray([0.0], dtype=float)

        safe_distances = np.asarray(distances, dtype=float).copy()
        for index in range(1, len(safe_distances)):
            safe_distances[index] = max(safe_distances[index], safe_distances[index - 1] + 1e-4)

        degree = min(3, len(points) - 1)
        x_values = np.asarray([point.x for point in points], dtype=float)
        y_values = np.asarray([point.y for point in points], dtype=float)

        fit_x_s = np.polyfit(safe_distances, x_values, deg=degree)
        fit_y_s = np.polyfit(safe_distances, y_values, deg=degree)
        return fit_x_s, fit_y_s

    @staticmethod
    def _evaluate_poly(coeffs: np.ndarray, values: np.ndarray) -> np.ndarray:
        # 在给定 s 位置上评估多项式，得到对应的 x 或 y。
        return np.polyval(coeffs, np.asarray(values, dtype=float))

    @staticmethod
    def _calculate_curvature(
        fit_x_s: np.ndarray,
        fit_y_s: np.ndarray,
        distances: np.ndarray,
    ) -> np.ndarray:
        # 根据 x(s)、y(s) 的一阶/二阶导数计算曲率：
        #   kappa = (x' * y'' - y' * x'') / (x'^2 + y'^2)^(3/2)
        # 这里的曲率描述的是平滑后轨迹在平面内的弯曲程度。
        x_prime = np.polyval(np.polyder(fit_x_s, 1), distances)
        y_prime = np.polyval(np.polyder(fit_y_s, 1), distances)
        x_double_prime = np.polyval(np.polyder(fit_x_s, 2), distances)
        y_double_prime = np.polyval(np.polyder(fit_y_s, 2), distances)

        denominator = np.power(x_prime * x_prime + y_prime * y_prime, 1.5)
        denominator = np.maximum(denominator, 1e-6)
        return (x_prime * y_double_prime - y_prime * x_double_prime) / denominator

    @staticmethod
    def _calculate_yaw_from_path(
        fit_x_s: np.ndarray,
        fit_y_s: np.ndarray,
        distances: np.ndarray,
    ) -> np.ndarray:
        # 由平滑路径切线方向计算 yaw。
        # 因为本文件采用 x=纵向、y=横向，
        # 所以这里使用 arctan2(dy/ds, dx/ds) 得到轨迹朝向角。
        x_prime = np.polyval(np.polyder(fit_x_s, 1), distances)
        y_prime = np.polyval(np.polyder(fit_y_s, 1), distances)
        return np.arctan2(y_prime, x_prime)


def compute_path_profile(raw_trajectory_points: Sequence[Sequence[float]],) -> list[TrajectoryPoint]:
    return TrajectoryPostProcessor(raw_trajectory_points=raw_trajectory_points).compute_path_profile()


ReferenceLine = TrajectoryPostProcessor


__all__ = ["TrajectoryPoint", "TrajectoryPostProcessor", "ReferenceLine", "compute_path_profile"]
