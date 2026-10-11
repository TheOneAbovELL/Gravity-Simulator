"""Orbit camera with perspective projection (pure math, no GUI dependency).

The camera orbits a *target* (the focus point plus a pan offset). World
coordinates are the simulation's x-y-z frame with z "up" (for the solar
system presets, z is the ecliptic north pole).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

MIN_ELEVATION = -89.5
MAX_ELEVATION = 89.5
MIN_DISTANCE = 1.0e3  # 1 km
MAX_DISTANCE = 1.0e18  # ~100 light-years
MIN_FOV, MAX_FOV = 10.0, 120.0


@dataclass(frozen=True)
class Projection:
    """Screen-space result for M points (all arrays have length M)."""

    x: np.ndarray
    y: np.ndarray
    depth: np.ndarray
    scale: np.ndarray
    """Pixels per metre at each point's depth."""
    visible: np.ndarray
    """True where the point is in front of the camera."""


@dataclass
class Camera:
    azimuth: float = -90.0
    """Degrees; -90 looks along +y so +x points right on screen."""
    elevation: float = 55.0
    """Degrees above the x-y plane (90 = looking straight down)."""
    distance: float = 1.0e12
    """Metres from the camera to its target."""
    fov: float = 45.0
    """Field of view across the shorter screen dimension, degrees."""
    offset: np.ndarray = field(default_factory=lambda: np.zeros(3))
    """Pan offset of the target from the focus point, metres."""

    def __post_init__(self) -> None:
        self.elevation = min(MAX_ELEVATION, max(MIN_ELEVATION, self.elevation))
        self.distance = min(MAX_DISTANCE, max(MIN_DISTANCE, self.distance))

    # ----------------------------------------------------------------- basis
    def basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Unit vectors ``(right, up, forward)`` of the view in world space."""
        az, el = math.radians(self.azimuth), math.radians(self.elevation)
        forward = np.array([-math.cos(el) * math.cos(az), -math.cos(el) * math.sin(az), -math.sin(el)])
        right = np.array([-math.sin(az), math.cos(az), 0.0])
        up = np.cross(right, forward)
        return right, up, forward

    def focal_px(self, width: int, height: int) -> float:
        return 0.5 * min(width, height) / math.tan(math.radians(self.fov) / 2)

    def pixels_per_meter(self, width: int, height: int) -> float:
        """Scale at the target's depth."""
        return self.focal_px(width, height) / self.distance

    # ------------------------------------------------------------ projection
    def project(self, points: np.ndarray, width: int, height: int) -> Projection:
        """Project points given relative to the focus point (shape (M, 3))."""
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3) - self.offset
        right, up, forward = self.basis()
        focal = self.focal_px(width, height)
        xc = pts @ right
        yc = pts @ up
        depth = self.distance + pts @ forward
        visible = depth > self.distance * 1e-3
        safe = np.where(visible, depth, np.inf)
        scale = focal / safe
        return Projection(
            x=width / 2 + xc * scale,
            y=height / 2 - yc * scale,
            depth=depth,
            scale=scale,
            visible=visible,
        )

    def axis_screen_directions(self) -> list[tuple[float, float, float]]:
        """For the world x, y, z axes: screen ``(dx, dy)`` unit-ish direction and depth sign."""
        right, up, forward = self.basis()
        out = []
        for axis in np.eye(3):
            out.append((float(axis @ right), float(-(axis @ up)), float(axis @ forward)))
        return out

    # ----------------------------------------------------------- interaction
    def orbit(self, d_azimuth: float, d_elevation: float) -> None:
        self.azimuth = (self.azimuth + d_azimuth + 180.0) % 360.0 - 180.0
        self.elevation = min(MAX_ELEVATION, max(MIN_ELEVATION, self.elevation + d_elevation))

    def zoom(self, factor: float) -> None:
        if factor <= 0 or not math.isfinite(factor):
            raise ValueError("zoom factor must be positive")
        self.distance = min(MAX_DISTANCE, max(MIN_DISTANCE, self.distance * factor))

    def pan_pixels(self, dx: float, dy: float, width: int, height: int) -> None:
        """Move the scene by ``(dx, dy)`` screen pixels (drag direction)."""
        right, up, _ = self.basis()
        meters_per_px = self.distance / self.focal_px(width, height)
        self.offset = self.offset - right * dx * meters_per_px + up * dy * meters_per_px

    def set_fov(self, fov: float) -> None:
        self.fov = min(MAX_FOV, max(MIN_FOV, fov))

    def fit(self, radius: float) -> None:
        """Place the camera so a sphere of ``radius`` around the target fills the view."""
        radius = max(radius, MIN_DISTANCE)
        self.distance = min(MAX_DISTANCE, max(MIN_DISTANCE, 1.1 * radius / math.tan(math.radians(self.fov) / 2)))
        self.offset = np.zeros(3)
