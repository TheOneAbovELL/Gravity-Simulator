import math

import numpy as np
import pytest

from gravity_sim.camera import MAX_ELEVATION, MIN_DISTANCE, Camera
from gravity_sim.colors import blend, colormap, hex_to_rgb, rgb_to_hex, star_color
from gravity_sim.constants import AU, DAY, EARTH_MASS, SOLAR_MASS, YEAR
from gravity_sim.units import format_distance, format_duration, format_mass, format_sim_time, format_speed

W, H = 800, 600


def test_target_projects_to_screen_center() -> None:
    cam = Camera(distance=5 * AU)
    p = cam.project(np.zeros((1, 3)), W, H)
    assert (p.x[0], p.y[0]) == pytest.approx((W / 2, H / 2))
    assert p.scale[0] == pytest.approx(cam.pixels_per_meter(W, H))


def test_top_down_view_orientation() -> None:
    cam = Camera(azimuth=-90, elevation=MAX_ELEVATION, distance=10 * AU)
    p = cam.project(np.array([[AU, 0, 0], [0, AU, 0]]), W, H)
    assert p.x[0] > W / 2 and p.y[0] == pytest.approx(H / 2, abs=1.0)  # +x to the right
    assert p.y[1] < H / 2  # +y up


def test_points_behind_camera_are_invisible() -> None:
    cam = Camera(azimuth=-90, elevation=0, distance=AU)
    behind = -cam.basis()[2] * 2 * AU  # opposite the view direction, beyond the camera
    p = cam.project(np.array([behind]), W, H)
    assert not p.visible[0]


def test_nearer_points_have_larger_scale() -> None:
    cam = Camera(distance=AU)
    forward = cam.basis()[2]
    p = cam.project(np.array([-0.5 * AU * forward, 0.5 * AU * forward]), W, H)
    assert p.scale[0] > p.scale[1]


def test_fit_frames_requested_radius() -> None:
    cam = Camera(fov=60)
    cam.fit(3 * AU)
    edge = cam.basis()[0] * 3 * AU
    p = cam.project(np.array([edge]), W, H)
    assert 0 < p.x[0] - W / 2 < min(W, H) / 2


def test_pan_moves_scene_with_the_drag() -> None:
    cam = Camera(distance=AU)
    cam.pan_pixels(50, 20, W, H)
    p = cam.project(np.zeros((1, 3)), W, H)
    assert (p.x[0], p.y[0]) == pytest.approx((W / 2 + 50, H / 2 + 20))


def test_orbit_zoom_and_fov_clamp() -> None:
    cam = Camera()
    cam.orbit(400, 500)
    assert -180 <= cam.azimuth < 180 and cam.elevation == MAX_ELEVATION
    cam.zoom(1e-30)
    assert cam.distance == MIN_DISTANCE
    with pytest.raises(ValueError):
        cam.zoom(0)
    cam.set_fov(500)
    assert cam.fov == 120


def test_axis_directions_are_unit_length() -> None:
    for dx, dy, depth in Camera(azimuth=30, elevation=20).axis_screen_directions():
        assert math.isclose(dx * dx + dy * dy + depth * depth, 1.0, rel_tol=1e-12)


def test_formatting() -> None:
    assert format_mass(SOLAR_MASS) == "1.00 M☉"
    assert format_mass(EARTH_MASS, ascii_only=True) == "1.00 Mearth"
    assert format_distance(AU) == "1.00 AU"
    assert format_distance(6.371e6) == "6,371 km"
    assert format_speed(29780) == "29.8 km/s"
    assert format_duration(DAY / 4) == "6.00 h"
    assert format_duration(2 * YEAR) == "2.00 yr"
    assert format_sim_time(YEAR + 10 * DAY) == "1 yr  10.0 d"
    assert format_sim_time(DAY / 2) == "0.50 d"


def test_colors() -> None:
    assert rgb_to_hex(hex_to_rgb("#12abef")) == "#12abef"
    assert blend("#000000", "#ffffff", 0.5) == "#808080"
    assert colormap(0.0) == "#0d0887" and colormap(1.0) == "#f0f921"
    assert colormap(float("nan")) == colormap(0.0)
    assert star_color(0.05) == "#ff7a4d" and star_color(50) == "#9fbcff"
    with pytest.raises(ValueError):
        hex_to_rgb("#fff")
