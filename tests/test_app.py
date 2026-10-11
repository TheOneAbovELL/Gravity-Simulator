"""GUI smoke tests: drive the real Tk window without a main loop."""

from __future__ import annotations

import contextlib
import tkinter as tk
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from gravity_sim import app as app_module
from gravity_sim.app import COLOR_BY_LABELS, OCTREE_LABELS, Focus, GravityApp
from gravity_sim.presets import PRESETS, build_scenario

pytestmark = pytest.mark.gui


@pytest.fixture(scope="module")
def tk_root() -> Iterator[tk.Tk]:
    try:
        r = tk.Tk()
    except tk.TclError as exc:  # no display available (e.g. headless CI without xvfb)
        pytest.skip(f"no display: {exc}")
    r.withdraw()
    yield r
    r.destroy()


@pytest.fixture
def root(tk_root: tk.Tk) -> Iterator[tk.Toplevel]:
    top = tk.Toplevel(tk_root)
    top.geometry("1200x800+0+0")
    yield top
    with contextlib.suppress(tk.TclError):  # the test may already have closed it
        top.destroy()


@pytest.fixture
def app(root: tk.Toplevel) -> Iterator[GravityApp]:
    a = GravityApp(root, build_scenario("inner"), "inner", autostart=False)
    root.update()
    yield a


def pump(app: GravityApp, frames: int = 3) -> None:
    for _ in range(frames):
        app._advance(app.steps_per_frame)
        app.render()
        app._update_inspector()
        app.root.update()


def test_runs_and_draws_bodies(app: GravityApp) -> None:
    steps0 = app.sim.steps
    pump(app)
    assert app.sim.steps > steps0
    assert len(app.canvas.find_all()) > app.sim.n  # bodies + trails + overlay
    assert app.body_title.cget("text") == "Sun"  # preset follows the Sun


def test_pause_and_single_step(app: GravityApp) -> None:
    app.toggle_running()
    assert not app.running and app.play_button.cget("text") == "Play"
    steps = app.sim.steps
    app.single_step()
    assert app.sim.steps == steps + 1


def test_every_preset_loads_and_renders(app: GravityApp) -> None:
    for key, preset in PRESETS.items():
        app.v_preset.set(preset.title)
        app._on_preset_selected()
        if preset.default_bodies:
            app.v_bodies.set("40")
        app.load_selected_preset()
        assert app.preset_key == key
        pump(app, 1)
        assert app.status.cget("text").startswith("Loaded")


def test_invalid_scenario_input_shows_error(app: GravityApp) -> None:
    app.v_preset.set(PRESETS["cluster"].title)
    app._on_preset_selected()
    app.v_bodies.set("abc")
    app.load_selected_preset()
    assert "Cannot load" in app.status.cget("text")
    assert app.preset_key == "inner"  # unchanged


def test_view_options_render(app: GravityApp) -> None:
    for var in (app.v_octree, app.v_com, app.v_true_scale, app.v_relative, app.v_help):
        var.set(True)
    for kind in OCTREE_LABELS.values():
        app.v_octree_kind.set(kind)
        for label in COLOR_BY_LABELS.values():
            app.v_color_by.set(label)
            pump(app, 1)
    app.v_trails.set(False)
    app.v_hud.set(False)
    pump(app, 1)


def test_physics_controls_update_simulation(app: GravityApp) -> None:
    app.v_dt.set("0.5")
    app._apply_dt()
    assert app.sim.config.dt == pytest.approx(0.5 * 86400)
    app.v_dt.set("fast")
    app._apply_dt()
    assert "number" in app.status.cget("text")
    assert app.v_dt.get() == "0.5"  # widget reverted to the real value

    app.v_solver.set("barnes-hut")
    app._apply_physics(solver="barnes-hut")
    assert app.sim.active_solver == "barnes-hut"
    app.v_theta.set(0.8)
    app._on_theta_drag()
    assert app.sim.config.theta == pytest.approx(0.8)

    app.v_softening.set("0.01")
    app._apply_softening()
    assert app.sim.config.softening > 0 and "baseline" in app.status.cget("text")

    app.v_spf.set("0")
    app._apply_spf()
    assert app.steps_per_frame == 4
    app.v_trail_len.set("50")
    app._apply_trail_length()
    assert app.sim.config.trail_length == 50
    pump(app, 2)


def test_focus_cycling_and_picking(app: GravityApp) -> None:
    app.cycle_focus(1)
    assert app.focus.kind == "body"
    app.set_focus(Focus("com"))
    app.v_focus.set("Earth")
    app._on_focus_selected()
    earth = app.sim.names.index("Earth")
    assert app.focus.body_id == app.sim.ids[earth]
    # Clicking the centre of the view picks the followed body.
    app.root.update()
    w, h = app._canvas_size()
    app.set_focus(Focus("com"))
    app.render()
    proj = app.camera.project(app.sim.pos - app._focus_position(), w, h)
    app._pick(int(proj.x[earth]), int(proj.y[earth]))
    assert app.focus.body_id == app.sim.ids[earth]
    app._pick(3, 3)  # empty space clears the selection, keeps following
    assert app.selected is None and app.focus.body_id == app.sim.ids[earth]


def test_camera_controls(app: GravityApp) -> None:
    d0 = app.camera.distance
    app._zoom_by(0.5)
    assert app.camera.distance == pytest.approx(d0 / 2)
    az = app.camera.azimuth
    app._drag = ("orbit", 100, 100, False)
    app._motion(type("E", (), {"x": 160, "y": 100})())
    assert app.camera.azimuth != az
    app.reset_camera()
    assert np.allclose(app.camera.offset, 0)


def test_merges_keep_focus_on_survivor(root: tk.Toplevel) -> None:
    app = GravityApp(root, build_scenario("cluster", 60), "cluster", 60, autostart=False)
    sim = app.sim
    # Force a merge: put star 1 on top of star 0 and follow star 1.
    sim.pos[1] = sim.pos[0]
    sim.vel[1] = sim.vel[0]
    absorbed = sim.ids[1] if sim.mass[1] < sim.mass[0] else sim.ids[0]
    app.set_focus(Focus("body", absorbed))
    app._advance(1)
    assert sim.n < 60
    assert app.focus.kind == "body" and app.focus.body_id == sim.resolve_id(absorbed)
    assert len(app.focus_box.cget("values")) == sim.n + 2


def test_restart_keeps_settings(app: GravityApp) -> None:
    app._apply_physics(dt=3600.0)
    pump(app, 2)
    app.restart()
    assert app.sim.steps == 0 and app.sim.config.dt == 3600.0


def test_main_loop_ticks_and_close_stops_it(tk_root: tk.Tk, root: tk.Toplevel) -> None:
    app = GravityApp(root, build_scenario("binary"), "binary")  # real after()-driven loop

    def finish() -> None:
        app.close()
        tk_root.quit()

    tk_root.after(400, finish)
    tk_root.mainloop()
    assert app.closed and app.sim.steps > 0
    assert app._fps > 0


def test_save_image_eps_and_png(app: GravityApp, tmp_path: Path) -> None:
    pump(app, 1)
    assert app.save_image_to(tmp_path / "view.eps")
    assert (tmp_path / "view.eps").stat().st_size > 1000
    pytest.importorskip("PIL")
    app.root.lift()
    assert app.save_image_to(tmp_path / "view.png")
    from PIL import Image

    with Image.open(tmp_path / "view.png") as img:
        assert img.size[0] > 100 and img.size[1] > 100


def test_recording_writes_the_requested_number_of_frames(root: tk.Toplevel, tmp_path: Path) -> None:
    pytest.importorskip("PIL")
    frames = tmp_path / "frames"
    app = GravityApp(root, build_scenario("binary"), "binary", record_dir=frames, record_frames=3, autostart=False)
    root.update()
    for _ in range(5):
        app._tick()  # one real frame: physics, render, capture
        if app._after_id is not None:
            root.after_cancel(app._after_id)
    assert sorted(p.name for p in frames.iterdir()) == ["frame_00000.png", "frame_00001.png", "frame_00002.png"]
    assert "Recording finished" in app.status.cget("text")


def test_physics_budget_keeps_frames_responsive(app: GravityApp, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app_module, "PHYSICS_BUDGET_S", 0.0)
    steps = app.sim.steps
    app._advance(50)
    assert app.sim.steps == steps + 1  # stopped after the budget ran out
    app.render()
    items = [i for i in app.canvas.find_all() if app.canvas.type(i) == "text"]  # type: ignore[comparison-overlap]  # stub says int
    texts = [str(app.canvas.itemcget(i, "text")) for i in items]  # type: ignore[no-untyped-call]
    assert any("physics-limited: 1/50" in t for t in texts)
