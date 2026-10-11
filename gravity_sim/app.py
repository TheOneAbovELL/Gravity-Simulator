"""Tkinter user interface.

The UI owns a :class:`~gravity_sim.simulation.Simulation`, a
:class:`~gravity_sim.camera.Camera` and the view options. A single
``after()``-driven loop advances the physics and redraws the canvas; while
paused the canvas is only redrawn when something changes.
"""

from __future__ import annotations

import math
import sys
import time
import tkinter as tk
import tkinter.font as tkfont
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Literal

import numpy as np

from .bodies import BodyKind
from .camera import Camera
from .collisions import COLLISION_MODES
from .colors import blend, colormap
from .constants import AU, DAY, EARTH_RADIUS
from .forces import SOLVERS
from .presets import PRESETS, Scenario, build_scenario
from .simulation import NumericalInstabilityError, Simulation
from .units import (
    format_acceleration,
    format_density,
    format_distance,
    format_duration,
    format_energy,
    format_mass,
    format_sim_time,
    format_speed,
)

# ----------------------------------------------------------------- palette
BG = "#07090f"
PANEL = "#11151e"
FIELD = "#1a202c"
BORDER = "#262e3d"
FG = "#d9dee8"
MUTED = "#8891a5"
ACCENT = "#f0b44c"
GOOD = "#6fcf97"
WARN = "#f2c94c"
BAD = "#eb5757"
GRID = "#2f7a5b"
COM_COLOR = "#e6e6e6"

FRAME_MS = 16
PHYSICS_BUDGET_S = 0.1  # keep the window responsive: at most this much physics per frame
INSPECT_INTERVAL = 0.25  # seconds between inspector refreshes
MAX_OCTREE_BOXES = 1500
MAX_COORD = 1.0e6  # clamp screen coordinates handed to Tk
CROWDED = 30  # above this many bodies: no halos, labels or trail fading (keeps drawing fast)

FocusKind = Literal["com", "origin", "body"]
ColorBy = Literal["body", "speed", "acceleration", "mass"]
COLOR_BY_LABELS: dict[ColorBy, str] = {
    "body": "Body colour",
    "speed": "Speed",
    "acceleration": "Acceleration",
    "mass": "Mass",
}
OCTREE_LABELS = {"internal": "Regional nodes", "leaf": "Body nodes", "all": "All nodes"}

HELP_TEXT = """\
Mouse
  Left-drag          rotate the view
  Right/middle-drag  pan
  Wheel              zoom
  Click a body       select and follow it

Keyboard
  Space   play / pause        S   single step
  Left/Right  previous / next body to follow
  + / -   zoom                Home  reset camera
  T trails   O octree   C centre of mass
  L labels   H overlay    F11 fullscreen
  F1 or ?  toggle this help"""


@dataclass
class Focus:
    kind: FocusKind = "com"
    body_id: int | None = None


class GravityApp:
    """Main window."""

    def __init__(
        self,
        root: tk.Tk | tk.Toplevel,
        scenario: Scenario,
        preset_key: str,
        bodies: int | None = None,
        seed: int = 1,
        paused: bool = False,
        record_dir: Path | None = None,
        record_frames: int = 3600,
        autostart: bool = True,
    ) -> None:
        self.root = root
        self.preset_key = preset_key
        self.scenario = scenario
        self.running = not paused
        self.steps_per_frame = scenario.steps_per_frame
        self.camera = Camera()
        self.focus = Focus()
        self.selected: int | None = None
        self.dirty = True
        self.closed = False
        self.fullscreen = False
        self.record_dir = record_dir
        self.record_frames_left = record_frames if record_dir else 0
        self.frame_count = 0
        self._after_id: str | None = None
        self._last_inspect = 0.0
        self._last_tick = time.perf_counter()
        self._fps = 0.0
        self._phys_ms = 0.0
        self._steps_done = 0
        self._steps_wanted = 0
        self._draw_ms = 0.0
        self._known_n = -1
        self._drag: tuple[str, int, int, bool] | None = None
        self._color_cache: dict[tuple[str, float], str] = {}

        self._init_style()
        self._init_vars(bodies, seed)
        self._build_layout()
        self._bind_events()
        self._load(scenario)
        self._set_status("Drag to rotate, right-drag to pan, wheel to zoom, click a body to follow it.  F1: help")
        if autostart:
            self._tick()

    # ================================================================ setup
    def _init_style(self) -> None:
        root = self.root
        root.title("Gravity Simulation")
        root.configure(background=PANEL)
        root.minsize(900, 600)
        families = set(tkfont.families(root))
        mono = next((f for f in ("Cascadia Mono", "Consolas", "Menlo", "DejaVu Sans Mono") if f in families), "Courier")
        self.mono = tkfont.Font(root=root, family=mono, size=10)
        self.mono_small = tkfont.Font(root=root, family=mono, size=9)
        self.ui_font = tkfont.nametofont("TkDefaultFont")
        self.title_font = tkfont.Font(root=root, family=self.ui_font.actual("family"), size=13, weight="bold")

        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure(
            ".",
            background=PANEL,
            foreground=FG,
            fieldbackground=FIELD,
            bordercolor=BORDER,
            lightcolor=PANEL,
            darkcolor=PANEL,
            troughcolor=FIELD,
            focuscolor=ACCENT,
            selectbackground="#2b3447",
            selectforeground=FG,
            insertcolor=FG,
            arrowcolor=FG,
        )
        style.configure("TButton", padding=(10, 5), background=FIELD)
        style.map("TButton", background=[("pressed", "#2b3447"), ("active", "#232a39")])
        style.configure("Accent.TButton", background=ACCENT, foreground="#1d1405")
        style.map("Accent.TButton", background=[("pressed", "#c99335"), ("active", "#f5c46a")])
        style.configure("TNotebook", background=PANEL, borderwidth=0, tabmargins=(0, 4, 0, 0))
        style.configure("TNotebook.Tab", padding=(10, 4), background=PANEL, foreground=MUTED, borderwidth=0)
        style.map("TNotebook.Tab", background=[("selected", FIELD)], foreground=[("selected", FG)])
        style.configure("TLabelframe", background=PANEL, bordercolor=BORDER)
        style.configure("TLabelframe.Label", background=PANEL, foreground=MUTED)
        style.configure("Muted.TLabel", foreground=MUTED)
        style.configure("Value.TLabel", font=self.mono_small)
        style.configure("Title.TLabel", font=self.title_font)
        style.configure("Status.TLabel", background="#0c1017", foreground=MUTED, padding=(8, 3))
        style.configure("StatusError.TLabel", background="#0c1017", foreground=BAD, padding=(8, 3))
        style.configure("TCheckbutton", background=PANEL, indicatorbackground=FIELD)
        style.map("TCheckbutton", background=[("active", PANEL)], indicatorbackground=[("selected", ACCENT)])
        style.configure("TRadiobutton", background=PANEL, indicatorbackground=FIELD)
        style.map("TRadiobutton", background=[("active", PANEL)], indicatorbackground=[("selected", ACCENT)])
        style.configure("Horizontal.TScale", background=PANEL)
        # clam paints read-only/disabled fields light by default; keep them on the dark palette.
        for widget in ("TCombobox", "TSpinbox", "TEntry"):
            style.map(
                widget,
                fieldbackground=[("readonly", FIELD), ("disabled", PANEL)],
                foreground=[("disabled", MUTED), ("readonly", FG)],
                selectbackground=[("readonly", FIELD)],
                selectforeground=[("readonly", FG)],
            )
        root.option_add("*TCombobox*Listbox.background", FIELD)
        root.option_add("*TCombobox*Listbox.foreground", FG)
        root.option_add("*TCombobox*Listbox.selectBackground", "#2b3447")
        root.option_add("*TCombobox*Listbox.selectForeground", FG)

    def _init_vars(self, bodies: int | None, seed: int) -> None:
        r = self.root
        self.v_preset = tk.StringVar(r, PRESETS[self.preset_key].title)
        self.v_bodies = tk.StringVar(r, str(bodies or PRESETS[self.preset_key].default_bodies or ""))
        self.v_seed = tk.StringVar(r, str(seed))
        self.v_dt = tk.StringVar(r)
        self.v_spf = tk.StringVar(r)
        self.v_solver = tk.StringVar(r)
        self.v_theta = tk.DoubleVar(r)
        self.v_softening = tk.StringVar(r)
        self.v_collisions = tk.StringVar(r)
        self.v_restitution = tk.DoubleVar(r)
        self.v_focus = tk.StringVar(r)
        self.v_relative = tk.BooleanVar(r, False)
        self.v_trails = tk.BooleanVar(r, True)
        self.v_trail_len = tk.StringVar(r)
        self.v_octree = tk.BooleanVar(r, False)
        self.v_octree_kind = tk.StringVar(r, OCTREE_LABELS["internal"])
        self.v_com = tk.BooleanVar(r, False)
        self.v_labels = tk.BooleanVar(r, True)
        self.v_hud = tk.BooleanVar(r, True)
        self.v_axes = tk.BooleanVar(r, True)
        self.v_color_by = tk.StringVar(r, COLOR_BY_LABELS["body"])
        self.v_true_scale = tk.BooleanVar(r, False)
        self.v_fov = tk.DoubleVar(r, self.camera.fov)
        self.v_help = tk.BooleanVar(r, False)

    def _build_layout(self) -> None:
        root = self.root
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(root, background=BG, highlightthickness=0, cursor="crosshair", width=960, height=720)
        self.canvas.grid(row=0, column=0, sticky="nsew")

        side = ttk.Frame(root, padding=(12, 10, 12, 8), width=320)
        side.grid(row=0, column=1, sticky="ns")
        side.grid_propagate(False)
        side.columnconfigure(0, weight=1)
        side.rowconfigure(3, weight=1)

        ttk.Label(side, text="Gravity Simulation", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        self.scenario_label = ttk.Label(side, text="", style="Muted.TLabel", wraplength=290, justify="left")
        self.scenario_label.grid(row=1, column=0, sticky="we", pady=(2, 8))

        transport = ttk.Frame(side)
        transport.grid(row=2, column=0, sticky="we", pady=(0, 8))
        for c in range(3):
            transport.columnconfigure(c, weight=1, uniform="t")
        self.play_button = ttk.Button(transport, text="Pause", style="Accent.TButton", command=self.toggle_running)
        self.play_button.grid(row=0, column=0, sticky="we", padx=(0, 4))
        ttk.Button(transport, text="Step", command=self.single_step).grid(row=0, column=1, sticky="we", padx=4)
        ttk.Button(transport, text="Restart", command=self.restart).grid(row=0, column=2, sticky="we", padx=(4, 0))

        notebook = ttk.Notebook(side)
        notebook.grid(row=3, column=0, sticky="nsew")
        self._build_scenario_tab(notebook)
        self._build_physics_tab(notebook)
        self._build_view_tab(notebook)
        self._build_inspect_tab(notebook)

        self.status = ttk.Label(root, text="", style="Status.TLabel", anchor="w")
        self.status.grid(row=1, column=0, columnspan=2, sticky="we")

    @staticmethod
    def _tab(notebook: ttk.Notebook, title: str) -> ttk.Frame:
        frame = ttk.Frame(notebook, padding=(4, 10, 4, 4))
        frame.columnconfigure(1, weight=1)
        notebook.add(frame, text=title)
        return frame

    @staticmethod
    def _row_label(parent: ttk.Frame, row: int, text: str) -> None:
        ttk.Label(parent, text=text).grid(row=row, column=0, sticky="w", pady=3, padx=(0, 8))

    def _build_scenario_tab(self, nb: ttk.Notebook) -> None:
        f = self._tab(nb, "Scenario")
        self._row_label(f, 0, "Preset")
        self.preset_box = ttk.Combobox(
            f, textvariable=self.v_preset, state="readonly", values=[p.title for p in PRESETS.values()]
        )
        self.preset_box.grid(row=0, column=1, sticky="we")
        self.preset_box.bind("<<ComboboxSelected>>", lambda _e: self._on_preset_selected())
        self.preset_desc = ttk.Label(f, text="", style="Muted.TLabel", wraplength=270, justify="left")
        self.preset_desc.grid(row=1, column=0, columnspan=2, sticky="we", pady=(4, 8))
        self._row_label(f, 2, "Bodies")
        self.bodies_spin = ttk.Spinbox(f, textvariable=self.v_bodies, from_=2, to=2000, increment=10, width=8)
        self.bodies_spin.grid(row=2, column=1, sticky="w")
        self._row_label(f, 3, "Seed")
        self.seed_spin = ttk.Spinbox(f, textvariable=self.v_seed, from_=0, to=2**31 - 1, width=8)
        self.seed_spin.grid(row=3, column=1, sticky="w")
        ttk.Button(f, text="Load scenario", command=self.load_selected_preset).grid(
            row=4, column=0, columnspan=2, sticky="we", pady=(10, 0)
        )
        ttk.Label(
            f,
            text="Loading a preset applies its recommended time step and physics settings. "
            "Restart rewinds the current scenario but keeps your settings.",
            style="Muted.TLabel",
            wraplength=270,
            justify="left",
        ).grid(row=5, column=0, columnspan=2, sticky="we", pady=(10, 0))
        self._on_preset_selected()

    def _build_physics_tab(self, nb: ttk.Notebook) -> None:
        f = self._tab(nb, "Physics")
        self._row_label(f, 0, "Time step (days)")
        dt_box = ttk.Combobox(
            f, textvariable=self.v_dt, width=10, values=["0.01", "0.05", "0.1", "0.25", "0.5", "1", "2", "5", "10"]
        )
        dt_box.grid(row=0, column=1, sticky="we")
        for seq in ("<Return>", "<FocusOut>", "<<ComboboxSelected>>"):
            dt_box.bind(seq, lambda _e: self._apply_dt())

        self._row_label(f, 1, "Steps per frame")
        spf = ttk.Spinbox(f, textvariable=self.v_spf, from_=1, to=1000, width=8, command=self._apply_spf)
        spf.grid(row=1, column=1, sticky="w")
        for seq in ("<Return>", "<FocusOut>"):
            spf.bind(seq, lambda _e: self._apply_spf())
        self.speed_label = ttk.Label(f, text="", style="Muted.TLabel")
        self.speed_label.grid(row=2, column=0, columnspan=2, sticky="w", pady=(0, 8))

        self._row_label(f, 3, "Force solver")
        solver = ttk.Combobox(f, textvariable=self.v_solver, state="readonly", values=list(SOLVERS), width=12)
        solver.grid(row=3, column=1, sticky="we")
        solver.bind("<<ComboboxSelected>>", lambda _e: self._apply_physics(solver=self.v_solver.get()))
        self.solver_label = ttk.Label(f, text="", style="Muted.TLabel")
        self.solver_label.grid(row=4, column=0, columnspan=2, sticky="w")

        self._row_label(f, 5, "Barnes-Hut θ")
        theta_row = ttk.Frame(f)
        theta_row.grid(row=5, column=1, sticky="we")
        theta_row.columnconfigure(0, weight=1)
        ttk.Scale(theta_row, from_=0.0, to=1.5, variable=self.v_theta, command=lambda _v: self._on_theta_drag()).grid(
            row=0, column=0, sticky="we"
        )
        self.theta_label = ttk.Label(theta_row, text="", style="Value.TLabel", width=5)
        self.theta_label.grid(row=0, column=1, padx=(6, 0))

        self._row_label(f, 6, "Softening (AU)")
        soft = ttk.Entry(f, textvariable=self.v_softening, width=10)
        soft.grid(row=6, column=1, sticky="we", pady=(6, 0))
        for seq in ("<Return>", "<FocusOut>"):
            soft.bind(seq, lambda _e: self._apply_softening())

        self._row_label(f, 7, "Collisions")
        colbox = ttk.Combobox(f, textvariable=self.v_collisions, state="readonly", values=list(COLLISION_MODES), width=12)
        colbox.grid(row=7, column=1, sticky="we", pady=(6, 0))
        colbox.bind("<<ComboboxSelected>>", lambda _e: self._apply_physics(collisions=self.v_collisions.get()))

        self._row_label(f, 8, "Restitution")
        rest_row = ttk.Frame(f)
        rest_row.grid(row=8, column=1, sticky="we")
        rest_row.columnconfigure(0, weight=1)
        ttk.Scale(rest_row, from_=0.0, to=1.0, variable=self.v_restitution, command=lambda _v: self._on_restitution_drag()).grid(
            row=0, column=0, sticky="we"
        )
        self.rest_label = ttk.Label(rest_row, text="", style="Value.TLabel", width=5)
        self.rest_label.grid(row=0, column=1, padx=(6, 0))

        ttk.Label(
            f,
            text="Leapfrog (velocity Verlet) integration. Softening smooths close encounters; "
            "changing it resets the energy baseline. Merging conserves mass and momentum.",
            style="Muted.TLabel",
            wraplength=270,
            justify="left",
        ).grid(row=9, column=0, columnspan=2, sticky="we", pady=(12, 0))

    def _build_view_tab(self, nb: ttk.Notebook) -> None:
        f = self._tab(nb, "View")
        self._row_label(f, 0, "Follow")
        self.focus_box = ttk.Combobox(f, textvariable=self.v_focus, state="readonly", width=18)
        self.focus_box.grid(row=0, column=1, sticky="we")
        self.focus_box.bind("<<ComboboxSelected>>", lambda _e: self._on_focus_selected())

        ttk.Checkbutton(f, text="Trails relative to followed object", variable=self.v_relative, command=self._mark_dirty).grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(6, 0)
        )
        checks = [
            ("Orbit trails", self.v_trails),
            ("Labels", self.v_labels),
            ("Centre-of-mass marker", self.v_com),
            ("Axes indicator", self.v_axes),
            ("Overlay (time, energy)", self.v_hud),
            ("True body scale", self.v_true_scale),
        ]
        for i, (text, var) in enumerate(checks):
            ttk.Checkbutton(f, text=text, variable=var, command=self._mark_dirty).grid(
                row=2 + i, column=0, columnspan=2, sticky="w"
            )

        row = 2 + len(checks)
        self._row_label(f, row, "Trail samples")
        tl = ttk.Spinbox(
            f, textvariable=self.v_trail_len, from_=10, to=20000, increment=100, width=8, command=self._apply_trail_length
        )
        tl.grid(row=row, column=1, sticky="w")
        for seq in ("<Return>", "<FocusOut>"):
            tl.bind(seq, lambda _e: self._apply_trail_length())

        ttk.Checkbutton(f, text="Barnes-Hut octree", variable=self.v_octree, command=self._mark_dirty).grid(
            row=row + 1, column=0, sticky="w", pady=(6, 0)
        )
        ok = ttk.Combobox(f, textvariable=self.v_octree_kind, state="readonly", values=list(OCTREE_LABELS.values()), width=14)
        ok.grid(row=row + 1, column=1, sticky="we", pady=(6, 0))
        ok.bind("<<ComboboxSelected>>", lambda _e: self._mark_dirty())

        self._row_label(f, row + 2, "Colour by")
        cb = ttk.Combobox(f, textvariable=self.v_color_by, state="readonly", values=list(COLOR_BY_LABELS.values()), width=14)
        cb.grid(row=row + 2, column=1, sticky="we", pady=(6, 0))
        cb.bind("<<ComboboxSelected>>", lambda _e: self._mark_dirty())

        self._row_label(f, row + 3, "Field of view")
        ttk.Scale(f, from_=15, to=100, variable=self.v_fov, command=lambda _v: self._on_fov()).grid(
            row=row + 3, column=1, sticky="we", pady=(6, 0)
        )

        buttons = ttk.Frame(f)
        buttons.grid(row=row + 4, column=0, columnspan=2, sticky="we", pady=(12, 0))
        for c in range(3):
            buttons.columnconfigure(c, weight=1, uniform="b")
        ttk.Button(buttons, text="Reset view", command=self.reset_camera).grid(row=0, column=0, sticky="we", padx=(0, 3))
        ttk.Button(buttons, text="Clear trails", command=self.clear_trails).grid(row=0, column=1, sticky="we", padx=3)
        ttk.Button(buttons, text="Save image", command=self.save_image).grid(row=0, column=2, sticky="we", padx=(3, 0))

    def _build_inspect_tab(self, nb: ttk.Notebook) -> None:
        f = self._tab(nb, "Inspect")
        self.body_title = ttk.Label(f, text="", style="Title.TLabel")
        self.body_title.grid(row=0, column=0, columnspan=2, sticky="w")
        self.body_rows: dict[str, ttk.Label] = {}
        body_keys = ["Type", "Mass", "Radius", "Density", "Distance", "Speed", "Acceleration"]
        for i, key in enumerate(body_keys, start=1):
            ttk.Label(f, text=key, style="Muted.TLabel").grid(row=i, column=0, sticky="w", padx=(0, 8))
            lbl = ttk.Label(f, text="", style="Value.TLabel")
            lbl.grid(row=i, column=1, sticky="w")
            self.body_rows[key] = lbl
        start = len(body_keys) + 1
        ttk.Separator(f).grid(row=start, column=0, columnspan=2, sticky="we", pady=8)
        ttk.Label(f, text="System", style="Title.TLabel").grid(row=start + 1, column=0, columnspan=2, sticky="w")
        self.sys_rows: dict[str, ttk.Label] = {}
        sys_keys = [
            "Time",
            "Steps",
            "Bodies",
            "Total mass",
            "Kinetic E",
            "Potential E",
            "Total E",
            "Energy drift",
            "Collision ΔE",
            "|Momentum|",
            "Collisions",
            "Octree",
        ]
        for i, key in enumerate(sys_keys, start=start + 2):
            ttk.Label(f, text=key, style="Muted.TLabel").grid(row=i, column=0, sticky="w", padx=(0, 8))
            lbl = ttk.Label(f, text="", style="Value.TLabel")
            lbl.grid(row=i, column=1, sticky="w")
            self.sys_rows[key] = lbl
        ttk.Label(
            f,
            text="Energy drift = (E + collision ΔE - E₀)/|E₀|: the numerical error, excluding energy changed by collisions.",
            style="Muted.TLabel",
            wraplength=270,
            justify="left",
        ).grid(row=start + 2 + len(sys_keys), column=0, columnspan=2, sticky="we", pady=(8, 0))

    def _bind_events(self) -> None:
        c = self.canvas
        c.bind("<Configure>", lambda _e: self._mark_dirty())
        c.bind("<ButtonPress-1>", lambda e: self._press(e, "orbit"))
        c.bind("<B1-Motion>", self._motion)
        c.bind("<ButtonRelease-1>", self._release)
        for button in (2, 3):
            c.bind(f"<ButtonPress-{button}>", lambda e: self._press(e, "pan"))
            c.bind(f"<B{button}-Motion>", self._motion)
            c.bind(f"<ButtonRelease-{button}>", self._release)
        c.bind("<MouseWheel>", self._wheel)
        c.bind("<Button-4>", lambda e: self._zoom_by(0.85))  # X11 wheel
        c.bind("<Button-5>", lambda e: self._zoom_by(1 / 0.85))

        r = self.root
        keys = {
            "<space>": self.toggle_running,
            "<s>": self.single_step,
            "<Left>": lambda: self.cycle_focus(-1),
            "<Right>": lambda: self.cycle_focus(1),
            "<plus>": lambda: self._zoom_by(0.8),
            "<equal>": lambda: self._zoom_by(0.8),
            "<KP_Add>": lambda: self._zoom_by(0.8),
            "<minus>": lambda: self._zoom_by(1.25),
            "<KP_Subtract>": lambda: self._zoom_by(1.25),
            "<Home>": self.reset_camera,
            "<t>": lambda: self._toggle(self.v_trails),
            "<o>": lambda: self._toggle(self.v_octree),
            "<c>": lambda: self._toggle(self.v_com),
            "<l>": lambda: self._toggle(self.v_labels),
            "<h>": lambda: self._toggle(self.v_hud),
            "<F1>": lambda: self._toggle(self.v_help),
            "<question>": lambda: self._toggle(self.v_help),
            "<F11>": self.toggle_fullscreen,
            "<Escape>": self._escape,
        }
        for seq, fn in keys.items():
            r.bind(seq, partial(self._key, fn=fn))
        r.protocol("WM_DELETE_WINDOW", self.close)

    # ============================================================ scenario
    def _load(self, scenario: Scenario, config_from: Simulation | None = None) -> None:
        config = config_from.config if config_from is not None else scenario.config
        self.sim = Simulation(scenario.bodies, config)
        self.scenario = scenario
        self.scenario_label.configure(text=scenario.description)
        self._known_n = -1
        self.selected = None
        self._refresh_focus_choices()
        if config_from is None:
            self.steps_per_frame = scenario.steps_per_frame
            self._set_initial_focus(scenario.focus)
            self.reset_camera()
        else:
            # Body ids are assigned in scenario order, so the followed body keeps its id.
            if self._focus_index() is None and self.focus.kind == "body":
                self.focus = Focus("com")
            self.selected = self.focus.body_id
            self.v_focus.set(self._focus_label())
        self._sync_physics_widgets()
        self._mark_dirty()

    def _set_initial_focus(self, focus: str) -> None:
        if focus == "com":
            self.focus = Focus("com")
        elif focus == "origin":
            self.focus = Focus("origin")
        else:
            idx = self.sim.names.index(focus) if focus in self.sim.names else None
            self.focus = Focus("body", self.sim.ids[idx]) if idx is not None else Focus("com")
        self.selected = self.focus.body_id
        self.v_focus.set(self._focus_label())

    def _on_preset_selected(self) -> None:
        key = self._preset_key_from_title(self.v_preset.get())
        preset = PRESETS[key]
        self.preset_desc.configure(text=preset.description)
        state = "normal" if preset.default_bodies else "disabled"
        self.bodies_spin.configure(state=state, to=preset.max_bodies or 2)
        self.seed_spin.configure(state=state)
        if not preset.default_bodies:
            self.v_bodies.set("")  # fixed system
        elif key != self.preset_key or not self.v_bodies.get():
            self.v_bodies.set(str(preset.default_bodies))

    @staticmethod
    def _preset_key_from_title(title: str) -> str:
        for key, preset in PRESETS.items():
            if preset.title == title:
                return key
        raise KeyError(title)

    def load_selected_preset(self) -> None:
        key = self._preset_key_from_title(self.v_preset.get())
        preset = PRESETS[key]
        try:
            bodies = int(self.v_bodies.get()) if preset.default_bodies else None
            seed = int(self.v_seed.get())
            scenario = build_scenario(key, bodies, seed)
        except ValueError as exc:
            self._set_status(f"Cannot load scenario: {exc}", error=True)
            return
        self.preset_key = key
        self._load(scenario)
        self._set_status(f"Loaded {scenario.title} ({self.sim.n} bodies).")

    def restart(self) -> None:
        self._load(self.scenario, config_from=self.sim)
        self._set_status(f"Restarted {self.scenario.title} with the current settings.")

    # ============================================================ physics UI
    def _sync_physics_widgets(self) -> None:
        cfg = self.sim.config
        self.v_dt.set(f"{cfg.dt / DAY:g}")
        self.v_spf.set(str(self.steps_per_frame))
        self.v_solver.set(cfg.solver)
        self.v_theta.set(cfg.theta)
        self.v_softening.set(f"{cfg.softening / AU:g}")
        self.v_collisions.set(cfg.collisions)
        self.v_restitution.set(cfg.restitution)
        self.v_trail_len.set(str(cfg.trail_length))
        self.theta_label.configure(text=f"{cfg.theta:.2f}")
        self.rest_label.configure(text=f"{cfg.restitution:.2f}")
        self._update_speed_label()

    def _apply_physics(self, **changes: object) -> bool:
        try:
            self.sim.update_config(**changes)
        except ValueError as exc:
            self._set_status(f"Invalid setting: {exc}", error=True)
            self._sync_physics_widgets()
            return False
        self._update_speed_label()
        self._mark_dirty()
        return True

    def _apply_dt(self) -> None:
        try:
            days = float(self.v_dt.get())
        except ValueError:
            self._set_status("Time step must be a number of days.", error=True)
            self._sync_physics_widgets()
            return
        if days * DAY != self.sim.config.dt:
            self._apply_physics(dt=days * DAY)

    def _apply_spf(self) -> None:
        try:
            value = int(self.v_spf.get())
            if not 1 <= value <= 1000:
                raise ValueError
        except ValueError:
            self._set_status("Steps per frame must be an integer between 1 and 1000.", error=True)
            self.v_spf.set(str(self.steps_per_frame))
            return
        self.steps_per_frame = value
        self._update_speed_label()

    def _apply_softening(self) -> None:
        try:
            soft = float(self.v_softening.get()) * AU
        except ValueError:
            self._set_status("Softening must be a number (AU).", error=True)
            self._sync_physics_widgets()
            return
        if soft != self.sim.config.softening and self._apply_physics(softening=soft):
            self._set_status("Softening changed: energy baseline reset.")

    def _apply_trail_length(self) -> None:
        try:
            self._apply_physics(trail_length=int(self.v_trail_len.get()))
        except ValueError:
            self._set_status("Trail samples must be an integer.", error=True)
            self._sync_physics_widgets()

    def _on_theta_drag(self) -> None:
        theta = round(float(self.v_theta.get()), 2)
        self.theta_label.configure(text=f"{theta:.2f}")
        if theta != self.sim.config.theta:
            self._apply_physics(theta=theta)

    def _on_restitution_drag(self) -> None:
        e = round(float(self.v_restitution.get()), 2)
        self.rest_label.configure(text=f"{e:.2f}")
        if e != self.sim.config.restitution:
            self._apply_physics(restitution=e)

    def _update_speed_label(self) -> None:
        per_frame = self.sim.config.dt * self.steps_per_frame
        self.speed_label.configure(text=f"≈ {format_duration(per_frame * 1000 / FRAME_MS)} per real second at 60 fps")
        active = self.sim.active_solver
        note = "exact" if active == "direct" else f"θ = {self.sim.config.theta:.2f}"
        self.solver_label.configure(text=f"Active: {active} ({note})")

    # =============================================================== focus
    def _focus_choices(self) -> list[str]:
        return ["Centre of mass", "Origin", *self.sim.names]

    def _refresh_focus_choices(self) -> None:
        self.focus_box.configure(values=self._focus_choices())
        self._known_n = self.sim.n

    def _focus_label(self) -> str:
        if self.focus.kind == "com":
            return "Centre of mass"
        if self.focus.kind == "origin":
            return "Origin"
        idx = self._focus_index()
        return self.sim.names[idx] if idx is not None else "Centre of mass"

    def _focus_index(self) -> int | None:
        if self.focus.kind != "body" or self.focus.body_id is None:
            return None
        return self.sim.index_of(self.focus.body_id)

    def _on_focus_selected(self) -> None:
        label = self.v_focus.get()
        if label == "Centre of mass":
            self.set_focus(Focus("com"))
        elif label == "Origin":
            self.set_focus(Focus("origin"))
        elif label in self.sim.names:
            self.set_focus(Focus("body", self.sim.ids[self.sim.names.index(label)]))

    def set_focus(self, focus: Focus) -> None:
        self.focus = focus
        if focus.kind == "body":
            self.selected = focus.body_id
        self.camera.offset = np.zeros(3)
        self.v_focus.set(self._focus_label())
        self._last_inspect = 0.0
        self._mark_dirty()

    def cycle_focus(self, direction: int) -> None:
        if self.sim.n == 0:
            return
        idx = self._focus_index()
        nxt = 0 if idx is None else (idx + direction) % self.sim.n
        if idx is None and direction < 0:
            nxt = self.sim.n - 1
        self.set_focus(Focus("body", self.sim.ids[nxt]))

    def _focus_position(self) -> np.ndarray:
        if self.focus.kind == "com":
            return self.sim.center_of_mass()
        idx = self._focus_index()
        if idx is not None:
            return self.sim.pos[idx].copy()
        return np.zeros(3)

    def _focus_trail(self, last: int) -> np.ndarray | None:
        if self.focus.kind == "com":
            return self.sim.com_trail(last)
        if self.focus.kind == "body" and self.focus.body_id is not None:
            return self.sim.trail(self.focus.body_id, last)
        return None

    def _after_bodies_changed(self) -> None:
        """Keep focus/selection valid after merges."""
        if self.focus.kind == "body" and self.focus.body_id is not None:
            new = self.sim.resolve_id(self.focus.body_id)
            self.focus = Focus("body", new) if new is not None else Focus("com")
        if self.selected is not None:
            self.selected = self.sim.resolve_id(self.selected)
        self._refresh_focus_choices()
        self.v_focus.set(self._focus_label())

    # ============================================================ controls
    def toggle_running(self) -> None:
        self.running = not self.running
        self.play_button.configure(text="Pause" if self.running else "Play")
        self._mark_dirty()

    def single_step(self) -> None:
        if self.running:
            self.toggle_running()
        self._advance(1)
        self._mark_dirty()

    def reset_camera(self) -> None:
        self.camera = Camera(elevation=self.scenario.elevation, fov=float(self.v_fov.get()))
        self.camera.fit(self.scenario.view_radius)
        self._mark_dirty()

    def clear_trails(self) -> None:
        self.sim.clear_trails()
        self._mark_dirty()

    def toggle_fullscreen(self) -> None:
        self.fullscreen = not self.fullscreen
        self.root.attributes("-fullscreen", self.fullscreen)

    def _escape(self) -> None:
        if self.v_help.get():
            self.v_help.set(False)
            self._mark_dirty()
        elif self.fullscreen:
            self.toggle_fullscreen()

    def _toggle(self, var: tk.BooleanVar) -> None:
        var.set(not var.get())
        self._mark_dirty()

    def _key(self, event: tk.Event, fn: Callable[[], object]) -> None:
        # Let text fields receive their own keystrokes, and let a focused
        # button/checkbox handle Space itself (standard Tk behaviour).
        if isinstance(event.widget, (tk.Entry, ttk.Entry, ttk.Spinbox, ttk.Combobox)) and event.keysym not in (
            "F1",
            "F11",
            "Escape",
        ):
            return
        if isinstance(event.widget, (ttk.Button, ttk.Checkbutton, ttk.Radiobutton)) and event.keysym == "space":
            return
        fn()

    def _on_fov(self) -> None:
        self.camera.set_fov(float(self.v_fov.get()))
        self._mark_dirty()

    def _mark_dirty(self) -> None:
        self.dirty = True

    def _set_status(self, text: str, error: bool = False) -> None:
        self.status.configure(text=text, style="StatusError.TLabel" if error else "Status.TLabel")

    # =============================================================== mouse
    def _press(self, event: tk.Event, mode: str) -> None:
        self.canvas.focus_set()  # keyboard shortcuts apply to the view again
        self._drag = (mode, event.x, event.y, False)

    def _motion(self, event: tk.Event) -> None:
        if self._drag is None:
            return
        mode, x0, y0, moved = self._drag
        dx, dy = event.x - x0, event.y - y0
        if not moved and abs(dx) + abs(dy) < 3:
            return
        w, h = self._canvas_size()
        if mode == "orbit":
            self.camera.orbit(-dx * 0.35, dy * 0.35)
        else:
            self.camera.pan_pixels(dx, dy, w, h)
        self._drag = (mode, event.x, event.y, True)
        self._mark_dirty()

    def _release(self, event: tk.Event) -> None:
        if self._drag is not None and self._drag[0] == "orbit" and not self._drag[3]:
            self._pick(event.x, event.y)
        self._drag = None

    def _wheel(self, event: tk.Event) -> None:
        if event.delta:
            self._zoom_by(0.88 ** (event.delta / 120))

    def _zoom_by(self, factor: float) -> None:
        self.camera.zoom(factor)
        self._mark_dirty()

    def _pick(self, x: int, y: int) -> None:
        w, h = self._canvas_size()
        proj = self.camera.project(self.sim.pos - self._focus_position(), w, h)
        radii = self._screen_radii(proj.scale)
        d = np.hypot(proj.x - x, proj.y - y)
        hit = proj.visible & (d <= np.maximum(radii + 4, 9))
        if not hit.any():
            self.selected = None
            self._last_inspect = 0.0
            self._mark_dirty()
            return
        candidates = np.nonzero(hit)[0]
        idx = int(candidates[np.argmin(proj.depth[candidates])])  # front-most
        self.set_focus(Focus("body", self.sim.ids[idx]))
        self._set_status(f"Following {self.sim.names[idx]}.")

    # ================================================================ loop
    def _advance(self, steps: int) -> None:
        """Run up to ``steps`` steps, stopping early if the frame's physics budget is spent."""
        t0 = time.perf_counter()
        deadline = t0 + PHYSICS_BUDGET_S
        done = 0
        try:
            while done < steps:
                self.sim.step(1)
                done += 1
                if time.perf_counter() > deadline:
                    break
        except NumericalInstabilityError as exc:
            if self.running:
                self.toggle_running()
            self._set_status(f"Paused: {exc}.", error=True)
        self._steps_done = done
        self._steps_wanted = steps
        self._phys_ms = (time.perf_counter() - t0) * 1000
        if self.sim.n != self._known_n:
            self._after_bodies_changed()

    def _tick(self) -> None:
        if self.closed:
            return
        start = time.perf_counter()
        if self.running:
            self._advance(self.steps_per_frame)
            self.dirty = True
        if self.dirty:
            t0 = time.perf_counter()
            self.render()
            self._draw_ms = (time.perf_counter() - t0) * 1000
            self.dirty = False
            self._maybe_record()
        if time.perf_counter() - self._last_inspect > INSPECT_INTERVAL:
            self._update_inspector()
            self._last_inspect = time.perf_counter()
        elapsed = time.perf_counter() - start
        interval = start - self._last_tick
        self._last_tick = start
        if self.running and interval > 0:
            self._fps = 0.9 * self._fps + 0.1 / interval
        # Always leave a few ms so Tk can repaint and handle input between frames.
        delay = max(5, int(FRAME_MS - elapsed * 1000))
        self._after_id = self.root.after(delay, self._tick)

    def close(self) -> None:
        self.closed = True
        if self._after_id is not None:
            self.root.after_cancel(self._after_id)
        self.root.destroy()

    # ============================================================ rendering
    def _canvas_size(self) -> tuple[int, int]:
        return max(self.canvas.winfo_width(), 2), max(self.canvas.winfo_height(), 2)

    def _dim(self, color: str, t: float) -> str:
        key = (color, round(t, 2))
        if key not in self._color_cache:
            self._color_cache[key] = blend(color, BG, t)
        return self._color_cache[key]

    def _screen_radii(self, scale: np.ndarray) -> np.ndarray:
        """On-screen radius (px) of every body."""
        if self.v_true_scale.get():
            return np.maximum(self.sim.radius * scale, 1.0)
        target = self.camera.pixels_per_meter(*self._canvas_size())
        depth_factor = np.sqrt(np.clip(scale / target, 0.25, 4.0))
        n = self.sim.n
        # Exaggerated sizes ~ radius^0.3 (Earth = base px); crowded scenes get small dots.
        base, cap = (3.2, 36.0) if n <= CROWDED else (1.3, 7.0) if n <= 300 else (1.0, 5.0)
        size = base * (self.sim.radius / EARTH_RADIUS) ** 0.3
        is_bh = np.array([k == "blackhole" for k in self.sim.kinds])
        size = np.where(is_bh, base * 2.2 if n <= CROWDED else cap * 0.8, size)
        return np.clip(size * depth_factor, 1.2, cap)

    def _body_colors(self) -> tuple[list[str], str | None]:
        mode = self._color_mode()
        sim = self.sim
        if mode == "body" or sim.n == 0:
            return list(sim.colors), None
        if mode == "speed":
            values = np.linalg.norm(sim.vel - sim.com_velocity(), axis=1)
            lo, hi = 0.0, float(np.percentile(values, 98)) or 1.0
            norm = values / hi
            legend = f"speed 0 – {format_speed(hi)}"
        elif mode == "acceleration":
            values = np.log10(np.linalg.norm(sim.acc, axis=1) + 1e-30)
            lo, hi = float(np.percentile(values, 2)), float(np.percentile(values, 98))
            norm = (values - lo) / max(hi - lo, 1e-9)
            legend = f"acceleration 10^{lo:.1f} – 10^{hi:.1f} m/s²"
        else:
            values = np.log10(sim.mass)
            lo, hi = float(values.min()), float(values.max())
            norm = (values - lo) / max(hi - lo, 1e-9)
            legend = f"mass {format_mass(10**lo)} – {format_mass(10**hi)}"
        return [colormap(float(v)) for v in norm], legend

    def _color_mode(self) -> ColorBy:
        label = self.v_color_by.get()
        for key, text in COLOR_BY_LABELS.items():
            if text == label:
                return key
        return "body"

    @staticmethod
    def _runs(x: np.ndarray, y: np.ndarray, visible: np.ndarray) -> list[list[float]]:
        """Split a projected polyline into drawable runs of visible points."""
        runs: list[list[float]] = []
        x = np.clip(x, -MAX_COORD, MAX_COORD)
        y = np.clip(y, -MAX_COORD, MAX_COORD)
        idx = np.nonzero(visible)[0]
        if idx.size < 2:
            return runs
        breaks = np.nonzero(np.diff(idx) > 1)[0] + 1
        for seg in np.split(idx, breaks):
            if seg.size >= 2:
                runs.append(np.column_stack((x[seg], y[seg])).ravel().tolist())
        return runs

    def render(self) -> None:
        c = self.canvas
        c.delete("all")
        w, h = self._canvas_size()
        sim = self.sim
        cam = self.camera
        focus = self._focus_position()

        if self.v_octree.get():
            self._draw_octree(focus, w, h)
        colors, legend = self._body_colors()
        if self.v_trails.get():
            self._draw_trails(focus, colors, w, h)

        proj = cam.project(sim.pos - focus, w, h)
        radii = self._screen_radii(proj.scale)

        if self.v_com.get():
            p = cam.project(sim.center_of_mass() - focus, w, h)
            if p.visible[0]:
                x, y = float(p.x[0]), float(p.y[0])
                c.create_line(x - 7, y, x + 7, y, fill=COM_COLOR)
                c.create_line(x, y - 7, x, y + 7, fill=COM_COLOR)
                c.create_oval(x - 4, y - 4, x + 4, y + 4, outline=COM_COLOR)

        order = np.argsort(-proj.depth)  # far to near
        halos = sim.n <= CROWDED
        sel_idx = sim.index_of(self.selected) if self.selected is not None else None
        on_screen = proj.visible & (np.abs(proj.x) < MAX_COORD) & (np.abs(proj.y) < MAX_COORD)
        for i in order.tolist():
            if not on_screen[i]:
                continue
            x, y, r = float(proj.x[i]), float(proj.y[i]), float(radii[i])
            self._draw_body(sim.kinds[i], x, y, r, colors[i], halos)
            if i == sel_idx:
                c.create_oval(x - r - 5, y - r - 5, x + r + 5, y + r + 5, outline=ACCENT, width=1.5)
        self._draw_labels(proj.x, proj.y, radii, on_screen, sel_idx)

        if self.v_axes.get():
            self._draw_axes(w, h)
        self._draw_scale_bar(w, h)
        if self.v_hud.get():
            self._draw_hud(w, h, legend)
        if not self.running:
            c.create_text(w / 2, 22, text="PAUSED  •  space to resume", fill=ACCENT, font=self.mono)
        if self.v_help.get():
            self._draw_help(w, h)

    def _draw_labels(self, xs: np.ndarray, ys: np.ndarray, radii: np.ndarray, on_screen: np.ndarray, sel_idx: int | None) -> None:
        """Name labels, most massive first, skipping any that would overlap one already placed."""
        sim = self.sim
        candidates: list[int] = []
        if self.v_labels.get() and sim.n <= CROWDED:
            candidates = [int(i) for i in np.argsort(-sim.mass) if on_screen[i]]
        if sel_idx is not None and on_screen[sel_idx]:
            candidates = [sel_idx] + [i for i in candidates if i != sel_idx]
        char_w = self.mono_small.measure("M")
        line_h = self.mono_small.metrics("linespace")
        placed: list[tuple[float, float, float, float]] = []
        for i in candidates:
            x0 = float(xs[i] + radii[i] + 6)
            y0 = float(ys[i] - radii[i] - 2 - line_h / 2)
            box = (x0, y0, x0 + char_w * len(sim.names[i]), y0 + line_h)
            if any(box[0] < b[2] and b[0] < box[2] and box[1] < b[3] and b[1] < box[3] for b in placed):
                continue
            placed.append(box)
            color = FG if i == sel_idx else MUTED
            self.canvas.create_text(box[0], box[1], text=sim.names[i], fill=color, anchor="nw", font=self.mono_small)

    def _draw_body(self, kind: BodyKind, x: float, y: float, r: float, color: str, halo: bool = True) -> None:
        c = self.canvas
        if kind == "blackhole":
            if halo:
                c.create_oval(x - r * 1.9, y - r * 1.9, x + r * 1.9, y + r * 1.9, outline=self._dim(color, 0.65), width=1)
            c.create_oval(x - r, y - r, x + r, y + r, fill="#000000", outline=color, width=2)
            return
        if halo and kind == "star" and r >= 3:
            g = r * 1.8
            c.create_oval(x - g, y - g, x + g, y + g, fill=self._dim(color, 0.82), outline="")
        c.create_oval(x - r, y - r, x + r, y + r, fill=color, outline="")

    def _draw_trails(self, focus: np.ndarray, colors: list[str], w: int, h: int) -> None:
        sim, cam, c = self.sim, self.camera, self.canvas
        relative = self.v_relative.get() and self.focus.kind != "origin"
        cap = sim.config.trail_length
        ref = self._focus_trail(cap) if relative else None
        crowded = sim.n > CROWDED
        segments: list[tuple[int, np.ndarray]] = []
        for i, bid in enumerate(sim.ids):
            pts = sim.trail(bid)
            if len(pts) < 2:
                continue
            if ref is not None:
                m = min(len(pts), len(ref))
                pts = pts[-m:] - ref[-m:]
            else:
                pts = pts - focus
            segments.append((i, np.vstack((pts, sim.pos[i] - focus))))  # connect the trail to the body
        if not segments:
            return
        # One projection call for every trail point (far cheaper than one per body).
        p = cam.project(np.concatenate([s for _, s in segments]), w, h)
        start = 0
        for i, pts in segments:
            n = len(pts)
            xs, ys, vis = p.x[start : start + n], p.y[start : start + n], p.visible[start : start + n]
            start += n
            if crowded:  # one faint line per body
                for run in self._runs(xs, ys, vis):
                    c.create_line(*run, fill=self._dim(colors[i], 0.6), width=1)
                continue
            # Older thirds are drawn dimmer and thinner: a cheap fade.
            bounds = [0, n // 3, (2 * n) // 3, n]
            for k in range(3):
                lo, hi = max(bounds[k] - 1, 0), bounds[k + 1]
                if hi - lo < 2:
                    continue
                color = self._dim(colors[i], (0.78, 0.55, 0.25)[k])
                for run in self._runs(xs[lo:hi], ys[lo:hi], vis[lo:hi]):
                    c.create_line(*run, fill=color, width=(1.0, 1.3, 1.8)[k])

    def _draw_octree(self, focus: np.ndarray, w: int, h: int) -> None:
        if self.sim.n == 0:
            return
        kind = next((k for k, v in OCTREE_LABELS.items() if v == self.v_octree_kind.get()), "internal")
        boxes = self.sim.octree().boxes(kind, limit=MAX_OCTREE_BOXES)  # type: ignore[arg-type]
        if not boxes:
            return
        centers = np.array([b[0] for b in boxes]) - focus
        halves = np.array([b[1] for b in boxes])
        signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=float)
        corners = centers[:, None, :] + halves[:, None, None] * signs[None, :, :]
        p = self.camera.project(corners.reshape(-1, 3), w, h)
        xs, ys, vis = p.x.reshape(-1, 8), p.y.reshape(-1, 8), p.visible.reshape(-1, 8)
        # Corner index = 4*sx + 2*sy + sz (0 = negative). Path covering all 12 edges.
        path = [0, 1, 3, 2, 0, 4, 5, 7, 6, 4, 6, 2, 3, 7, 5, 1]
        color = GRID
        for b in range(len(boxes)):
            if not vis[b].all():
                continue
            coords = np.column_stack((xs[b, path], ys[b, path])).clip(-MAX_COORD, MAX_COORD).ravel().tolist()
            self.canvas.create_line(*coords, fill=color, width=1)

    def _draw_axes(self, w: int, h: int) -> None:
        c = self.canvas
        cx, cy, length = w - 60, h - 60, 34
        axes = self.camera.axis_screen_directions()
        colors = ("#e5736b", "#7ccf8a", "#6fa8ff")
        # Draw the axis pointing away from the viewer first.
        for k in sorted(range(3), key=lambda k: -axes[k][2]):
            dx, dy, depth = axes[k]
            front = depth < 0
            c.create_line(
                cx,
                cy,
                cx + dx * length,
                cy + dy * length,
                fill=colors[k] if front else self._dim(colors[k], 0.4),
                width=2.5 if front else 1.5,
            )
            c.create_text(cx + dx * (length + 10), cy + dy * (length + 10), text="XYZ"[k], fill=colors[k], font=self.mono_small)

    def _draw_scale_bar(self, w: int, h: int) -> None:
        ppm = self.camera.pixels_per_meter(w, h)
        target_px = 120
        meters = target_px / ppm
        unit, unit_m = ("AU", AU) if meters >= 0.01 * AU else ("km", 1e3)
        value = meters / unit_m
        nice = 10 ** math.floor(math.log10(value))
        for m in (5, 2, 1):
            if m * nice <= value:
                nice *= m
                break
        px = nice * unit_m * ppm
        x0, y0 = 20, h - 24
        c = self.canvas
        c.create_line(x0, y0, x0 + px, y0, fill=MUTED, width=2)
        c.create_line(x0, y0 - 4, x0, y0 + 4, fill=MUTED)
        c.create_line(x0 + px, y0 - 4, x0 + px, y0 + 4, fill=MUTED)
        c.create_text(x0, y0 - 12, text=f"{nice:g} {unit} (at focus)", fill=MUTED, anchor="w", font=self.mono_small)

    def _draw_hud(self, w: int, h: int, legend: str | None) -> None:
        sim = self.sim
        energy = sim.energy()
        drift = energy.drift
        drift_color = GOOD if abs(drift) < 1e-4 else WARN if abs(drift) < 1e-2 else BAD
        if not math.isfinite(drift):
            drift_color = MUTED
        c = self.canvas
        lines = [
            (f"t = {format_sim_time(sim.time)}", FG),
            (f"{sim.n} bodies • dt {format_duration(sim.config.dt)} × {self.steps_per_frame}/frame", MUTED),
            (f"{sim.active_solver} solver • following {self._focus_label()}", MUTED),
        ]
        y = 16
        for text, color in lines:
            c.create_text(16, y, text=text, fill=color, anchor="nw", font=self.mono)
            y += 18
        c.create_text(
            16, y, text=f"ΔE/E₀ {drift:+.2e}{'' if energy.exact else ' (approx.)'}", fill=drift_color, anchor="nw", font=self.mono
        )
        if legend:
            c.create_text(16, y + 20, text=f"colour: {legend}", fill=MUTED, anchor="nw", font=self.mono_small)
        perf = f"{self._fps:4.0f} fps • physics {self._phys_ms:5.1f} ms • draw {self._draw_ms:5.1f} ms"
        c.create_text(w - 16, 16, text=perf, fill=MUTED, anchor="ne", font=self.mono_small)
        if self.running and self._steps_done < self._steps_wanted:
            limited = f"physics-limited: {self._steps_done}/{self._steps_wanted} steps per frame"
            c.create_text(w - 16, 52, text=limited, fill=WARN, anchor="ne", font=self.mono_small)
        if self.record_frames_left > 0:
            c.create_text(w - 16, 34, text=f"● REC {self.record_frames_left}", fill=BAD, anchor="ne", font=self.mono_small)

    def _draw_help(self, w: int, h: int) -> None:
        c = self.canvas
        bw, bh = 420, 300
        x0, y0 = (w - bw) / 2, (h - bh) / 2
        c.create_rectangle(x0, y0, x0 + bw, y0 + bh, fill=PANEL, outline=BORDER)
        c.create_text(x0 + 20, y0 + 18, text=HELP_TEXT, fill=FG, anchor="nw", font=self.mono_small)
        c.create_text(x0 + bw - 14, y0 + bh - 14, text="Esc to close", fill=MUTED, anchor="se", font=self.mono_small)

    # ============================================================ inspector
    def _update_inspector(self) -> None:
        sim = self.sim
        idx = sim.index_of(self.selected) if self.selected is not None else self._focus_index()
        if idx is None:
            self.body_title.configure(text="No body selected")
            for lbl in self.body_rows.values():
                lbl.configure(text="—")
            self.body_rows["Type"].configure(text="click a body to inspect it")
        else:
            b = sim.body(idx)
            if self._focus_index() == idx:
                distance = f"{format_distance(float(np.linalg.norm(b.position)))} from origin"
            else:
                gap = float(np.linalg.norm(b.position - self._focus_position()))
                distance = f"{format_distance(gap)} from {self._focus_label()}"
            self.body_title.configure(text=b.name)
            values = {
                "Type": b.kind,
                "Mass": format_mass(b.mass),
                "Radius": format_distance(b.radius),
                "Density": "—" if b.kind == "blackhole" else format_density(b.density),
                "Distance": distance,
                "Speed": f"{format_speed(b.speed)} (inertial)",
                "Acceleration": format_acceleration(float(np.linalg.norm(b.acceleration))),
            }
            for key, text in values.items():
                self.body_rows[key].configure(text=text)

        e = sim.energy()
        tree = "—"
        if self.v_octree.get() or sim.active_solver == "barnes-hut":
            st = sim.octree().stats()
            tree = f"{st['nodes']} nodes, depth {st['depth']}"
        drift = e.drift
        rows = {
            "Time": format_sim_time(sim.time),
            "Steps": f"{sim.steps:,}",
            "Bodies": f"{sim.n:,} ({len(sim.merged_into)} merged)",
            "Total mass": format_mass(sim.total_mass()),
            "Kinetic E": format_energy(e.kinetic),
            "Potential E": format_energy(e.potential) + ("" if e.exact else " ~"),
            "Total E": format_energy(e.total),
            "Energy drift": f"{drift:+.3e}" if math.isfinite(drift) else "undefined (E₀ = 0)",
            "Collision ΔE": format_energy(e.collision_energy),
            "|Momentum|": f"{float(np.linalg.norm(sim.momentum())):.3e} kg·m/s",
            "Collisions": f"{sim.collision_events:,}",
            "Octree": tree,
        }
        for key, text in rows.items():
            self.sys_rows[key].configure(text=text)

    # ============================================================== capture
    def _grab_canvas(self) -> object:
        from PIL import ImageGrab  # optional dependency

        c = self.canvas
        x, y = c.winfo_rootx(), c.winfo_rooty()
        return ImageGrab.grab(bbox=(x, y, x + c.winfo_width(), y + c.winfo_height()))

    def save_image(self) -> None:
        path = filedialog.asksaveasfilename(
            parent=self.root,
            title="Save image",
            defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("Encapsulated PostScript", "*.eps")],
        )
        if path:
            self.save_image_to(Path(path))

    def save_image_to(self, path: Path) -> bool:
        """Save the canvas as ``.eps`` (built in) or a raster image such as ``.png`` (needs Pillow)."""
        try:
            if path.suffix.lower() == ".eps":
                self.canvas.postscript(file=str(path), colormode="color")  # type: ignore[no-untyped-call]
            else:
                self.root.update_idletasks()
                self._grab_canvas().save(path)  # type: ignore[attr-defined]
        except ImportError:
            messagebox.showerror(
                "Save image", "Saving PNG needs Pillow (pip install pillow). You can save as .eps instead.", parent=self.root
            )
            return False
        except (OSError, ValueError, tk.TclError) as exc:
            messagebox.showerror("Save image", f"Could not save the image:\n{exc}", parent=self.root)
            return False
        self._set_status(f"Saved {path}")
        return True

    def _maybe_record(self) -> None:
        if self.record_frames_left <= 0 or self.record_dir is None or not self.running:
            return
        try:
            self.record_dir.mkdir(parents=True, exist_ok=True)
            self._grab_canvas().save(self.record_dir / f"frame_{self.frame_count:05d}.png")  # type: ignore[attr-defined]
        except (ImportError, OSError) as exc:
            self.record_frames_left = 0
            self._set_status(f"Recording stopped: {exc}", error=True)
            return
        self.frame_count += 1
        self.record_frames_left -= 1
        if self.record_frames_left == 0:
            self._set_status(f"Recording finished: {self.frame_count} frames in {self.record_dir}")


def _enable_dpi_awareness() -> None:
    """Crisp rendering and correct screen-capture coordinates on Windows."""
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except (AttributeError, OSError):
            pass


def run_app(
    scenario: Scenario,
    preset_key: str,
    bodies: int | None = None,
    seed: int = 1,
    fullscreen: bool = False,
    paused: bool = False,
    record_dir: Path | None = None,
    record_frames: int = 3600,
) -> int:
    _enable_dpi_awareness()
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        print(f"gravity-sim: error: cannot open a window ({exc}); use --headless", file=sys.stderr)
        return 2
    app = GravityApp(root, scenario, preset_key, bodies, seed, paused, record_dir, record_frames)
    if fullscreen:
        app.toggle_fullscreen()
    root.mainloop()
    return 0
