import csv
from pathlib import Path

import numpy as np
import pytest

from gravity_sim import cli, forces
from gravity_sim import simulation as simmod


def run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str, str]:
    code = cli.main(list(argv))
    out, err = capsys.readouterr()
    return code, out, err


def test_list_presets(capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = run(capsys, "--list-presets")
    assert code == 0
    for key in ("inner", "solar", "binary", "figure8", "cluster", "galaxies"):
        assert key in out


def test_headless_run_reports_and_writes_csv(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    out_csv = tmp_path / "out" / "diag.csv"
    code, out, err = run(
        capsys, "--headless", "--preset", "inner", "--steps", "40", "--report-every", "10", "--csv", str(out_csv)
    )
    assert code == 0, err
    out.encode("ascii")  # console output must work on any code page
    assert "done: 40 steps" in out
    rows = list(csv.DictReader(out_csv.open(encoding="utf-8")))
    assert [int(r["step"]) for r in rows] == [0, 10, 20, 30, 40]
    assert abs(float(rows[-1]["energy_drift"])) < 1e-5


def test_headless_overrides_are_applied(capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = run(
        capsys,
        "--headless",
        "--preset",
        "cluster",
        "--bodies",
        "30",
        "--dt",
        "0.1",
        "--solver",
        "barnes-hut",
        "--theta",
        "0.7",
        "--softening",
        "0.3",
        "--collisions",
        "off",
        "--steps",
        "5",
    )
    assert code == 0
    assert "30 bodies, dt = 0.1 d, solver = barnes-hut, collisions = off, softening = 0.3 AU" in out


@pytest.mark.parametrize(
    "argv",
    [
        ["--headless", "--dt", "-1"],
        ["--headless", "--preset", "inner", "--bodies", "10"],
        ["--headless", "--restitution", "3"],
        ["--headless", "--steps", "0"],
        ["--headless", "--steps-per-frame", "0"],
    ],
)
def test_invalid_arguments_exit_with_usage_error(capsys: pytest.CaptureFixture[str], argv: list[str]) -> None:
    code, _, err = run(capsys, *argv)
    assert code == cli.EXIT_USAGE
    assert "error" in err


def test_instability_exits_with_dedicated_code(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(pos: np.ndarray, mass: np.ndarray, softening: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        return np.full_like(pos, np.inf), np.zeros(len(mass))

    original = forces.direct_accelerations
    calls = {"n": 0}

    def flaky(pos: np.ndarray, mass: np.ndarray, softening: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        calls["n"] += 1
        return original(pos, mass, softening) if calls["n"] < 5 else broken(pos, mass, softening)

    monkeypatch.setattr(simmod, "direct_accelerations", flaky)
    code, out, _ = run(capsys, "--headless", "--steps", "20", "--report-every", "5")
    assert code == cli.EXIT_UNSTABLE
    assert "non-finite" in out
