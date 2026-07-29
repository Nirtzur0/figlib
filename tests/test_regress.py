"""Golden-figure regression harness: compare a fresh render against the
committed baseline SVG, and quantify any drift in pixels.

These tests use a synthetic figure program written into tmp_path; they
never touch the real figures/ corpus.
"""

import textwrap

import pytest

from figlib.regress import compare_figure, discover_programs

TOY = textwrap.dedent('''
    import numpy as np
    from figlib.scene import Curve, MathLabel, Scene
    from figlib.style import Role

    CLAIM = "The scaled parabola touches the origin."
    PARAMS = {"n": 51, "a": 1.0}

    def compute(p):
        x = np.linspace(-1, 1, p["n"])
        return np.column_stack([x, p["a"] * x**2])

    def build(g):
        return Scene(items=[
            Curve(g, role=Role.CONTENT),
            MathLabel(r"y = a x^2", (0.35, 0.75), ha="left"),
        ])

    def assertions(g):
        assert abs(g[:, 1].min()) < 1e-12, "parabola should touch zero"
''')


@pytest.fixture()
def toy(tmp_path):
    """(program_path, baseline_dir) with the baseline already committed."""
    prog = tmp_path / "toy_regress.py"
    prog.write_text(TOY)
    out = tmp_path / "out"
    from figlib.program import run
    run(prog, out_dir=out)
    return prog, out


def test_match_is_byte_identical(toy):
    """Re-rendering an untouched program reproduces the baseline exactly.

    This is also the determinism check: the renderer must be a pure
    function of the program, so two runs agree on every byte of SVG.
    """
    prog, out = toy
    first = compare_figure(prog, out)
    second = compare_figure(prog, out)
    assert first.status == "match", first.detail
    assert second.status == "match", second.detail
    assert first.changed_fraction == 0.0
    assert first.ok


def test_missing_baseline_is_new(tmp_path):
    prog = tmp_path / "toy_regress.py"
    prog.write_text(TOY)
    res = compare_figure(prog, tmp_path / "empty_out")
    assert res.status == "new"
    assert res.ok


def test_perturbed_param_is_drift_with_pixel_metric(toy):
    prog, out = toy
    prog.write_text(TOY.replace('"a": 1.0', '"a": 0.55'))
    res = compare_figure(prog, out)
    assert res.status == "drift"
    assert not res.ok
    assert 0.0 < res.changed_fraction <= 1.0
    assert res.rms > 0.0


def test_cosmetic_drift_is_smaller_than_structural(toy):
    """The metric must separate a nudged label from a redrawn curve."""
    prog, out = toy
    prog.write_text(TOY.replace('(0.35, 0.75)', '(0.36, 0.75)'))
    cosmetic = compare_figure(prog, out)
    prog.write_text(TOY.replace('"a": 1.0', '"a": 0.2'))
    structural = compare_figure(prog, out)
    assert cosmetic.status == structural.status == "drift"
    assert cosmetic.changed_fraction < structural.changed_fraction


def test_raising_program_is_error(toy):
    prog, out = toy
    perturbed = TOY.replace('    x = np.linspace(-1, 1, p["n"])',
                            "    raise ValueError('boom')")
    assert perturbed != TOY
    prog.write_text(perturbed)
    res = compare_figure(prog, out)
    assert res.status == "error"
    assert not res.ok
    assert "boom" in res.detail


def test_failed_gate_is_error(toy):
    """A program that renders but fails its own gates is not a silent pass."""
    prog, out = toy
    prog.write_text(TOY.replace("< 1e-12", "> 1.0"))
    res = compare_figure(prog, out)
    assert res.status == "error"
    assert "numerical" in res.detail


def test_discover_skips_private_and_out(tmp_path):
    figs = tmp_path / "figures"
    (figs / "out").mkdir(parents=True)
    (figs / "alpha.py").write_text("")
    (figs / "beta.py").write_text("")
    (figs / "_helper.py").write_text("")
    (figs / "out" / "gamma.py").write_text("")
    (figs / "notes.md").write_text("")
    assert [p.name for p in discover_programs(figs)] == ["alpha.py", "beta.py"]


def test_discover_walks_subject_subdirectories(tmp_path):
    """The corpus bins programs by subject one level deep; out/ still holds
    artifacts, not programs, at any depth."""
    figs = tmp_path / "figures"
    (figs / "complex").mkdir(parents=True)
    (figs / "signals").mkdir()
    (figs / "out" / "complex").mkdir(parents=True)
    (figs / "toplevel.py").write_text("")
    (figs / "complex" / "alpha.py").write_text("")
    (figs / "complex" / "_helper.py").write_text("")
    (figs / "signals" / "beta.py").write_text("")
    (figs / "out" / "complex" / "gamma.py").write_text("")
    found = [str(p.relative_to(figs)) for p in discover_programs(figs)]
    assert found == ["complex/alpha.py", "signals/beta.py", "toplevel.py"]


def test_discover_does_not_recurse_two_levels(tmp_path):
    """One level only — a nested subject tree is not a thing, and silently
    picking up a scratch program two levels down would poison the sweep."""
    figs = tmp_path / "figures"
    (figs / "complex" / "scratch").mkdir(parents=True)
    (figs / "complex" / "alpha.py").write_text("")
    (figs / "complex" / "scratch" / "deep.py").write_text("")
    found = [str(p.relative_to(figs)) for p in discover_programs(figs)]
    assert found == ["complex/alpha.py"]


def test_artifact_dir_mirrors_subject(tmp_path):
    from figlib.regress import artifact_dir

    figs = tmp_path / "figures"
    assert artifact_dir(figs / "complex" / "x.py", figs) == figs / "out" / "complex"
    assert artifact_dir(figs / "x.py", figs) == figs / "out"


def test_sweep_finds_baselines_in_subject_subdirs(tmp_path):
    """A subject-binned program's baseline lives in the mirrored out/ dir;
    the sweep must look there and not in a flat out/."""
    from figlib.program import run
    from figlib.regress import sweep

    figs = tmp_path / "figures"
    (figs / "complex").mkdir(parents=True)
    prog = figs / "complex" / "toy_regress.py"
    prog.write_text(TOY)
    run(prog, out_dir=figs / "out" / "complex")
    run(prog, out_dir=figs / "out" / "complex", paper=True)

    # both grounds are committed for every program, so the sweep compares two
    results = sweep(figs)
    assert [r.status for r in results] == ["match", "match"]


def test_cli_regress_exit_code_tracks_drift(tmp_path, capsys):
    """The sweep is only useful in CI if drift actually fails the build."""
    from figlib.cli import main

    figs = tmp_path / "figures"
    figs.mkdir()
    (figs / "toy_regress.py").write_text(TOY)
    argv = ["--regress", "--figures-dir", str(figs)]

    assert main(argv) == 0                      # no baseline yet -> NEW
    assert "NEW" in capsys.readouterr().out
    assert main(["--update", "--figures-dir", str(figs)]) == 0
    assert main(argv) == 0                      # baseline now matches

    (figs / "toy_regress.py").write_text(TOY.replace('"a": 1.0', '"a": 0.55'))
    assert main(argv) == 1
    assert "DRIFT" in capsys.readouterr().out


def test_compare_does_not_touch_the_baseline_dir(toy):
    prog, out = toy
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    prog.write_text(TOY.replace('"a": 1.0', '"a": 0.55'))
    compare_figure(prog, out)
    after = {p.name: p.read_bytes() for p in out.iterdir()}
    assert before == after
