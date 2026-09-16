"""Tests for furniture-path and maple MDL rewrite helpers (no Isaac)."""

from pathlib import Path

from rlinf.envs.isaaclab.tasks.pick_place_cube_plate.scene_guard import (
    ensure_local_maple_usd,
    is_broken_mdl_ref,
    is_fixture_prim_path,
    is_unresolved_mdl_ref,
    resolve_local_mdl,
    rewrite_maple_usda,
)

_SAMPLE = """
    bool physics:kinematicEnabled = 0
    bool physics:rigidBodyEnabled = 1
                uniform asset info:mdl:sourceAsset = @../../../Materials/Base/Wood/Oak.mdl@
                uniform token info:mdl:sourceAsset:subIdentifier = "Oak"
                uniform asset info:mdl:sourceAsset = @../../../Materials/vMaterials_2/Plastic/Plastic_Thick_Translucent.mdl@
                uniform token info:mdl:sourceAsset:subIdentifier = "plastic_blue"
                uniform asset info:mdl:sourceAsset = @../../../Materials/vMaterials_2/Paint/Carpaint/Carpaint_Solid.mdl@
                uniform token info:mdl:sourceAsset:subIdentifier = "Black_Matte"
"""


def test_fixture_paths():
    assert is_fixture_prim_path("/World/envs/env_53/MapleTable/table_01/top")
    assert is_fixture_prim_path("/World/envs/env_0/franka_table")
    assert is_fixture_prim_path("/World/envs/env_2/Robot_Stand")
    assert not is_fixture_prim_path("/World/envs/env_53/Robot/panda_link0")
    assert not is_fixture_prim_path("/World/envs/env_0/Cube")


def test_broken_mdl_refs():
    assert is_broken_mdl_ref(
        "../../../Materials/vMaterials_2/Plastic/Plastic_Thick_Translucent.mdl"
    )
    assert is_broken_mdl_ref(".../Paint/Carpaint/Carpaint_Solid.mdl")
    assert not is_broken_mdl_ref("../../../Materials/Base/Wood/Oak.mdl")


def test_unresolved_relative_oak():
    assert is_unresolved_mdl_ref("../../../Materials/Base/Wood/Oak.mdl")
    assert not is_unresolved_mdl_ref("/abs/Materials/Base/Wood/Oak.mdl")


def test_resolve_local_mdl_maps_vmaterials(tmp_path: Path | None = None):
    root = Path(tmp_path) if tmp_path is not None else Path("/tmp/rlinf_mdl_test")
    plastic = root / "Base/Plastics/Plastic.mdl"
    oak = root / "Base/Wood/Oak.mdl"
    plastic.parent.mkdir(parents=True, exist_ok=True)
    oak.parent.mkdir(parents=True, exist_ok=True)
    plastic.write_text("mdl")
    oak.write_text("mdl")
    abs_oak, name = resolve_local_mdl("../../../Materials/Base/Wood/Oak.mdl", root)
    assert Path(abs_oak) == oak.resolve()
    assert name == "Oak"
    abs_p, pname = resolve_local_mdl(
        "../../../Materials/vMaterials_2/Plastic/Plastic_Thick_Translucent.mdl",
        root,
    )
    assert Path(abs_p) == plastic.resolve()
    assert pname == "Plastic"


def test_rewrite_maple_usda_absolute_and_kinematic(tmp_path: Path | None = None):
    root = Path(tmp_path) if tmp_path is not None else Path("/tmp/rlinf_mdl_test2")
    plastic = root / "Base/Plastics/Plastic.mdl"
    oak = root / "Base/Wood/Oak.mdl"
    plastic.parent.mkdir(parents=True, exist_ok=True)
    oak.parent.mkdir(parents=True, exist_ok=True)
    plastic.write_text("mdl")
    oak.write_text("mdl")
    out = rewrite_maple_usda(_SAMPLE, root)
    assert "kinematicEnabled = 1" in out
    assert str(oak.resolve()) in out
    assert "vMaterials" not in out
    assert 'subIdentifier = "plastic_blue"' not in out
    assert 'subIdentifier = "Plastic"' in out
    assert 'subIdentifier = "Oak"' in out


def test_ensure_local_maple_usd_writes_sidecar(tmp_path: Path | None = None):
    root = Path(tmp_path) if tmp_path is not None else Path("/tmp/rlinf_mdl_test3")
    fixtures = root / "fixtures"
    materials = root / "Materials"
    fixtures.mkdir(parents=True)
    oak = materials / "Base/Wood/Oak.mdl"
    oak.parent.mkdir(parents=True)
    oak.write_text("mdl")
    maple = fixtures / "table_maple.usda"
    maple.write_text(_SAMPLE)
    wrapper = fixtures / "table_maple_arena.usda"
    wrapper.write_text('prepend payload = @./table_maple.usda@\n')
    out = ensure_local_maple_usd(maple, wrapper, materials)
    assert out.name == "table_maple_arena.localmdl.usda"
    assert "@./table_maple.localmdl.usda@" in out.read_text()
    sidecar = fixtures / "table_maple.localmdl.usda"
    assert sidecar.is_file()
    assert "vMaterials" not in sidecar.read_text()
    assert str(oak.resolve()) in sidecar.read_text()
