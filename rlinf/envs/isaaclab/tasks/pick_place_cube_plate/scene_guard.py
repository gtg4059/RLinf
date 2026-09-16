# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pin furniture and rewrite maple-table MDL refs before Isaac loads them.

Isaac's MDL compiler turns ``../../../Materials/Base/Wood/Oak.mdl`` into
``::..::..::..::Materials::Base::Wood::Oak`` and fails even when the file
exists. Nucleus ``vMaterials_2`` paths are not shipped. Both must be rewritten
to absolute local files *before* ``gym.make``, or Hydra emits thousands of
``SdrShaderNode`` errors during stage load.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

_FIXTURE_TOKENS = ("MapleTable", "franka_table", "Robot_Stand")
_BROKEN_MDL_MARKERS = (
    "vMaterials",
    "Plastic_Thick_Translucent",
    "Carpaint_Solid",
    "Carpaint/",
)

# Vendored by ``download_isaaclab_arena_assets.sh`` under Materials/.
_LOCAL_MDL_REL = {
    "Oak": "Base/Wood/Oak.mdl",
    "Walnut_Planks": "Base/Wood/Walnut_Planks.mdl",
    "Bamboo": "Base/Wood/Bamboo.mdl",
    "RustedMetal": "Base/Metals/RustedMetal.mdl",
    "Plastic": "Base/Plastics/Plastic.mdl",
    "Plastic_ABS": "Base/Plastics/Plastic_ABS.mdl",
    "Plastic_Clear": "Base/Plastics/Plastic_Clear.mdl",
}
_FALLBACK_STEMS = {
    "Plastic_Thick_Translucent": "Plastic",
    "Plastic_Thick_Translucent_Flakes": "Plastic",
    "Carpaint_Solid": "Plastic",
}
_FALLBACK_KEY = "Plastic"

_ASSET_PAIR_RE = re.compile(
    r"(uniform asset info:mdl:sourceAsset = @)([^@]+)(@)"
    r"(\s+uniform token info:mdl:sourceAsset:subIdentifier = \")([^\"]+)(\")",
)
_KINEMATIC_RE = re.compile(r"(bool physics:kinematicEnabled = )0\b")

# scene_guard.py → pick_place_cube_plate → … → repo (parents[5])
_REPO_ROOT = Path(__file__).resolve().parents[5]
_DEFAULT_ARENA = _REPO_ROOT / ".assets" / "isaaclab_arena"
_DEFAULT_MATERIALS = _DEFAULT_ARENA / "Materials"
_DEFAULT_MAPLE = (
    _DEFAULT_ARENA
    / "object_library"
    / "srl_robolab_assets"
    / "fixtures"
    / "table_maple.usda"
)
_DEFAULT_WRAPPER = _DEFAULT_MAPLE.with_name("table_maple_arena.usda")


def is_fixture_prim_path(path: str) -> bool:
    """True for maple desk / pedestal / stand prims (not the robot or cube)."""
    return any(token in path for token in _FIXTURE_TOKENS)


def is_broken_mdl_ref(ref: str) -> bool:
    """True when a shader still points at Nucleus vMaterials that we do not ship."""
    return any(marker in ref for marker in _BROKEN_MDL_MARKERS)


def is_unresolved_mdl_ref(ref: str) -> bool:
    """True when Isaac would compile this as a ``::..::`` module, not a file."""
    if is_broken_mdl_ref(ref):
        return True
    cleaned = ref.strip().strip("@")
    if ".." in cleaned:
        return True
    return not Path(cleaned).is_absolute()


def default_materials_root() -> Path:
    env_root = (os.environ.get("ARENA_ASSETS_ROOT") or "").strip()
    if env_root and not env_root.startswith(("http://", "https://")):
        candidate = Path(env_root) / "Materials"
        if candidate.is_dir():
            return candidate
    return _DEFAULT_MATERIALS


def _mdl_stem(ref: str) -> str:
    name = Path(ref.strip().strip("@").split("?")[0]).name
    return Path(name).stem


def resolve_local_mdl(
    ref: str, materials_root: Path
) -> tuple[str, str]:
    """Map a USD ``sourceAsset`` to ``(absolute_mdl, export_name)``."""
    root = Path(materials_root)
    stem = _mdl_stem(ref)
    key = _FALLBACK_STEMS.get(stem, stem)
    if key not in _LOCAL_MDL_REL:
        key = _FALLBACK_KEY
    rel = _LOCAL_MDL_REL[key]
    path = (root / rel).resolve()
    if path.is_file():
        return str(path), key
    fallback = (root / _LOCAL_MDL_REL[_FALLBACK_KEY]).resolve()
    if fallback.is_file():
        return str(fallback), _FALLBACK_KEY
    return str(path), key


def rewrite_maple_usda(text: str, materials_root: Path) -> str:
    """Rewrite MDL refs to absolute local files and pin table kinematics."""

    def _replace(match: re.Match[str]) -> str:
        abs_path, export_name = resolve_local_mdl(match.group(2), materials_root)
        return (
            f"{match.group(1)}{abs_path}{match.group(3)}"
            f"{match.group(4)}{export_name}{match.group(6)}"
        )

    rewritten = _ASSET_PAIR_RE.sub(_replace, text)
    return _KINEMATIC_RE.sub(r"\g<1>1", rewritten, count=1)


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.read_text() == text:
        return
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(text)
    tmp.replace(path)


def ensure_local_maple_usd(
    maple_usda: Path | None = None,
    wrapper_usda: Path | None = None,
    materials_root: Path | None = None,
) -> Path:
    """Write sidecars with absolute MDLs. Return the wrapper Isaac should spawn."""
    maple = Path(maple_usda or _DEFAULT_MAPLE)
    wrapper = Path(wrapper_usda or _DEFAULT_WRAPPER)
    materials = Path(materials_root or default_materials_root())
    if not maple.is_file():
        return wrapper if wrapper.is_file() else maple

    orig = Path(str(maple) + ".orig")
    src = orig if orig.is_file() else maple
    sidecar = maple.with_name(f"{maple.stem}.localmdl.usda")
    _atomic_write(sidecar, rewrite_maple_usda(src.read_text(), materials))

    if not wrapper.is_file():
        return sidecar
    wrap_sidecar = wrapper.with_name(f"{wrapper.stem}.localmdl.usda")
    wrap_text = wrapper.read_text().replace(
        "@./table_maple.usda@", f"@./{sidecar.name}@"
    )
    _atomic_write(wrap_sidecar, wrap_text)
    return wrap_sidecar


def _disable_low_res_dlss(settings) -> None:
    """DLSS min input is 300px; DROID tiled cams are 224 and trip a warning."""
    for key, value in (
        ("/rtx/post/dlss/enabled", False),
        ("/rtx/post/aa/algo", 0),
    ):
        try:
            settings.set(key, value)
        except Exception:
            continue


def _add_mdl_search_path(settings, materials_root: Path) -> None:
    path = str(materials_root)
    for key in (
        "/rtx/neuraylib/userMdlSearchPaths",
        "/app/mdl/additionalSearchPaths",
    ):
        try:
            current = settings.get(key)
            if current is None:
                settings.set(key, path)
                continue
            if isinstance(current, str):
                if path not in current:
                    settings.set(key, f"{current};{path}" if current else path)
            elif isinstance(current, (list, tuple)) and path not in current:
                settings.set(key, list(current) + [path])
        except Exception:
            continue


def prepare_isaac_render(materials_root: Path | None = None) -> Path:
    """After AppLauncher: MDL search path, no DLSS, local maple USD sidecar."""
    root = Path(materials_root or default_materials_root())
    try:
        import carb

        settings = carb.settings.get_settings()
        _add_mdl_search_path(settings, root)
        _disable_low_res_dlss(settings)
    except Exception:
        pass
    return ensure_local_maple_usd(materials_root=root)


def pin_static_scene_prims(env) -> int:
    """Set furniture rigid bodies kinematic so contact cannot launch them.

    Arena maple USD parts can carry dynamic RigidBodyAPI. Isaac
    ``reset_scene_to_default`` does not restore AssetBase poses, so one impulse
    leaves the table at 1e12 forever. Kinematic bodies keep collision.
    """
    from pxr import UsdPhysics

    stage = env.sim.stage
    pinned = 0
    for prim in stage.Traverse():
        if not is_fixture_prim_path(str(prim.GetPath())):
            continue
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        api = UsdPhysics.RigidBodyAPI(prim)
        attr = api.GetKinematicEnabledAttr()
        if not attr:
            attr = api.CreateKinematicEnabledAttr()
        attr.Set(True)
        pinned += 1
    return pinned


def rebind_broken_maple_shaders(env, materials_root: Path | None = None) -> int:
    """Point leftover maple shaders at absolute local MDLs (post-load safety)."""
    from pxr import Sdf

    root = Path(materials_root or default_materials_root())
    stage = env.sim.stage
    rewritten = 0
    for prim in stage.Traverse():
        if prim.GetTypeName() != "Shader":
            continue
        path = str(prim.GetPath())
        if "MapleTable" not in path and "Looks" not in path:
            continue
        attr = prim.GetAttribute("info:mdl:sourceAsset")
        if attr is None or not attr.HasValue():
            continue
        value = str(attr.Get())
        if not is_unresolved_mdl_ref(value) and Path(
            value.strip().strip("@")
        ).is_file():
            continue
        abs_path, export_name = resolve_local_mdl(value, root)
        if not Path(abs_path).is_file():
            continue
        attr.Set(Sdf.AssetPath(abs_path))
        sub = prim.GetAttribute("info:mdl:sourceAsset:subIdentifier")
        if not sub:
            sub = prim.CreateAttribute(
                "info:mdl:sourceAsset:subIdentifier", Sdf.ValueTypeNames.Token
            )
        sub.Set(export_name)
        rewritten += 1
    return rewritten


def harden_isaac_scene(env) -> None:
    """Apply furniture pin + leftover MDL rewrite after ``gym.make``."""
    pin_static_scene_prims(env)
    rebind_broken_maple_shaders(env)
