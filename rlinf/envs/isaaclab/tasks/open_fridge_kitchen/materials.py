# Copyright 2025 The RLinf Authors.
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

"""Fix Lightwheel fridge shaders so RTX / tiled cameras can see the appliance.

The kitchen USD already has ``T_Refrigerator032_*.png`` and OmniPBR bindings,
but fridge shaders omit ``info:id`` (walls set ``info:id = OmniPBR``). RTX then
drops the MDL surface and the 1.75 m body is invisible. This writes a small
sublayer that stamps ``info:id`` and a UsdPreviewSurface fallback.
"""

from __future__ import annotations

from pathlib import Path

_FRIDGE_PBR_SHADERS = (
    "material_Body001_material",
    "material_freezer0_material",
    "material_fridge_door_material",
    "material_fridge_drawer0_material",
)
_FRIDGE_GLASS_SHADERS = (
    "material_fridge_door_Clear_material",
    "material_Body001_Clear_material",
)
_FRIDGE_ALBEDO = "./textures/T_Refrigerator032_BC001_0.png"
_OVERLAY_NAME = "scene_fridge_visible.usda"


def kitchen_fridge_overlay_usda() -> str:
    """Return a USD sublayer that makes fridge OmniPBR/OmniGlass shaders valid."""
    blocks = [
        "#usda 1.0",
        "(",
        '    defaultPrim = "world"',
        '    upAxis = "Z"',
        "    metersPerUnit = 1",
        '    subLayers = [@./scene.usd@]',
        ")",
        "",
        'over "world"',
        "{",
        '    over "fridge_main_group"',
        "    {",
        '        over "Looks"',
        "        {",
    ]
    for name in _FRIDGE_PBR_SHADERS:
        blocks.extend(
            [
                f'            over "{name}"',
                "            {",
                '                token outputs:surface.connect = '
                f'</world/fridge_main_group/Looks/{name}/PreviewSurface.outputs:surface>',
                '                over "Shader"',
                "                {",
                '                    uniform token info:id = "OmniPBR"',
                "                    bool inputs:enable_ORM_texture = 0",
                "                }",
                '                def Shader "PreviewSurface"',
                "                {",
                '                    uniform token info:id = "UsdPreviewSurface"',
                "                    color3f inputs:diffuseColor.connect = "
                f'</world/fridge_main_group/Looks/{name}/PreviewTexture.outputs:rgb>',
                "                    float inputs:metallic = 0.65",
                "                    float inputs:roughness = 0.35",
                "                    float inputs:opacity = 1",
                "                    token outputs:surface",
                "                }",
                '                def Shader "PreviewTexture"',
                "                {",
                '                    uniform token info:id = "UsdUVTexture"',
                f"                    asset inputs:file = @{_FRIDGE_ALBEDO}@",
                "                    token outputs:rgb",
                "                }",
                "            }",
                "",
            ]
        )
    for name in _FRIDGE_GLASS_SHADERS:
        blocks.extend(
            [
                f'            over "{name}"',
                "            {",
                '                over "Shader"',
                "                {",
                '                    uniform token info:id = "OmniGlass"',
                "                }",
                "            }",
                "",
            ]
        )
    blocks.extend(
        [
            "        }",
            '        over "Refrigerator032"',
            "        {",
            '            over "Visuals"',
            "            {",
            '                over "Refrigerator032_Body001"',
            "                {",
            "                    bool doubleSided = 1",
            "                }",
            "            }",
            "        }",
            "    }",
            "}",
            "",
        ]
    )
    return "\n".join(blocks)


def prepare_kitchen_fridge_materials(kitchen_usd: str | Path) -> str:
    """Write the overlay next to ``scene.usd`` and return its path.

    If the kitchen file is missing, the original path is returned unchanged.
    """
    src = Path(kitchen_usd)
    if not src.is_file():
        return str(src)
    out = src.with_name(_OVERLAY_NAME)
    text = kitchen_fridge_overlay_usda()
    if not out.is_file() or out.read_text(encoding="utf-8") != text:
        out.write_text(text, encoding="utf-8")
    return str(out)


def patch_fridge_material_terminals(
    stage,
    kitchen_prim_path: str,
    albedo_path: str | None = None,
) -> int:
    """Rebind fridge body meshes to a UsdPreviewSurface RTX can rasterize.

    Lightwheel appliance OmniPBR uses an ORM map. In this tiled-camera path
    that MDL terminal fails and the mesh is skipped (stove/microwave match).
    Walls work because they use ``project_uvw`` OmniPBR without ORM. A new
    preview material is bound stronger than the authored OmniPBR.
    """
    from pxr import Sdf, UsdGeom, UsdShade

    root = f"{kitchen_prim_path.rstrip('/')}/fridge_main_group"
    looks_path = f"{root}/Looks/FridgeVisiblePreview"
    mat = UsdShade.Material.Define(stage, looks_path)
    shader = UsdShade.Shader.Define(stage, f"{looks_path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    diffuse = shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
    diffuse.Set((0.48, 0.48, 0.50))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.7)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.35)
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(1.0)
    albedo = Path(albedo_path) if albedo_path else None
    if albedo is not None and albedo.is_file():
        tex = UsdShade.Shader.Define(stage, f"{looks_path}/PreviewTexture")
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(albedo.resolve()))
        diffuse.ConnectToSource(tex.ConnectableAPI(), "rgb")
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

    n = 0
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not path.startswith(root) or prim.GetTypeName() != "Mesh":
            continue
        if "Clear" in path or "/Collisions/" in path or "/Sites/" in path:
            continue
        UsdGeom.Mesh(prim).CreateDoubleSidedAttr(True)
        UsdShade.MaterialBindingAPI(prim).Bind(mat)
        n += 1
    return n
