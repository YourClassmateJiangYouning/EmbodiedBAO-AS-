"""Apply one of the Stage 3 scenes to a live USD stage.

The catalogue in ``scenes.py`` says what a scene is; this module puts it on the stage.  It is
written defensively because it cannot be exercised on the development machine -- ``pxr`` only
exists inside Isaac Sim -- so every optional step records what actually happened instead of
assuming it worked.  ``apply_scene`` returns a report, and the runner writes it beside
args.json, which is what makes "this run used the real Concrete_Panels texture" a fact about
the data rather than a claim about the machine.

Three rules it must not break, all enforced in ``test_bao_scenes.py``:

* the marker is the task, so its size, position and centre height come from the catalogue and
  are identical for all 25;
* dressing never collides, so nothing here applies a collider or a rigid body to it;
* nothing here touches the obstacle wall, the opening, the room's bounds or the success
  plane.  Those belong to ``environment.py`` and the scene must leave them exactly as they
  were -- which is what makes the scene a skin rather than a variable.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import scenes as sc


# ---------------------------------------------------------------------------
# Materials
# ---------------------------------------------------------------------------


def _flat_material(stage: Any, path: str, rgb: Sequence[float]) -> Any:
    """A constant-colour UsdPreviewSurface.  Always available, no download, exact colour."""
    from pxr import Gf, UsdShade

    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, os.path.join(path, "Shader"))
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", "color").Set(Gf.Vec3f(*[float(v) for v in rgb]))
    shader.CreateInput("roughness", "float").Set(0.55)
    shader.CreateInput("metallic", "float").Set(0.0)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def resolve_material(stage: Any, url: str, name: str, path: str,
                     fallback_rgb: Sequence[float]) -> Tuple[Any, str]:
    """Create the material an MDL URL names, or a flat colour if that is not possible.

    Returns (material, how) where ``how`` is ``"mdl"`` or ``"fallback"``.  The fallback is
    not a failure to hide: the bucket may be unreachable from the lab, and a scene that
    falls back to flat colours is still a scene.  What matters is that the report says so.
    """
    try:
        import omni.kit.commands  # type: ignore

        omni.kit.commands.execute(
            "CreateMdlMaterialPrimCommand", mtl_url=url, mtl_name=name, mtl_path=path)
        from pxr import UsdShade

        material = UsdShade.Material(stage.GetPrimAtPath(path))
        if material:
            return material, "mdl"
    except Exception:  # noqa: BLE001
        pass
    return _flat_material(stage, path, fallback_rgb), "fallback"


# ---------------------------------------------------------------------------
# The marker
# ---------------------------------------------------------------------------


def _prism(stage: Any, path: str, polygon: Sequence[Sequence[float]],
           centre: Sequence[float], thickness: float) -> Any:
    """One polygon extruded along x, standing in the y/z plane of the far wall."""
    from pxr import Gf, UsdGeom, Vt

    cx, cy, cz = (float(v) for v in centre)
    half = thickness / 2.0
    front = [Gf.Vec3f(cx - half, cy + float(py), cz + float(px)) for px, py in polygon]
    back = [Gf.Vec3f(cx + half, cy + float(py), cz + float(px)) for px, py in polygon]
    points = front + back
    count = len(polygon)
    counts = [count, count] + [4] * count
    indices: List[int] = list(range(count)) + [count + i for i in range(count)]
    for i in range(count):
        j = (i + 1) % count
        indices += [i, j, count + j, count + i]
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray(points))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray(counts))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(indices))
    mesh.CreateSubdivisionSchemeAttr("none")
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    zs = [p[2] for p in points]
    mesh.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(min(xs), min(ys), min(zs)),
                                         Gf.Vec3f(max(xs), max(ys), max(zs))]))
    return mesh.GetPrim()


def build_marker(stage: Any, scene: str, slot: int) -> Dict[str, Any]:
    """Build this scene's marker for one repeat, and bind its colour.

    The existing environment builds a red square; that is marker 1 of the baseline scene, so
    ``build_marker(stage, "stage1.1", 1)`` reproduces it.  Slot numbering is 1-based to match
    the round numbering in the logs.
    """
    entries = sc.MARKERS[scene]
    if not 1 <= slot <= len(entries):
        raise ValueError(f"{scene} has {len(entries)} markers, asked for slot {slot}")
    shape, colour_key = entries[slot - 1]
    rgb = sc.COLOURS[colour_key]
    centre = (sc.MARKER_X_M, sc.MARKER_Y_M, sc.MARKER_Z_M)
    paths = []
    for index, polygon in enumerate(sc.SHAPES[shape]):
        paths.append(_prism(stage, f"/World/GoalMarker/part{index}", polygon, centre,
                            sc.MARKER_THICKNESS_M))
    material_path = f"/World/Looks/Marker_{scene}_{slot}"
    material = _flat_material(stage, material_path, rgb)
    for path in paths:
        _bind(stage, path, material)
    return {"scene": scene, "slot": slot, "shape": shape, "colour": colour_key,
            "rgb": tuple(rgb), "parts": len(paths)}


def _bind(stage: Any, prim_path: str, material: Any) -> bool:
    try:
        from pxr import UsdShade

        prim = stage.GetPrimAtPath(prim_path)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)
        return True
    except Exception:  # noqa: BLE001
        return False


# ---------------------------------------------------------------------------
# Dressing
# ---------------------------------------------------------------------------


def _disable_collision(prim: Any) -> None:
    """Say so explicitly, rather than relying on not having added a collider.

    A missing collider and an explicitly disabled one look the same to PhysX but not to
    someone reading the stage, and this property is the one the whole design rests on.
    """
    try:
        from pxr import Sdf

        prim.CreateAttribute("physics:collisionEnabled",
                             Sdf.ValueTypeNames.Bool, False).Set(False)
    except Exception:  # noqa: BLE001
        pass


def place_dressing(stage: Any, scene: str) -> List[Dict[str, Any]]:
    """Place the scene's dressing: reference the named asset, or stand a box in for it."""
    from pxr import Gf, UsdGeom

    placed: List[Dict[str, Any]] = []
    for item in sc.SCENES[scene]["dressing"]:
        path = f"/World/Dressing/{item['name']}"
        used_asset = False
        if item.get("asset"):
            try:
                prim = UsdGeom.Xform.Define(stage, path).GetPrim()
                prim.GetReferences().AddReference(item["asset"])
                used_asset = True
            except Exception:  # noqa: BLE001
                used_asset = False
        if not used_asset:
            cube = UsdGeom.Cube.Define(stage, path)
            cube.GetSizeAttr().Set(1.0)
            cube.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in item["size"]]))
            prim = cube.GetPrim()
        UsdGeom.Xformable(prim).AddTranslateOp().Set(Gf.Vec3f(*[float(v) for v in item["at"]]))
        _disable_collision(prim)
        placed.append({"name": item["name"], "mount": item["mount"],
                       "asset": item["asset"], "used_asset": used_asset})
    return placed


# ---------------------------------------------------------------------------
# The whole scene
# ---------------------------------------------------------------------------


def apply_scene(stage: Any, scene: str, slot: int,
                surfaces: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Put a scene on the stage and report what resolved.

    ``surfaces`` maps floor / side_wall / ceiling / far_wall to the prim paths of OUR room,
    so this module never has to know how environment.py builds or names things.  A baseline
    scene has no materials and no dressing, so it is a no-op that still reports.
    """
    if scene not in sc.SCENES:
        raise ValueError(f"unknown scene {scene!r}, expected one of {list(sc.SCENES)}")
    spec = sc.SCENES[scene]
    report: Dict[str, Any] = {"scene": scene, "slot": slot, "label": spec["label"],
                              "materials": {}, "marker": None, "dressing": []}

    fallbacks = {"floor": (0.55, 0.55, 0.55), "side_wall": (0.72, 0.73, 0.75),
                 "ceiling": (0.92, 0.93, 0.95), "far_wall": (0.13, 0.42, 0.20)}
    for surface, url in spec["materials"].items():
        prim_path = (surfaces or {}).get(surface)
        if not prim_path:
            report["materials"][surface] = {"url": url, "how": "no-surface-given"}
            continue
        name = os.path.basename(url).replace(".mdl", "")
        material, how = resolve_material(
            stage, url, name, f"/World/Looks/{surface}_{scene}", fallbacks.get(surface, (0.6,) * 3))
        _bind(stage, prim_path, material)
        report["materials"][surface] = {"url": url, "how": how, "prim": prim_path}

    report["marker"] = build_marker(stage, scene, slot)
    report["dressing"] = place_dressing(stage, scene)
    report["dressing_count"] = len(report["dressing"])
    return report


def format_report(report: Dict[str, Any]) -> str:
    """One line per fact, for the run log."""
    lines = [f"[scene] {report['scene']} ({report['label']}) marker slot {report['slot']}"]
    marker = report["marker"]
    lines.append(f"[scene] marker: {marker['shape']} in {marker['colour']} "
                 f"({marker['parts']} mesh part(s))")
    for surface, info in report["materials"].items():
        lines.append(f"[scene] {surface:<9} {info['how']:<11} {info.get('url', '')}")
    if report["dressing"]:
        used = sum(1 for item in report["dressing"] if item["used_asset"])
        lines.append(f"[scene] dressing: {report['dressing_count']} items, "
                     f"{used} from assets, rest boxes, all non-collidable")
    return "\n".join(lines)
