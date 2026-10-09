"""Apply one of the Stage 1 scenes to a live USD stage.

These are the Stage 1 (A/S threshold) scene skins, specified in ``STAGE1_SCENES.md``; see
``scenes.py`` for why the stage number matters here -- the document they come from used to
be named ``STAGE3_SCENES.md``, which is what put the wrong label on this module too.

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


def prim_name(text: Any) -> str:
    """A token safe to use inside a USD prim path.

    A scene is called ``stage1.2`` and a dot is a property separator in SdfPath, so
    ``/World/Looks/Marker_stage1.1_1`` is ill-formed and USD refuses it -- measured on the
    lab machine, which is the only reason this function exists.  The failure was quiet in the
    worst way: the material was simply never created, so every scene would have run with its
    surfaces unbound and nothing in the results would have said so.
    """
    return "".join(
        character if character.isalnum() or character == "_" else "_"
        for character in str(text)
    )


def paint(stage: Any, prim_path: str, rgb: Sequence[float]) -> bool:
    """Paint a prim with USD's own displayColor.  No shader graph, no material, no download.

    Two attempts at building a UsdPreviewSurface by hand both failed on the lab machine: the
    first passed a plain string where the bindings wanted a type token, the second connected
    an output that had never been created and got "Used null prim".  displayColor is the
    primitive that needs neither, it is respected by RTX for unbound geometry, and it holds
    the exact catalogue colour -- which matters because the marker's colour is the one thing
    this experiment varies and the uniqueness check compares colours numerically.

    Returns whether it worked, so callers can report it rather than assume it.
    """
    try:
        from pxr import Gf, UsdGeom, Vt

        prim = stage.GetPrimAtPath(prim_path)
        gprim = UsdGeom.Gprim(prim)
        if not gprim:
            return False
        gprim.CreateDisplayColorAttr().Set(Vt.Vec3fArray(
            [Gf.Vec3f(float(rgb[0]), float(rgb[1]), float(rgb[2]))]))
        return True
    except Exception:  # noqa: BLE001
        return False


LOCAL_ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "isaac")


def local_asset_for(url: str) -> Optional[str]:
    """The vendored copy of a remote material, if one is in the repository.

    The scenes name their materials by their NVIDIA URL, and on the lab machine that URL is
    what failed: the material library's cache is not writable there, so a material created from
    a URL arrived without its shaders and every surface rendered as noise.  Rather than rewrite
    every catalogue entry, the vendored set -- committed under assets/isaac by
    tools/shrink_textures.py -- is consulted first, and the network is only used for materials
    nobody vendored.  A clone therefore renders the same scene the author saw, without the
    network and without the cache.
    """
    if not url or url.startswith("assets") or os.path.isabs(url):
        return url if os.path.exists(url) else None
    name = os.path.basename(url)
    for folder, _, files in os.walk(LOCAL_ASSETS):
        if name in files:
            return os.path.join(folder, name)
    return None


def resolve_material(stage: Any, url: str, name: str, path: str,
                     surface_prim: str, fallback_rgb: Sequence[float]) -> Tuple[Any, str]:
    """Create the material an MDL URL names, or a flat colour if that is not possible.

    Returns (material, how) where ``how`` is ``"mdl"`` or ``"fallback"``.  The fallback is
    not a failure to hide: the bucket may be unreachable from the lab, and a scene that
    falls back to flat colours is still a scene.  What matters is that the report says so.
    """
    try:
        import omni.kit.commands  # type: ignore

        source = local_asset_for(url) or url
        omni.kit.commands.execute(
            "CreateMdlMaterialPrimCommand", mtl_url=source, mtl_name=name, mtl_path=path)
        from pxr import UsdShade

        material = UsdShade.Material(stage.GetPrimAtPath(path))
        if material:
            return material, ("mdl-local" if source != url else "mdl")
    except Exception:  # noqa: BLE001
        pass
    paint(stage, surface_prim, fallback_rgb)
    return None, "fallback-paint"


# ---------------------------------------------------------------------------
# The marker
# ---------------------------------------------------------------------------


def _prism(stage: Any, path: str, polygon: Sequence[Sequence[float]],
           centre: Sequence[float], thickness: float) -> Any:
    """One polygon extruded along x, standing in the plane of the far wall.

    The polygon is given in USER coordinates -- (px, py) is (sideways, height) -- and every point
    is converted to the stage's WORLD frame before it becomes geometry.  Without that conversion
    the plate lands at height zero and 1.40 m sideways instead of 1.40 m up the wall, which is
    why no marker could be seen on the far wall in any render.
    """
    from pxr import Gf, UsdGeom, Vt

    cx, cy, cz = sc.to_world(centre)
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
    # Two-sided, because the marker is a flat plate seen from one direction and the winding of a
    # face decides whether the renderer culls it.  The polygons come from SHAPES with whatever
    # winding reads naturally in (px, py), and those coordinates map to (z, y): with the camera on
    # the -x side, a polygon that is counter-clockwise as drawn is clockwise as seen, so the front
    # face can end up back-facing and be invisible.  That is one explanation for the marker
    # missing from every preview.  doubleSided removes the question entirely; a marker that is a
    # plate has no wrong side.
    mesh.CreateDoubleSidedAttr(True)
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
        prim = _prism(stage, f"/World/GoalMarker/part{index}", polygon, centre,
                      sc.MARKER_THICKNESS_M)
        paint(stage, str(prim.GetPath()), rgb)
        paths.append(prim)
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


def prop_asset_for(url: Optional[str]) -> Optional[str]:
    """The vendored .usd mesh for a catalogue ``asset``, or None if there is not one.

    Returns a local path, never a network one: a prop fetched from a URL arrives without its
    material layers on the workstation, so a reference built from one renders as an untextured
    shell -- the same failure that made the material library local-first.

    Three cases are refused on purpose, and each of them was a real bug on the way here:

      * an ``.mdl`` is a MATERIAL, not a stage asset.  USD answers "Cannot determine file
        format", which is why the original place_dressing wrote boxes and the catalogue kept
        MI_SignB and M_TrafficCone as notes rather than references.
      * anything not under assets/isaac has not been vendored, so referencing it would work on
        this machine and not on a clone.
      * a bare wrapper layer.  Measured: small_KLT.usd is 6.6 kB and sektion_cabinet_
        instanceable.usd is 5.8 kB, while small_KLT_visual.usd is 180 kB and
        sektion_cabinet_visuals.usd is 167 kB -- the geometry lives in the ``_visual`` layer
        and the small file is a proxy that renders as nothing useful.  If a vendored
        ``<stem>_visual...usd`` exists beside the named file, that is the one to reference.
    """
    if not url or not url.endswith(".usd"):
        return None
    name = os.path.basename(url)
    # A name that is already a visual layer is what the caller meant -- use it as written.
    # (Before this check, asking for small_KLT_visual.usd chose small_KLT_visual_collision.usd,
    # because the collision layer is longer and also matches "stem + visual".)
    if "visual" in name:
        for root, _dirs, files in os.walk(LOCAL_ASSETS):
            if name in files:
                return os.path.join(root, name)
        return None
    stem = name[: -len(".usd")]
    for root, _dirs, files in os.walk(LOCAL_ASSETS):
        if name not in files:
            continue
        # The named file may be a wrapper whose geometry is in a sibling visual layer:
        # small_KLT.usd is 6.6 kB against small_KLT_visual.usd's 180 kB, and a proxy layer
        # references without error and renders as nothing useful.
        for candidate in sorted(files):
            if candidate.endswith(".usd") and candidate != name and stem in candidate \
                    and "visual" in candidate and "collision" not in candidate:
                return os.path.join(root, candidate)
        return os.path.join(root, name)
    return None


def place_dressing(stage: Any, scene: str) -> List[Dict[str, Any]]:
    """Place the scene's dressing: a box, or the vendored mesh it names.

    Every item gets a box of its declared size, which is the geometry the occlusion and
    walking-band checks were written against; where the catalogue names a vendored ``.usd``
    prop, that prop is referenced on top of it so the item looks like a thing rather than a
    cube.  The box is what carries the placement and the scale, so a missing prop, a broken
    layer or a failed reference degrades to exactly the scene that shipped before this
    function could reference anything -- and the report says which happened.

    Only floor items take a reference.  A wall item's box is a picture plate and the props in
    the library are furniture, so referencing one there would put a cabinet through a wall for
    no gain.

    The first version referenced whatever ``asset`` named, which for MI_SignB and
    M_TrafficCone was an MDL: USD answered "Cannot determine file format" and the item stayed
    a box.  prop_asset_for() now refuses that case explicitly instead of relying on USD to.
    """
    from pxr import Gf, UsdGeom

    placed: List[Dict[str, Any]] = []
    for item in sc.SCENES[scene]["dressing"]:
        path = f"/World/Dressing/{item['name']}"
        cube = UsdGeom.Cube.Define(stage, path)
        cube.GetSizeAttr().Set(1.0)
        # Position and dimensions are converted from the catalogue's user frame (height second)
        # to the stage's world frame (height last).  Skipping this put every item at the wrong
        # height and the wrong sideways offset, several of them outside the 5 m wide room and one
        # or two across the agent's view.
        cube.AddScaleOp().Set(Gf.Vec3f(*sc.to_world_size(item["size"])))
        prim = cube.GetPrim()
        UsdGeom.Xformable(prim).AddTranslateOp().Set(Gf.Vec3f(*sc.to_world(item["at"])))
        _disable_collision(prim)
        if item.get("colour"):
            paint(stage, path, item["colour"])

        record = {"name": item["name"], "mount": item["mount"],
                  "asset": item["asset"], "used_asset": False}
        local = prop_asset_for(item["asset"]) if item["mount"] == "floor" else None
        if local:
            try:
                # Reference the prop onto an Xform child of the box:
                #     UsdGeom.Xform.Define(stage, path) -> schema
                #     schema.GetPrim()                  -> prim
                #     prim.GetReferences()              -> UsdReferences metadata
                #     .AddReference(local_path)         -> the reference itself
                # Written in that explicit shape because it cannot be exercised on the
                # development machine (no pxr).  USD's own idiom for this exists as a
                # convenience call that performs the same four steps; spelling them out keeps
                # the prim that receives the reference unambiguous, and the except below turns
                # any mistake here into a recorded message plus the box that was already
                # built, rather than a stage that fails to assemble mid-sweep.
                proxy_schema = UsdGeom.Xform.Define(stage, f"{path}/prop")
                proxy_prim = proxy_schema.GetPrim()
                proxy_prim.GetReferences().AddReference(local)
                # The proxy is NOT scaled.  The declared size is the box the item must occupy,
                # which is what the fixture checks are written against; a prop's own authored
                # size is a fact about the asset, and forcing one onto the other without a
                # measurement would change what those checks mean.  Calibrating scale from each
                # asset's extent is a separate step and it needs a renderer.
                _disable_collision(proxy_prim)
                record["used_asset"] = os.path.basename(local)
                record["reference"] = local
            except Exception as exc:  # noqa: BLE001
                # A prop that will not load must leave the box, not the scene, broken.
                record["reference_error"] = f"{type(exc).__name__}: {exc}"
        placed.append(record)
    return placed


# ---------------------------------------------------------------------------
# Finding our own room's surfaces
# ---------------------------------------------------------------------------


def classify_surfaces(boxes: Sequence[Tuple[str, Sequence[float], Sequence[float]]]
                      ) -> Dict[str, Any]:
    """Pick the floor, ceiling, far wall and side walls out of a list of named boxes.

    Pure, so it can be checked without a stage.  Boxes must arrive in the catalogue's frame --
    (x, height, lateral) -- which is what discover_surfaces converts to before calling this.
    Classification is by where a box sits and how big its faces are, not by its name:
    environment.py is a frozen file and this module must not depend on how it names things.

    Two defects lived here.  The floor rule asked for ``centre_y <= 0.05``, which any box below
    the floor line satisfies -- a side wall at y = -2.51 included, and that is exactly what it
    picked.  And only the largest side wall was returned, so the other one was never given a
    material and the two sides of the corridor could differ.  ``side_walls`` now carries all of
    them and the caller binds every one.
    """
    best: Dict[str, Tuple[float, str]] = {}
    side_walls: List[Tuple[float, str]] = []

    def consider(kind: str, score: float, name: str) -> None:
        if score > best.get(kind, (0.0, ""))[0]:
            best[kind] = (score, name)

    for name, low, high in boxes:
        size_x = abs(float(high[0]) - float(low[0]))
        size_y = abs(float(high[1]) - float(low[1]))
        size_z = abs(float(high[2]) - float(low[2]))
        centre_y = (float(low[1]) + float(high[1])) / 2.0
        centre_z = (float(low[2]) + float(high[2])) / 2.0
        minimum_x = min(float(low[0]), float(high[0]))
        if size_y <= 0.05 and abs(centre_y) <= 0.05:
            consider("floor", size_x * size_z, name)
        if size_y <= 0.05 and centre_y >= 2.5:
            consider("ceiling", size_x * size_z, name)
        if size_x <= 0.05 and minimum_x >= 15.5:
            consider("far_wall", size_y * size_z, name)
        if size_z <= 0.05 and abs(centre_z) >= 2.0:
            side_walls.append((size_x * size_y, name))
    found: Dict[str, Any] = {kind: name for kind, (score, name) in best.items() if score > 0.0}
    if side_walls:
        side_walls.sort(reverse=True)
        found["side_wall"] = side_walls[0][1]
        found["side_walls"] = tuple(name for _, name in side_walls)
    return found


def discover_surfaces(stage: Any) -> Dict[str, str]:
    """Read every prim's world bounds and classify them.  Never raises."""
    try:
        from pxr import Usd, UsdGeom
    except Exception:  # noqa: BLE001
        return {}
    boxes: List[Tuple[str, Sequence[float], Sequence[float]]] = []
    try:
        for prim in stage.Traverse():
            if not prim.IsA(UsdGeom.Boundable):
                continue
            boundable = UsdGeom.Boundable(prim)
            extent = boundable.GetExtentAttr().Get()
            if not extent or len(extent) != 2:
                continue
            xform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            low = xform.Transform(tuple(extent[0]))
            high = xform.Transform(tuple(extent[1]))
            # Back to the catalogue's frame before classifying.  The stage is Z-up and the
            # catalogue is height-second; feeding world boxes to user-coordinate rules is what
            # made the floor rule match a side wall (its y is negative, and the rule asked only
            # for y <= 0.05) and left the ceiling unclassified.
            boxes.append((str(prim.GetPath()),
                          sc.to_user((min(low[0], high[0]), min(low[1], high[1]),
                                      min(low[2], high[2]))),
                          sc.to_user((max(low[0], high[0]), max(low[1], high[1]),
                                      max(low[2], high[2])))))
    except Exception:  # noqa: BLE001
        pass
    return classify_surfaces(boxes)


# ---------------------------------------------------------------------------
# The whole scene
# ---------------------------------------------------------------------------


def describe_stage(stage: Any, limit: int = 8) -> List[str]:
    """The largest boundable boxes on the stage, for when a surface cannot be classified.

    Printed instead of guessing why.  The lab found floor, far wall and side wall but not the
    ceiling; without this the next step would have been another guess, and with it the actual
    extent of whatever the ceiling is comes back in the log.
    """
    try:
        from pxr import Usd, UsdGeom
    except Exception:  # noqa: BLE001
        return []
    found = []
    try:
        for prim in stage.Traverse():
            if not prim.IsA(UsdGeom.Boundable):
                continue
            extent = UsdGeom.Boundable(prim).GetExtentAttr().Get()
            if not extent or len(extent) != 2:
                continue
            xform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            low = xform.Transform(tuple(extent[0]))
            high = xform.Transform(tuple(extent[1]))
            size = [abs(high[i] - low[i]) for i in range(3)]
            found.append((max(size), str(prim.GetPath()),
                          tuple(round(min(low[i], high[i]), 2) for i in range(3)),
                          tuple(round(max(low[i], high[i]), 2) for i in range(3))))
    except Exception:  # noqa: BLE001
        return []
    found.sort(reverse=True)
    return [f"{name}  min={low} max={high}" for _, name, low, high in found[:limit]]


def apply_scene(stage: Any, scene: str, slot: int,
                surfaces: Optional[Dict[str, str]] = None,
                use_mdl: bool = True,
                parts: Sequence[str] = ("materials", "marker", "dressing")) -> Dict[str, Any]:
    """Put a scene on the stage and report what resolved.

    ``surfaces`` maps floor / side_wall / ceiling / far_wall to the prim paths of OUR room,
    so this module never has to know how environment.py builds or names things.  A baseline
    scene has no materials and no dressing, so it is a no-op that still reports.

    ``use_mdl`` is False by default, and that default was bought with a rendered frame.  On the
    lab machine the material library's cache is not writable, so MDL prims are created without
    their shaders and every surface that received one rendered as salt-and-pepper noise -- the
    whole scene came back looking broken.  Flat paint is ugly but correct, and a flat-painted
    room reads as a room; noise does not.  Opt in when the cache is writable.
    """
    if scene not in sc.SCENES:
        raise ValueError(f"unknown scene {scene!r}, expected one of {list(sc.SCENES)}")
    spec = sc.SCENES[scene]
    report: Dict[str, Any] = {"scene": scene, "slot": slot, "label": spec["label"],
                              "materials": {}, "marker": None, "dressing": []}

    fallbacks = {"floor": (0.55, 0.55, 0.55), "side_wall": (0.72, 0.73, 0.75),
                 "ceiling": (0.92, 0.93, 0.95), "far_wall": (0.13, 0.42, 0.20)}
    for surface, url in spec["materials"].items():
        if "materials" not in parts:
            report["materials"][surface] = {"url": url, "how": "skipped"}
            continue
        # A surface can be more than one prim.  There are two side walls and they have to look
        # alike: binding only the larger one left the other in its original colour, so the
        # corridor came out with one wall painted and one not, which reads as a lighting fault
        # rather than as a missing material.
        prim_paths: List[str] = []
        single = (surfaces or {}).get(surface)
        if single:
            prim_paths.append(str(single))
        for extra in (surfaces or {}).get(f"{surface}s", ()) or ():
            if str(extra) not in prim_paths:
                prim_paths.append(str(extra))
        if not prim_paths:
            report["materials"][surface] = {"url": url, "how": "no-surface-given"}
            continue
        prim_path = prim_paths[0]
        if not use_mdl:
            for path in prim_paths:
                paint(stage, path, fallbacks.get(surface, (0.6,) * 3))
            report["materials"][surface] = {"url": url, "how": "flat-paint",
                                            "prim": ", ".join(prim_paths)}
            continue
        name = os.path.basename(url).replace(".mdl", "")
        material, how = resolve_material(
            stage, url, name, f"/World/Looks/{surface}_{prim_name(scene)}",
            prim_path, fallbacks.get(surface, (0.6,) * 3))
        if material is not None:
            for path in prim_paths:
                _bind(stage, path, material)
        report["materials"][surface] = {"url": url, "how": how,
                                        "prim": ", ".join(prim_paths)}

    report["marker"] = build_marker(stage, scene, slot)
    report["parts"] = list(parts)
    report["dressing"] = place_dressing(stage, scene) if "dressing" in parts else []
    report["dressing_count"] = len(report["dressing"])
    return report


def format_report(report: Dict[str, Any]) -> str:
    """One line per fact, for the run log."""
    lines = [f"[scene] {report['scene']} ({report['label']}) marker slot {report['slot']}"]
    marker = report["marker"]
    lines.append(f"[scene] marker: {marker['shape']} in {marker['colour']} "
                 f"({marker['parts']} mesh part(s))")
    for surface, info in report["materials"].items():
        # The prim path is printed because without it the log cannot answer the question that
        # matters when a surface looks wrong: which prim did this material actually land on.  In
        # a preview of the library the right-hand wall came out in the floor's wood, and there
        # was no way to tell from the log whether the classifier had named the wrong prim, named
        # the same prim twice, or the wall prim was simply not the one being painted.
        lines.append(f"[scene] {surface:<9} {info['how']:<11} -> {info.get('prim', '-')}")
    if report["dressing"]:
        used = sum(1 for item in report["dressing"] if item["used_asset"])
        lines.append(f"[scene] dressing: {report['dressing_count']} items, "
                     f"{used} from assets, rest boxes, all non-collidable")
    return "\n".join(lines)
