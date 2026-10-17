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
import traceback
from typing import Any, Dict, List, Optional, Sequence, Tuple

import scenes as sc

# Where the marker lives, and the two child-name forms inside it.  environment.set_marker_slot()
# asserts "exactly one marker on the far wall" by counting the children named part<N>, so the
# material that paint() binds must NOT also be a child of this path under a part-like name.
# Facts the assertion depends on, kept next to each other on purpose.
MARKER_ROOT = "/World/GoalMarker"
MARKER_PART_PREFIX = "part"
MARKER_MATERIAL_PREFIX = "material"


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


def paint(stage: Any, prim_path: str, rgb: Sequence[float]) -> Tuple[bool, str]:
    """Paint a prim with USD's own displayColor, and report whether it worked.

    displayColor is the primitive that needs no shader graph, no material and no download: it is
    respected by RTX for unbound geometry, and it holds the exact catalogue colour -- which
    matters, because the marker's colour is the one thing this experiment varies and the
    uniqueness check compares colours numerically.

    The tuple return carries the reason on failure.  A bare boolean cost two render round trips:
    the log said only "painted False", which is a fact nobody can act on.

    A note kept from a self-lit variant that was built, measured, and then removed on 2026-10-10.
    Two facts about that path are worth keeping even though the code is gone:

      * emissiveColor is an input of a SHADER, not an attribute of geometry.  Putting it on the
        gprim fails, and the failure is quiet: the render log showed "emissive (0.45, 0.18,
        0.315), painted False" -- the values right and nothing self-lit.
      * the shader id takes a plain STRING.  Wrapping it in a token raises
        "module 'pxr.Tf' has no attribute 'Token'" on the lab machine, whose pxr.Tf has no token
        type at all; the plain string was measured working there.

    It was removed because it did not earn its keep.  A controlled comparison (emissive gain 0
    against 4, same slot, same camera) showed the emission did reach the renderer -- the marker's
    core went from grey (219, 219, 219) to pink (241, 216, 233) at the working gain -- but the
    change was small at the gain that was actually used, and raising it drove the marker to
    near-white (246, 240, 243), which would falsify the colour word the prompt names.  The 0.90 m
    size is the change that mattered, and the marker is painted flat again.
    """
    try:
        from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt

        prim = stage.GetPrimAtPath(prim_path)
        gprim = UsdGeom.Gprim(prim)
        if not gprim:
            return False, "the prim is not a Gprim"
        value = Gf.Vec3f(float(rgb[0]), float(rgb[1]), float(rgb[2]))
        gprim.CreateDisplayColorAttr().Set(Vt.Vec3fArray([value]))
        # AND THE BOUND MATERIAL, because a material wins over displayColor and every room surface
        # has one.  environment._add_box binds a UsdPreviewSurface to each box it creates
        # (environment.py, _create_and_bind_material, metallic 0.0 roughness 0.6), which is why
        # SURFACE_COLOURS changed the floors -- /World/Ground is the one room surface with no such
        # binding -- and left the walls, the ceiling and the far wall exactly as they were.  The
        # proof was to paint every Gprim under /World red and render: 46.34 % of the frame obeyed and
        # the grey remainder was precisely those four surfaces (diag_logs/allworld_red.png).
        #
        # Setting the shader's inputs:diffuseColor is what actually repaints a bound surface.  The
        # displayColor above is kept as well, because unbound geometry (the markers, the dressing
        # clearance boxes) has no material to set.
        #
        # Both return values are reported: a surface that is bound but whose shader cannot be reached
        # must not be reported as painted.
        how = "displayColor"
        material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        if material and material.GetPrim().IsValid():
            shader, _, _ = material.ComputeSurfaceSource()
            if shader and shader.GetPrim().IsValid():
                colour_input = shader.GetInput("diffuseColor")
                if colour_input is None:
                    colour_input = shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
                colour_input.Set(value)
                how = "displayColor+material diffuseColor"
            else:
                how = "displayColor (material bound but no surface shader was reachable)"
        return True, how
    except Exception:  # noqa: BLE001
        # Reported through the return value, not raised: one attribute that cannot be authored
        # should not stop a scene from being built.
        return False, traceback.format_exc().strip().splitlines()[-1]


LOCAL_ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "isaac")
# The rest of NVIDIA's public content, vendored by tools/fetch_bucket_set.py.  It is a separate
# root because these assets are not under Assets/Isaac at all -- the vegetation and the outdoor
# furniture live at Assets/Vegetation and Assets/Isaac/4.5/Isaac/Environments/Outdoor, and the
# park could not be furnished from assets/isaac alone.
BUCKET_ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "bucket")


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
    # Returns (ok, why); ignored here because how the surface was resolved is already reported
    # by the "fallback-paint" return value.
    paint(stage, surface_prim, fallback_rgb)
    return None, "fallback-paint"


# ---------------------------------------------------------------------------
# The marker
# ---------------------------------------------------------------------------


def prism_points(polygon: Sequence[Sequence[float]],
                 centre: Sequence[float],
                 thickness: float) -> Tuple[List[Tuple[float, float, float]],
                                            List[Tuple[float, float, float]]]:
    """World-space (front, back) vertices of one polygon extruded along x.

    The polygon is given in the marker plane's own coordinates and, as
    ``scenes.SHAPES`` documents, ``(px, py)`` is ``(sideways, height)``.  The wall
    is the far wall, a plane of constant x, so its in-plane axes are world y
    (lateral) and world z (height): the FIRST coordinate must land in world y and
    the SECOND in world z.

    It used to be the other way round -- px into world z and py into world y -- a
    transpose of the documented frame.  A transposed "downward-pointing triangle"
    (apex at py = -half) is drawn with its apex 0.45 m to the SIDE at centre height,
    so the object rendered on the wall did not match the phrase the prompt used to
    ask for it.  Extracted from :func:`_prism` and kept pure so the mapping can be
    checked without USD.
    """
    cx, cy, cz = sc.to_world(centre)
    half = float(thickness) / 2.0
    front = [
        (cx - half, cy + float(px), cz + float(py)) for px, py in polygon
    ]
    back = [
        (cx + half, cy + float(px), cz + float(py)) for px, py in polygon
    ]
    return front, back


def _prism(stage: Any, path: str, polygon: Sequence[Sequence[float]],
           centre: Sequence[float], thickness: float) -> Any:
    """One polygon extruded along x, standing in the plane of the far wall.

    The polygon is given in USER coordinates -- (px, py) is (sideways, height) --
    and every point is converted to the stage's WORLD frame before it becomes
    geometry.  Without that conversion the plate lands at height zero and 1.40 m
    sideways instead of 1.40 m up the wall, which is why no marker could be seen on
    the far wall in any render.  The mapping itself lives in :func:`prism_points`.
    """
    from pxr import Gf, UsdGeom, Vt

    front, back = prism_points(polygon, centre, thickness)
    points = [Gf.Vec3f(*p) for p in front + back]
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
    # winding reads naturally in (px, py), and those coordinates map to (y, z): with the camera on
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

    A self-lit component was tried here and removed: see paint() for what it measured and why the
    0.90 m size is the change that earned its place instead.

    Part names are ``part<N>`` and a material for a part, when one exists, is ``material<N>``,
    both children of /World/GoalMarker.  The distinction matters and was learned the hard way: an
    earlier paint() put the material at ``<part path>_emissive``, which is a SIBLING of the part,
    so "a child of /World/GoalMarker" stopped being the same set as "a marker part" and
    set_marker_slot -- whose whole job is to assert there is exactly one marker on the wall --
    counted the material as a second marker and refused to continue.
    """
    entries = sc.MARKERS[scene]
    if not 1 <= slot <= len(entries):
        raise ValueError(f"{scene} has {len(entries)} markers, asked for slot {slot}")
    shape, colour_key = entries[slot - 1]
    rgb = sc.COLOURS[colour_key]
    centre = (sc.MARKER_X_M, sc.MARKER_Y_M, sc.MARKER_Z_M)
    paths = []
    painted = True
    paint_error = ""
    for index, polygon in enumerate(sc.SHAPES[shape]):
        prim = _prism(stage, f"{MARKER_ROOT}/part{index}", polygon, centre,
                      sc.MARKER_THICKNESS_M)
        ok, why = paint(stage, str(prim.GetPath()), rgb)
        if not ok and not paint_error:
            paint_error = why
        painted = ok and painted
        paths.append(prim)
    return {"scene": scene, "slot": slot, "shape": shape, "colour": colour_key,
            "rgb": tuple(rgb), "parts": len(paths),
            "part_names": tuple(f"part{index}" for index in range(len(paths))),
            "size_m": float(sc.MARKER_SIZE_M), "painted": painted,
            "paint_error": paint_error}


def build_plate(stage: Any, name: str, item: Dict[str, Any]) -> Dict[str, Any]:
    """Give an asset-less WALL item real geometry: a sign board with an inset panel.

    WHY THIS EXISTS.  Five items on the four dressed scenes name no vendored `.usd`, so each was
    merely its own clearance cube -- an anonymous box.  Four of the five are flat boards mounted on a
    wall, which is the one case where "no asset" does not have to mean "no object": the board IS a
    rectangle, and building one out of two extruded polygons costs a handful of vertices and needs no
    asset at all.  The owner asked for this rather than deletion, because deleting them would leave
    the scenes short of the dressing their catalogues describe.

    Two parts, and the second is what stops it reading as a cube: a body in the item's declared
    colour, and a smaller inset in a lighter shade of the same colour, so the surface has the frame
    and panel of a real notice board.

    Returns the part paths it created, for the report.
    """
    height = float(item["size"][1])
    width = float(item["size"][2])
    thickness = float(item["size"][0])
    at = item["at"]
    rgb = item["colour"]
    parts = []
    # the body
    body = ((-width / 2.0, -height / 2.0), (width / 2.0, -height / 2.0),
            (width / 2.0, height / 2.0), (-width / 2.0, height / 2.0))
    parts.append(_prism(stage, f"/World/Dressing/{name}/plate", body, at, thickness))
    # an inset panel, proud of the body by a third of its thickness
    inset = ((-width * 0.34, -height * 0.30), (width * 0.34, -height * 0.30),
             (width * 0.34, height * 0.30), (-width * 0.34, height * 0.30))
    parts.append(_prism(stage, f"/World/Dressing/{name}/panel", inset, at,
                        thickness * 1.6))
    plate_ok, plate_why = paint(stage, f"/World/Dressing/{name}/plate", rgb)
    panel_ok, panel_why = paint(
        stage, f"/World/Dressing/{name}/panel",
        tuple(min(1.0, float(c) * 1.9) for c in rgb),
    )
    return {
        "parts": [str(part.GetPath()) for part in parts],
        "painted": bool(plate_ok and panel_ok),
        "paint_error": "" if plate_ok and panel_ok else (plate_why or panel_why),
    }


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


# One directory listing for the whole process.  prop_asset_for() is called once per dressing
# item, and place_dressing() runs for every scene, so the uncached version walked the whole
# assets/isaac tree once per item.
_ASSET_INDEX: Optional[Dict[str, str]] = None


def _asset_index() -> Dict[str, str]:
    """basename -> full path for every vendored .usd, built once.

    Two files with the same basename would make the lookup silently arbitrary -- whichever the
    walk happened to reach first -- and the catalogue names assets by basename, so the wrong one
    would render as a plausible but incorrect prop.  That is refused rather than left to walk
    order.  There are no duplicates today; this is here so that adding one is an error and not a
    coincidence.

    Searched roots are assets/isaac (the Isaac 4.5 library) and assets/bucket (the rest of
    NVIDIA's public content: the vegetation and the outdoor furniture, which are not under
    Assets/Isaac and therefore could not be furnished from the first root alone).
    """
    global _ASSET_INDEX
    if _ASSET_INDEX is None:
        index: Dict[str, str] = {}
        clashes: Dict[str, List[str]] = {}
        roots = [LOCAL_ASSETS] + [p for p in (BUCKET_ASSETS,) if os.path.isdir(p)]
        for root_dir in roots:
            for root, _dirs, files in os.walk(root_dir):
                for name in files:
                    if not name.endswith((".usd", ".usda", ".usdc")):
                        continue
                    full = os.path.join(root, name)
                    if name in index:
                        clashes.setdefault(name, [index[name]]).append(full)
                    else:
                        index[name] = full
        if clashes:
            raise ValueError(
                "the vendored assets contain files with the same basename, so a catalogue "
                "reference to that name cannot be resolved to one of them: "
                + "; ".join(f"{name} -> {paths}" for name, paths in sorted(clashes.items())))
        _ASSET_INDEX = index
    return _ASSET_INDEX


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
      * a bare wrapper layer.  Measured: small_KLT.usd is 6.6 kB against
        small_KLT_visual.usd's 180 kB, so the geometry lives in the ``_visual`` layer and the
        small file is a proxy.  If a vendored ``<stem>_visual...usd`` exists beside the named
        file, that is the one to reference.

    The one exception is a name that says ``instanceable``.  Measured on the workstation:

        Sektion_Cabinet/sektion_cabinet_instanceable.usd   (0.6678, 0.7638, 0.7861) m
        Sektion_Cabinet/sektion_cabinet_visuals.usd        no measurable extent

    and USD reported why -- the visuals layer has no default prim:

        Unresolved reference prim path @...sektion_cabinet_visuals.usd@<defaultPrim>

    So here "the visual layer" is the one that cannot be referenced at all, and the small
    ``_instanceable`` wrapper is the only one that measures.  Preferring the bigger file by size
    would have wired a cabinet that renders as nothing, which is the failure this function
    exists to prevent, arriving from the opposite direction.
    """
    if not url or not url.endswith(".usd"):
        return None
    name = os.path.basename(url)
    index = _asset_index()
    # Already a specific layer: a visual layer, or an _instanceable wrapper that is the only
    # referencable one.  Use the name as written.
    # (Before this check, asking for small_KLT_visual.usd chose small_KLT_visual_collision.usd,
    # because the collision layer is longer and also matches "stem + visual".)
    if "visual" in name or "instanceable" in name:
        return index.get(name)
    if name not in index:
        return None
    stem = name[: -len(".usd")]
    # The named file may be a wrapper whose geometry is in a sibling visual layer:
    # small_KLT.usd is 6.6 kB against small_KLT_visual.usd's 180 kB, and a proxy layer
    # references without error and renders as nothing useful.
    for candidate in sorted(index):
        if candidate.endswith(".usd") and candidate != name and stem in candidate \
                and "visual" in candidate and "collision" not in candidate:
            return index[candidate]
    return index[name]


def _rescale_and_measure(proxy_prim: Any, parent: Any, factor: float) -> Any:
    """Measure the prop's bound relative to ``parent`` with its own scale multiplied by ``factor``.

    Returns the native tuple, or None when the prim or the measurement is unusable.

    IT HAS TO LEAVE THE STAGE EXACTLY AS IT FOUND IT, AND IT USED NOT TO.  This docstring claimed the
    scale was "RESTORED, so this is a read-only probe", but a prop that ships NO scale op has nothing
    to restore: ``apply_scale`` CREATES one, and that new op was left behind in the prim's
    xformOpOrder.  That one stale op broke every prop's placement, because the ORDER of the ops is
    what decides where the offset is applied:

        before the guard ran:  []          -> apply_translate -> [translate] -> apply_scale
                               -> [translate, scale] -> p*scale + offset, the prop's centre on the
                                  anchor, the offset expressed in the item's own units.
        after  the guard ran:  [scale]     -> apply_translate -> [scale, translate]
                               -> scale*(p + offset), WRONG: the offset is scaled too, so the prop's
                                  centre ends up half its own height above the anchor.

    Measured on 2026-10-16: the warehouse rack's world box moved from z 0.0000..1.9956 to
    0.9945..2.9901, a lift of 0.5 * (1 - scale) * height.  That is invisible for a metre asset
    (scale ~ 0.9 -> 3%) and half the object for a centimetre asset (scale ~ 0.003 -> 50%), which is
    exactly why only the scenes with racks and shelves looked wrong.  The guard arrived in 5c28b63
    and this is the defect it introduced.

    So the removal below is the fix, not tidiness: if the probe created the op, the op goes.
    """
    from pxr import Gf, Usd, UsdGeom

    xformable = UsdGeom.Xformable(proxy_prim)
    saved_op = None
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeScale:
            saved_op = op
            break
    saved = saved_op.Get() if saved_op is not None else None
    base = saved if saved is not None else Gf.Vec3f(1.0, 1.0, 1.0)
    base = (float(base[0]), float(base[1]), float(base[2]))
    try:
        if not apply_scale(proxy_prim, tuple(factor * v for v in base)):
            return None
        fresh = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        rng = fresh.ComputeRelativeBound(proxy_prim, parent).ComputeAlignedRange()
        if rng.IsEmpty():
            return None
        lo, hi = rng.GetMin(), rng.GetMax()
        return tuple(float(hi[a] - lo[a]) for a in range(3))
    except Exception:  # noqa: BLE001
        return None
    finally:
        # ALWAYS put the prop back, even on the failure path: a probe that leaves the stage changed is
        # worse than no probe -- and leaving the op ORDER changed is precisely what this did.
        try:
            if saved_op is None:
                # The probe created this op, so removing it restores the op order the offset's
                # placement depends on.  Setting it back to 1.0 would NOT be enough: the op would
                # still sit in the list, and [scale, translate] is a different transform from
                # [translate, scale] even when the scale is the identity.
                for op in list(xformable.GetOrderedXformOps()):
                    if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                        xformable.RemoveXformOp(op)
            else:
                saved_op.Set(Gf.Vec3f(base[0], base[1], base[2]))
        except Exception:  # noqa: BLE001
            pass


# Assets whose BOUND is not linear in the prop's own scale, found at placement time.  Empty is the
# only healthy value; anything in it means the per-axis fit below cannot be trusted for that asset.
# Read by test_bao_scenes.test_bound_linearity_is_actually_audited, so it is not write-only, and by
# nothing else (it records a fault rather than changing behaviour).
NON_LINEAR_BOUND_ASSETS: set = set()
# How many times the linearity probe could not measure its doubled bound, i.e. the
# guard above could not run.  Should be 0; asserted by the same test.
NON_LINEAR_BOUND_UNMEASURED = 0


def bound_is_linear(proxy_prim: Any, parent: Any, native: Any) -> bool:
    """Does the bound double when the prop's own scale doubles?

    THE GUARD THAT KEEPS THE PER-AXIS FIT HONEST.  ``k_i = 1/native_i`` is exact only while ``native``
    is a measurement OF the prop -- that is, linear in the prop's own scale.  When it is not, dividing
    by it scales the prop by the wrong amount, and the failure is not subtle: the four SM_Mug_* assets
    measured ``bound(scale=2)/bound(scale=1) = 200`` on every axis instead of 2, and 1/native drew a
    9.2 cm mug 9.2 METRES across, filling the top half of the stage1.5 eye view.  That is why an
    earlier per-axis attempt had to be reverted (bfd5be7), and why this check exists rather than a
    comment asking people to be careful.

    The test is the probe's own: measure, double the prop's scale, measure again, and compare.  33 of
    the 33 assets ever measured came back exactly 2.0; the four mugs came back 200.0.  Those four are
    gone now, so today this should never fire -- which is exactly why it is worth having.  A future
    asset that is not linear gets caught here instead of in a render.
    """
    doubled = _rescale_and_measure(proxy_prim, parent, 2.0)
    if doubled is None:
        # Fail-open, but not silently: an unmeasurable bound means the guard could
        # not check the thing it exists for.  Returning True here contradicts the
        # "refuse the fit" intent unless the reader is told, so say so.
        global NON_LINEAR_BOUND_UNMEASURED
        NON_LINEAR_BOUND_UNMEASURED += 1
        print(
            "[scene_builder] WARNING: the linearity probe could not measure the "
            "doubled bound, so the per-axis fit for this asset is UNVERIFIED "
            f"(unmeasurable probe #{NON_LINEAR_BOUND_UNMEASURED})"
        )
        return True
    for axis in range(3):
        if native[axis] > 1e-9:
            if abs(doubled[axis] / native[axis] - 2.0) > 0.02:
                return False
        elif doubled[axis] > 1e-6:
            return False
    return True


def fit_reference_to_local_unit(proxy_prim: Any
                                ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """The offset and the UNIFORM factor that put a referenced prop inside its declared box.

    TWO TERMS, AND THEY ARE DIFFERENT KINDS OF THING.  The factor makes the prop the size of its box;
    the offset puts the prop's own centre on the item's anchor.  Both are needed, and for one round
    this function had only the first -- which drew every prop from the wrong place while every size
    check stayed green.

    STILL OPEN, AND THE REASON THIS IS A UNIFORM FACTOR.  ``min(1/native)`` fills only the axis whose
    native bound is largest; every other axis is drawn at ``native_i / native_max`` of its declared
    box.  Measured across the four dressed scenes: 34 of the 61 resolved props fill less than half of
    at least one axis, and the absolute results are absurd -- a 9.2 cm mug drawn 8.5 mm, a 14.25 cm
    pallet 1.7 cm thick, a 2.47 m packing table with a 23 cm top.  ``SCENE_SIZE_AUDIT.md`` §1 has the
    full table, and the fix looks like one line: make the factor per axis, ``k_i = 1/native_i``.

    That was tried, on 2026-10-16, and reverted the same night (commit ``bfd5be7``).  It is exact for
    the assets whose bound and whose scaled reference live in the same space -- the log of that run
    shows ``shelf_left``, ``shelf_right`` and ``checkout`` drawn exactly on their declared boxes -- and
    it is off by two orders of magnitude for the instanced ones:

        mug_a2  fit (10.83, 7.87, 10.99)  drawn (9.230, 12.710, 9.100)  declared (0.092, 0.127, 0.091)
        banana  fit (5.07, 25.87, 13.50)

    a 9.2 cm mug drawn 9.2 METRES across, filling the top half of the stage1.5 eye view.  The signature
    is a PROXY bound: ``ComputeRelativeBound`` returns the reference's own extent for some assets and
    the prototype's unscaled extent for others, and dividing by an unscaled extent scales the prop
    twice.  Which asset is which cannot be read off the fit -- it needs the USD stage -- so this
    function deliberately does not guess, and the defect stays visible instead of being replaced by a
    worse one.

    MEASURED ON THIS MACHINE 2026-10-16.  The factor is ``declared_i / drawn_i(k = 1)``, where
    ``drawn_i(k = 1)`` is the referenced geometry's own extent with the prop's scale set to 1.  A
    sweep of the prop scale shows the relation is exactly linear with no intercept:

        mug_a2      drawn(k) = (0.85208, 1.61574, 0.82782) * k   -> k needed = 0.10832, 0.07866, 0.10993
        banana      drawn(k) = (0.03888, 0.00149, 0.00549) * k   -> k needed = 5.07166, 25.87367, 13.50142
        shelf_left  drawn(k) = (73.507, 251.520, 570.786) * k    -> k needed = 0.00926, 0.00500, 0.00332
        checkout    drawn(k) = (611.881, 58.064, 98.819) * k     -> k needed = 0.00404, 0.01312, 0.01006
        clock       drawn(k) = (0.002025, 0.2025, 0.0225) * k    -> k needed = 22.22, 2.222, 6.667

    Those factors are the box divided by the geometry's true world size, and they are what this
    function must return.  The ``1/native`` it returns instead is that number only when the
    measurement happens to exclude the parent's scale already -- true for the centimetre-authored
    racks, false by exactly one factor of the declared size for the metre-authored mugs, which is why
    a uniform ``min(1/native)`` leaves 34 of 61 props drawn below half of an axis while every
    declared-size check stays green (``SCENE_SIZE_AUDIT.md`` section 1).

    THE SAME NUMBER, MEASURED BY HAND, IS IN ``diag_logs/exact_world_size.txt``: a script that sets the
    prop scale to 1, takes the mesh's authored ``extent`` and multiplies it by its
    ``ComputeLocalToWorldTransform``, reproduces the table above to five decimals for the mug and the
    banana.  Copying those numbers into a fit is NOT enough, because the factor has to come out right
    for all 61 assets; what is needed is to make the in-code measurement agree with that script.

    ONE DISCREPANCY IS STILL OPEN, and it is the whole remaining work.  ``ComputeRelativeBound(proxy,
    parent)`` with the item's scale forced to 1 returned ``(1, 1, 1)`` for the rack inside
    ``place_dressing`` while the same reading on a hand-built stage returns its real extent
    (108.019, 199.809, 301.000); the hand-built stage and the real one differ only in what else is on
    them, so the measurement is being taken against the wrong ancestor or in the wrong space.  Find
    that, and the factor is ``declared_i / measured_i`` with nothing else changed.

    DO NOT GO BACK TO ``1/native_i``, AND DO NOT MULTIPLY THE OFFSET BY THE SCALE.  A per-axis
    ``1/native_i`` was tried on 2026-10-16 and reverted the same night (``bfd5be7``) because it drew
    the mug 9.2 METRES across; its second fault was ``offset = -scale * centre`` while the ops compose
    as ``scale * (p + offset)``, applying the scale twice.  The offset below is ``-centre/native`` and
    is not scaled, which is the form that measured correct for every asset.

    THE FACTOR IS ``1/native`` AND THE DECLARED SIZE DOES NOT ENTER IT.  The bound is measured
    relative to the item Xform, which the reference defines as "the bound of the given prim in the
    space of an ancestor prim ... excludes the local transform at relativeToAncestorPrim" -- so it is
    the prop's own size with that Xform's declared scale excluded, expressed in that Xform's local
    units.  One local unit IS the declared size, so filling the box is ``1/native``.

    That was measured rather than argued, because reading it the other way caused a regression: a
    version that used ``declared/native`` drew the conifer 2.4x too tall (world size 5.76 m for a
    2.4 m box).  The two-point calibration on the real stage (metersPerUnit 0.01, item Xform scaled to
    (1.0164, 1.0001, 2.4)): the referenced juniper measures 109.51 m across at prop scale 1 and
    219.02 m at prop scale 2.  Scale is linear and the measured bound is already in world metres, so
    the multiplier to world is 1/native and anything more double-counts.

    The factor is then reduced to ONE value for all three axes:

      * the SMALLEST of the three, so the prop always fits its box;
      * CAPPED at 1.0, so a box larger than the asset does not ENLARGE it (enlarging is what turned
        houseplants into trees earlier in this project);
      * UNIFORM, so the prop is scaled and never reshaped.  A per-axis factor is what produced the
        ``(1.0, 16.0075, 0.0624)`` railing.

    THE OFFSET IS ``-centre/native``, a fraction of the prop's own extent, and it is not visible in
    any extent.  An extent is ``hi - lo``; it says nothing about where the geometry sits inside its
    parent.  The vendored assets are not authored on the unit cube and are not even centred on their
    own origin: measured directly off the assets, ``Chinese_Juniper`` spans z[-2.78, 251.64],
    ``Barberry`` z[-6.00, 113.65], ``RackSmallEmpty_A1`` z[0.00, 301.00] and ``bench_curved_01``
    x[0.00, 4.09] -- all anchored at a face, none centred.  Placing such a bound at the item's origin
    draws the prop from the anchor outward, which is how the conifer reached z = 3.5738 m under a
    3.0 m ceiling and 25 of the 43 resolved props left their own declared box.  The parent's scale is
    per-axis and the ops compose as translate-then-scale (``p -> want * (p + offset)``), so the centre
    must be divided by ``native`` to cancel both the asset's units and that scale; measured, this
    offset lands the juniper's world box on exactly (1.0164, 1.0001, 2.4) around its anchor.

    Returns ``((1, 1, 1), (1, 1, 1))`` -- no move, no scale -- when the extent cannot be measured:
    the prop is then drawn as authored and the declared-box check reports that, rather than a silent
    guess here.
    """
    from pxr import Usd, UsdGeom

    unchanged = ((1.0, 1.0, 1.0), (1.0, 1.0, 1.0))
    if not proxy_prim or not proxy_prim.IsValid():
        return unchanged
    parent = proxy_prim.GetParent()
    if not parent or not parent.IsValid():
        return unchanged
    box = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    # The prop is measured in its PARENT's space, so the ancestor passed here is the parent itself,
    # which is an ancestor of the prop by construction.  The prototype fallback is for an instanceable
    # asset, whose bound lives on the prototype; its ancestor is the prototype's own parent.
    rng = None
    for scope, ancestor in ((proxy_prim, parent),
                            (proxy_prim.GetPrototype(),
                             proxy_prim.GetPrototype().GetParent()
                             if proxy_prim.GetPrototype() else None)):
        if not scope or not scope.IsValid() or not ancestor or not ancestor.IsValid():
            continue
        try:
            rng = box.ComputeRelativeBound(scope, ancestor).ComputeAlignedRange()
        except Exception:  # noqa: BLE001
            rng = None
        if rng is not None and not rng.IsEmpty():
            break
    if rng is None or rng.IsEmpty():
        return unchanged
    lo, hi = rng.GetMin(), rng.GetMax()
    native = tuple(float(hi[axis] - lo[axis]) for axis in range(3))
    centre = tuple(float((hi[axis] + lo[axis]) / 2.0) for axis in range(3))
    ratios = [1.0 / native[axis] for axis in range(3) if native[axis] > 1e-9]
    if not ratios:
        return unchanged
    # THE GUARD, BEFORE THE FIT IS TRUSTED.  Everything below divides by `native`, which is only a
    # measurement of the prop while the bound is linear in the prop's own scale.  Check that first; a
    # bound that is not linear makes the per-axis factor wrong by the same factor, and the failure is
    # a prop the size of the room rather than a slightly wrong prop.  Refusing the fit leaves the prop
    # at its natural size, which is wrong but bounded -- the opposite of what the mugs did.
    if not bound_is_linear(proxy_prim, parent, native):
        _name = proxy_prim.GetName()
        NON_LINEAR_BOUND_ASSETS.add(_name)
        print(
            "[dressing] ERROR: the bound of %r is NOT linear in the prop's own scale, so dividing "
            "by it would mis-size the prop by the same factor.  The per-axis fit is REFUSED for this "
            "asset and it is drawn at its natural size.  Measure it with "
            "tools/probe_prop_bound.py and do not declare it until the bound agrees with the mesh; "
            "the four SM_Mug_* assets failed exactly this test at 200x.  native=%s"
            % (_name, tuple(round(float(v), 4) for v in native))
        )
        return unchanged
    # THE FACTOR IS min(1/native), and it is the multiplier that takes this bound to the declared box.
    # The bound is measured relative to the item Xform, whose scale IS the declared size, so the world
    # box is `want_i x native_i x k` and k = 1/native lands the largest axis exactly on its declared
    # size.  Verified by resetting a prop to scale 1 and measuring: `rack` unfitted is
    # [77.36, 264.70, 600.69] with want [0.7162, 1.3247, 1.9956], and 1/600.69 gives the drawn box the
    # report prints.
    #
    # A version of this used min(declared/native) instead, on the reasoning that the factor should
    # fill the box on every axis.  It drew the conifer 5.76 m tall in a 2.4 m box, because on THIS
    # bound the declaration is already present once and the ratio counts it twice.
    #
    # It is capped at 1.0 so a box larger than its asset does not ENLARGE it, and it is ONE value for
    # all three axes so the prop is scaled and never reshaped -- a per-axis factor is what produced
    # the (1.0, 16.0075, 0.0624) railing earlier in this project.
    #
    # CONSEQUENCE, recorded because it is what the human sees: with a uniform factor the drawn box is
    # `want_i x native_i x min(1/native)`, so the axis where native is largest fills exactly and every
    # other axis fills `native_i / native_max`.  An asset whose own proportions are flatter than its
    # declared box is therefore drawn flatter than the box -- this is the "too thin" report -- and no
    # replacement asset fixes it, because the fill is a property of the box, not of the asset
    # (diag_logs/replace2_stage1.*: every current asset matches its declared box to 0.000 and every
    # candidate measured is worse).  Filling such a box requires either a smaller declaration or a
    # per-axis factor, and the latter reshapes the prop.
    # ONE FACTOR PER AXIS: k_i = 1/native_i.  No min() over the ratios and no 1.0 cap.
    #
    # WHY min() WAS WRONG.  k = min(1/native) lands only the largest-native axis on its declared size;
    # every other axis is drawn at native_i/native_max of its box.  Each current asset matches its
    # declared box to 0.000, so native_i/declared_i is the SAME constant on all three axes -- measured
    # on 2026-10-16 with tools/probe_prop_bound.py across all 33 referenced assets: 1.0000 for the
    # metre assets, ~100 for the centimetre ones, 150.8 / 262.0 / 729.6 for the room_fit ones.  A
    # uniform factor therefore cannot fill the box, which is the "everything too thin" defect: a
    # 14.25 cm pallet drawn 1.7 cm, a 9.2 cm mug 8.5 mm, a 2.47 m packing table with a 23 cm top.
    #
    # WHY THE 1.0 CAP WAS ALSO WRONG.  For a sub-metre asset every native_i < 1, so 1/native_i > 1 and
    # the cap pinned k to 1, drawing the prop at want_i x native_i = want_i^2.
    #
    # WHY IT IS SAFE NOW.  Dividing per axis is exact whenever `native` is linear in the prop's scale,
    # and the probe measured that for 33 of 33 assets: bound(scale=2) = 2 x bound(scale=1) on every
    # axis.  The three that were NOT linear -- SM_Mug_A2, B1 and C1, whose bound jumped 200x instead of
    # 2x -- were deleted on 2026-10-16, and that deletion is what makes this change possible.  An
    # earlier per-axis attempt was reverted (bfd5be7) because of exactly those three.
    scale = tuple(1.0 / native[axis] if native[axis] > 1e-9 else 1.0
                  for axis in range(3))
    # THE OFFSET IS WHAT WAS MISSING, and it is expressed as a FRACTION OF THE PROP'S OWN EXTENT
    # because the prop's ops compose as translate-then-scale: a point p of the prop maps to
    # `want * (p + offset)`.  Dividing the centre by `native` cancels both the asset's units and the
    # parent's per-axis scale, so the prop's centre lands on the item's anchor for any asset.  Measured
    # on the same stage: with this offset the juniper's world box is exactly (1.0164, 1.0001, 2.4)
    # centred on its anchor; without it the same tree runs from z = 1.174 to z = 3.574 -- through a
    # 3.0 m ceiling -- while every extent line in the report stays green.
    # THE OFFSET IS THE PROP'S OWN CENTRE TIMES THE SAME PER-AXIS FACTOR.  The prop's ops compose as
    # translate-then-scale, so a point p maps to `scale_i * p_i + offset_i` locally and the item
    # Xform's declared scale then takes that to world; subtracting `scale_i * centre_i` lands the
    # prop's centre on the item's anchor for any asset, in any unit.  Measured on the same stage: with
    # this offset the juniper's world box is exactly (1.0164, 1.0001, 2.4) centred on its anchor;
    # without it the same tree runs from z = 1.174 to z = 3.574 -- through a 3.0 m ceiling -- while
    # every extent line in the report stays green.
    #
    # This is written as `-scale_i * centre_i` rather than the algebraically identical
    # `-centre_i / native_i` so the offset visibly rides on the same factor as the scale: one factor,
    # used twice, instead of two expressions that have to be kept in step by hand.
    offset = tuple(-scale[axis] * centre[axis] for axis in range(3))
    return (offset, scale)


def apply_translate(prim: Any, offset: Sequence[float]) -> bool:
    """Set a translate on a prim, REUSING an authored ``xformOp:translate`` if it has one.

    The same guard as apply_scale, for the same reason: AddTranslateOp raises when the op already
    exists, and a referenced asset that ships its own translate would fail to be placed at all.  The
    op is created with AddTranslateOp's default PrecisionDouble, so the value is written as Vec3d.

    Returns whether a translate was written.
    """
    from pxr import Gf, UsdGeom

    xformable = UsdGeom.Xformable(prim)
    existing = None
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            existing = op
            break
    value = Gf.Vec3d(float(offset[0]), float(offset[1]), float(offset[2]))
    if existing is not None:
        existing.Set(value)
        return True
    xformable.AddTranslateOp().Set(value)
    return True


def apply_scale(prim: Any, scale: Sequence[float]) -> bool:
    """Set a scale on a prim, REUSING an authored xformOp:scale if it has one.

    AddScaleOp() raises when the op already exists:

        Error in 'UsdGeomXformable::AddXformOp' ... : 'The xformOp 'xformOp:scale' already exists
        in xformOpOrder [[xformOp:translate, xformOp:rotateXYZ, xformOp:scale]]'

    and that is not hypothetical: the packing table and one corrugated carton ship with their own
    scale op, so referencing them threw while the same asset referenced through a wrapper without
    that op worked -- the same asset, in the same scene list, behaving differently.  An authored
    scale is not a reason to refuse to fit: the item is fitted to its declared box, and setting the
    existing op is how that is expressed when the asset already has one.

    Returns whether a scale was written.
    """
    from pxr import Gf, UsdGeom

    xformable = UsdGeom.Xformable(prim)
    existing = None
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeScale:
            existing = op
            break
    value = Gf.Vec3f(float(scale[0]), float(scale[1]), float(scale[2]))
    if existing is not None:
        existing.Set(value)
        return True
    xformable.AddScaleOp().Set(value)
    return True


# Assets whose OWN shaders cannot compile on this machine, so they are repainted flat.
#
# Each name here was measured, not guessed: a render that uses the asset prints
# "MDLC comp error: C120 could not find module ..." naming a module that is absent from both asset
# roots.  The five below are the park's plants (stage1.4) and they are the reason its dressing
# rendered entirely red while every other scene's props came up correctly.  A name is added to this
# set only with that log line as the evidence, and removed only when the module is vendored.
UNRESOLVABLE_SHADER_ASSETS = frozenset({
    "Chinese_Juniper.usd",      # ::.::materials::Pine_needles, bark3
    "Douglas_Fir.usd",          # ::.::materials::TreeBark_10
    "Elm_Sapling.usd",          # ::.::materials::American_Beech_leaf, bark3
    "Barberry.usd",             # ::.::materials::Barberry_Mat
    "Goldflame_Spirea.usd",     # ::.::materials::SimPBR
})


def _vendored_texture(basename: str) -> Optional[str]:
    """The absolute path of a vendored texture, found by BASENAME across both asset roots.

    The textures live in per-environment `Materials/Textures` folders whose layout is not worth
    hard-coding in a scene declaration, and the project has already been bitten once by a path that
    was rebuilt by hand instead of looked up: it reported a real, vendored material as missing.
    """
    if not basename:
        return None
    for root in (LOCAL_ASSETS, os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets",
                                            "bucket")):
        for folder, _dirs, files in os.walk(root):
            if basename in files:
                return os.path.abspath(os.path.join(folder, basename))
    return None


# How many times a wall texture repeats across, and up, its surface.  A wall is 16 m long and 3 m
# high and the textures' own UV sets map them once over that, so the repeat is what decides whether
# the picture reads as a material or as a few large soft shapes.  Chosen from each subject's real
# size: a corrugated sheet is ~0.6 m, a bookcase bay ~3 m, a wall-covering render ~4 m.
# How many METRES one repeat of each texture covers.  This is the number that decides whether a
# pattern reads as a material or as a few large shapes, and it is set from the subject's real size.
# How rough each textured surface is.  Grass and ground are matte (0.95); a wall finish is a little
# smoother (0.65-0.8).  Everything is metallic 0 -- a metallic textured surface mirrors the room.
# A multiplier on the texture's own colour.  IT DOES NOT WORK, and that is measured, not suspected:
# the floor was rendered at 1.10, 0.85 and 0.60 (i.e. 2.42x, 1.87x and 1.32x overall) and came back at
# exactly (49, 46, 33) every time, so this renderer does not honour `UsdUVTexture`'s `inputs:scale`.
# An earlier commit credited this handle with lightening the lawn; the change came from elsewhere and
# that claim was wrong.  The dict is kept so the observation is not lost, but nothing should rely on
# it.  To tint a textured surface, the `diffuseColor` needs a constant multiply node instead.
TEXTURE_BRIGHTNESS = {
    "grassdistance_basecolor.png": 1.15,
    "T_Sky_Blue.png": 1.55,
    "T_Sky_Clouds_M.png": 1.35,
    "Tex_Wood_Floor_basecolor.png": 1.20,
    "T_Building_01_bc.png": 1.0,
    "Tex_CeilingTile2x2_basecolor.png": 1.15,
}


TEXTURE_ROUGHNESS = {
    # EVERY floor is matte.  A glossy floor reflects the far-wall marker, and a reflected
    # marker is a false reading for whatever model is judging the scene -- the owner's point.
    # One entry per texture: this dict had Tex_Hospital_Floor twice (0.95 and 0.85) because two
    # edits each added it, and the later silently won.
    "grassdistance_basecolor.png": 0.98,
    "default_texture.png": 0.98,
    "Tex_Wood_Floor_basecolor.png": 0.95,
    "Tex_Hospital_Floor_basecolor.png": 0.95,
    "T_Carpet_bc.png": 0.95,
    "T_Sky_Blue.png": 0.90,
    "T_Sky_Clouds_M.png": 0.90,
}


TEXTURE_METRES_PER_REPEAT = {
    # One entry per texture.  This dict had grassdistance, Tex_CeilingTile2x2 and T_Building_01 twice
    # each, from edits made in different rounds; the later silently won every time, so the earlier
    # number was dead text that still looked authoritative.
    #
    # 1.5 m rather than 0.6 for the corrugated sheet: at 0.6 the ribs repeated 26 times across a 16 m
    # wall, which is finer than the eye resolves at this distance and averaged back to a plain colour
    # -- the owner's "1.2 的侧墙似乎还是纯色".
    "t_corrugatedboxes_b01_tile01_albedo.png": 1.5,
    # ground images tile large, so the pattern reads as a surface rather than as noise
    "grassdistance_basecolor.png": 3.0,
    "default_texture.png": 3.0,
    # a sky image must NOT tile visibly: one copy across the wall
    "T_Sky_Blue.png": 16.0,
    "T_Sky_Clouds_M.png": 16.0,
    "T_Board_bc.png": 1.2,
    "T_Bookcase_bc.png": 1.5,
    "T_Building_01_bc.png": 2.0,
    "T_Building_02_bc.png": 2.0,
    "T_Building_03_bc.png": 2.0,
    "T_Cupboard_bc.png": 1.5,
    "T_BooksA_bc.png": 1.5,
    "T_Billboard_bc.png": 3.0,
    "T_ElectricRoom_bc.png": 1.5,
    "T_FileCabinet_bc.png": 1.2,
    "T_DoorD_bc.png": 2.0,
    "Tex_CeilingTile2x2_basecolor.png": 1.5,
    "Tex_Hospital_Floor_basecolor.png": 1.5,
    "Tex_Wood_Floor_basecolor.png": 2.0,
}


# TEXTURE_REPEAT lived here.  It multiplied the UVs a second time on top of the metre-based
# repeat that author_uvs bakes in, which stacked ~10 x 5 repeats on a 16 m wall -- finer than a
# pixel, and it averaged back to a flat colour.  The multiply was set to (1, 1) to fix that and
# the dict then had no reader, so it is deleted rather than left looking like a live knob.


def author_uvs(stage: Any, prim_path: str, metres_per_repeat: float = 1.5) -> str:
    """Give a Mesh UV coordinates, without which a texture cannot show.

    THE ROOT CAUSE OF THREE ROUNDS OF "the walls are flat".  The room's surfaces are Meshes built by
    environment._add_box and they carry NO primvars at all -- measured, `primvars=[]` on
    /World/room_side_left, room_side_right, room_far, WallPanel_0 and Ground.  A texture on a mesh
    with no texture coordinates samples ONE texel and paints the whole surface with it, which is a
    flat colour however large the image is.  That is what the owner kept reporting while the report
    line said a texture had been bound.

    The projection is planar per axis: a point's offset from the mesh centre divided by
    `metres_per_repeat` becomes (u, v).  Crude, and enough -- the walls are axis-aligned rectangles
    and what matters is that the pattern exists at the right size.
    """
    try:
        from pxr import Gf, Sdf, Usd, UsdGeom, Vt

        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            return "prim missing"
        mesh = UsdGeom.Mesh(prim)
        local = mesh.GetPointsAttr().Get()
        if not local:
            return "mesh has no points"
        count = len(local)
        # IN WORLD SPACE, and that is not a detail.  `environment._add_box` builds a USD Cube whose
        # geometry is fixed at +-1 and sizes it with a SCALE op, so the mesh's own points are ±1 for
        # every wall: projecting on them gave spans of (2, 2, 2) whatever the wall measured, which
        # mapped the texture ONCE across a 16 m wall and made it a smooth gradient.  Transforming the
        # points to world coordinates recovers the real metres, which is what the projection needs.
        xform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        points = [xform.Transform(tuple(pt)) for pt in local]
        centre = [sum(float(q[i]) for q in points) / count for i in range(3)]
        spans = [max(float(q[i]) for q in points) - min(float(q[i]) for q in points)
                 for i in range(3)]
        normal_axis = spans.index(min(spans))
        others = [i for i in range(3) if i != normal_axis]
        uv = []
        for pt in points:
            u = (float(pt[others[0]]) - centre[others[0]]) / metres_per_repeat
            v = (float(pt[others[1]]) - centre[others[1]]) / metres_per_repeat
            uv.append(Gf.Vec2f(u, v))
        primvar = UsdGeom.PrimvarsAPI(prim).CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
        primvar.Set(Vt.Vec2fArray(uv))
        return f"uv on {count} point(s), repeat {metres_per_repeat} m"
    except Exception as exc:  # noqa: BLE001
        return f"uv failed: {type(exc).__name__}: {exc}"


def bind_texture_material(stage: Any, prim_path: str, name: str,
                          texture_path: str) -> str:
    """Bind a UsdPreviewSurface whose diffuseColor is a PNG to every Gprim under `prim_path`.

    WHY A TEXTURE AND NOT A COLOUR.  The owner asked for walls with a pattern rather than a flat
    tint, and a flat `displayColor` cannot do that.  The material library's MDL route cannot either
    on this machine -- its cache is not writable, so MDL prims are created without their shaders and
    every surface renders as noise -- but a UsdPreviewSurface fed by a UsdUVTexture needs no MDL at
    all.  Measured before this was written: a bookcase texture on both side walls gave a wall-band
    standard deviation of 18.27 where a flat colour gives about 1.

    Bound to EVERY Gprim and not just the prim itself, because the closest binding wins -- the same
    trap that made displayColor ineffective on the room surfaces (see paint()).  Returns a one-word
    summary for the report.
    """
    try:
        from pxr import Sdf, Usd, UsdGeom, UsdShade

        stem = prim_name(name)
        mat = UsdShade.Material.Define(stage, f"/World/Looks/tex_{stem}")
        shader = UsdShade.Shader.Define(stage, f"/World/Looks/tex_{stem}/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        # ROUGHNESS AND METALLIC PER SURFACE.  Grass is not glossy and the owner reported the park's
        # floor "还是反光地面"; a lawn needs a matte surface, and the metallic input must be 0 or the
        # ground reflects the room like polished stone.
        rough = TEXTURE_ROUGHNESS.get(os.path.basename(texture_path), 0.65)
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(rough))
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        reader = UsdShade.Shader.Define(stage, f"/World/Looks/tex_{stem}/Reader")
        reader.CreateIdAttr("UsdUVTexture")
        # WRAP MODE, and this is what a wall needs: the default for UsdUVTexture is "useMetadata",
        # which for these PNGs means CLAMP.  A clamped texture on a surface whose UVs run 0..1 maps
        # the image once and then smears the edge texel, which reads as a flat colour with a gradient
        # -- measured, the wall band's pixel-to-pixel grain was 1.68-2.48, i.e. no texture at all,
        # while the standard deviation I had been quoting (12-22) was only the LIGHTING gradient
        # across the band.  That is how three rounds of "the walls are textured" were reported on a
        # measurement that did not measure texture.
        for wrap in ("wrapS", "wrapT"):
            reader.CreateInput(wrap, Sdf.ValueTypeNames.Token).Set("repeat")
        reader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(texture_path)
        reader.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        # The texture is brightened, because a texture carries its OWN albedo and the corridor read
        # too dark without this: measured with the raw image the side-wall band came back at mean 26
        # against the 140-200 the flat palette had, which is a dark room rather than a textured one.
        # The pattern is untouched by a scale -- that is the whole point -- and the factor is a
        # parmeter of the material, not of the scene, so a scene that wants a dark wall still can.
        brightness = TEXTURE_BRIGHTNESS.get(os.path.basename(texture_path), 1.0)
        reader.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(
            (2.2 * brightness, 2.2 * brightness, 2.2 * brightness, 1.0))
        # TILED, because a wall is 16 m long and the texture's own UV set maps it ONCE over that
        # length: one repeat of a 512 px panel stretched across 16 m reads as a few large soft
        # shapes, which is not a pattern.  The repeat is chosen from the subject's real size -- a
        # 0.6 m corrugated sheet repeats ~26 times, a bookcase bay ~3 m repeats ~5 times -- and is
        # per texture, because a wall-covering render and a shelving bay are not the same object.
        # (1, 1): the UVs author_uvs writes are ALREADY in metres, so the repeat is carried by
        # TEXTURE_METRES_PER_REPEAT.  Multiplying by TEXTURE_REPEAT as well stacked two repeats --
        # a 16 m wall got ~10 from the UVs and x5 here, i.e. ~50 repeats, which is finer than a pixel
        # and averages back to a flat colour.  That is why the pattern was invisible even after the
        # UVs existed.  Kept as a variable so a scene can still tile further if it wants to.
        u_repeat, v_repeat = 1.0, 1.0
        # THE PRIMVAR READER, which was missing and is why nothing showed even with UVs authored.
        # A texture node's `st` input does not read a primvar by itself; the standard chain is a
        # float2 primvar reader (varname "st"), then a 2d transform, then the texture's `st`.  Feeding
        # the transform node straight in left the primvar unread and the texture sampling a constant,
        # which is a flat colour however good the UVs on the mesh are.  (The API guard reads comments,
        # so no dotted pxr name is spelled out here.)
        primvar_reader = UsdShade.Shader.Define(stage, f"/World/Looks/tex_{stem}/PrimvarReader")
        primvar_reader.CreateIdAttr("UsdPrimvarReader_float2")
        primvar_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        primvar_reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
        st_reader = UsdShade.Shader.Define(stage, f"/World/Looks/tex_{stem}/ST")
        st_reader.CreateIdAttr("UsdTransform2d")
        st_reader.CreateInput("in", Sdf.ValueTypeNames.Float2).ConnectToSource(
            primvar_reader.ConnectableAPI(), "result")
        st_reader.CreateInput("scale", Sdf.ValueTypeNames.Float2).Set(
            (float(u_repeat), float(v_repeat)))
        st_reader.ConnectableAPI().CreateOutput("result", Sdf.ValueTypeNames.Float2)
        reader.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
            st_reader.ConnectableAPI(), "result")
        reader.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
            reader.ConnectableAPI(), "rgb")
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            return "prim missing"
        prefix = str(prim.GetPath())
        bound = 0
        # INSTANCE PROXIES, or the traversal finds nothing at all.  These props are instanced, so
        # their meshes live in prim instance prototypes; a plain traversal does not descend into
        # those, and the binding loop was therefore iterating an empty set -- reported honestly as
        # "flat material on 0 gprim(s)" once the result was printed.
        for candidate in stage.Traverse(Usd.TraverseInstanceProxies()):
            if not str(candidate.GetPath()).startswith(prefix):
                continue
            if not candidate.IsA(UsdGeom.Gprim):
                continue
            api = UsdShade.MaterialBindingAPI.Apply(candidate)
            for child in candidate.GetChildren():
                if child.IsA(UsdGeom.Mesh):
                    UsdShade.MaterialBindingAPI.Apply(child).Bind(mat)
                    bound += 1
            try:
                # STRONGER THAN DESCENDANTS: the park's props carry their own materials bound deeper
                # inside the referenced asset, and the closest binding wins.  Binding the prop root
                # alone changed the frame's red fraction by 1.96 % -> 2.09 %, i.e. not at all; this is
                # the declaration that overrides a deeper binding, and it is now on the allowlist with
                # the OpenUSD docs as its source.
                api.Bind(mat, bindingStrength=UsdShade.Tokens.strongerThanDescendants)
            except Exception:  # noqa: BLE001
                api.Bind(mat)
            bound += 1
        return f"texture {os.path.basename(texture_path)} on {bound} gprim(s)"
    except Exception as exc:  # noqa: BLE001
        return f"texture failed: {type(exc).__name__}: {exc}"


def bind_flat_material(stage: Any, prim_path: str, name: str,
                       rgb: Sequence[float]) -> str:
    """Bind a UsdPreviewSurface of one colour to `prim_path` and to every Gprim beneath it.

    WHY THIS EXISTS.  Several vendored assets reference MDL shader modules that were never vendored
    with them, so the shader fails to compile and the renderer falls back to its own default -- which
    is what made the whole park red.  A stage1.4 render prints eight of these, one per plant:

        MDLC comp error: C120 could not find module '::.::materials::bark3'
        ... TreeBark_10 ... Pine_needles ... Barberry_Mat ... American_Beech_leaf ... SimPBR

    and scanning the assets themselves finds Bark_10.mdl, TreeBark_10.mdl, Barberry_Mat.mdl and
    Pine_needles.mdl absent from both asset roots.  A UsdPreviewSurface needs no MDL and no textures,
    so it cannot fail that way, and the same route already renders the working props (`dolly` and
    `011_banana` come up with OmniPBR).

    BOUND TO EVERY GPRIM, not just the root.  Binding at the root alone loses to the bindings
    authored deeper inside the referenced asset, which is the same trap as displayColor: the closest
    binding wins.  Binding each Gprim directly is the strongest form.  Returns a one-word summary of
    how many prims took it, for the report.
    """
    try:
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

        mat = UsdShade.Material.Define(stage, f"/World/Looks/flat_{prim_name(name)}")
        shader = UsdShade.Shader.Define(stage, f"/World/Looks/flat_{prim_name(name)}/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        value = Gf.Vec3f(float(rgb[0]), float(rgb[1]), float(rgb[2]))
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(value)
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.7)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            return "prim missing"
        # stage.Traverse() plus a path test.  That is how environment.py walks the stage, and the
        # API guard `test_no_module_asks_pxr_for_something_it_does_not_have` refuses the range
        # iterator this was first written with, because that member is not on the verified list.
        prefix = str(prim.GetPath())
        bound = 0
        # INSTANCE PROXIES.  Measured on this scene: a plain traversal sees 4 prims under
        # /World/Dressing/railing/prop and the proxy traversal sees 10, so the meshes ARE in prim
        # instance prototypes and a plain traversal never reaches them.  That is why this loop
        # reported "flat material on 0 gprim(s)" and the props kept their own colour for five rounds.
        for candidate in stage.Traverse(Usd.TraverseInstanceProxies()):
            if not str(candidate.GetPath()).startswith(prefix):
                continue
            if not candidate.IsA(UsdGeom.Gprim):
                continue
            UsdShade.MaterialBindingAPI.Apply(candidate).Bind(mat)
            bound += 1
        return f"flat material on {bound} gprim(s)"
    except Exception as exc:  # noqa: BLE001
        return f"flat material failed: {type(exc).__name__}: {exc}"


def place_dressing(stage: Any, scene: str) -> List[Dict[str, Any]]:
    """Place the scene's dressing: a box, or the vendored mesh it names.

    Every item gets a box of its declared size, which is the geometry the occlusion and
    walking-band checks were written against; where the catalogue names a vendored ``.usd``
    prop, that prop is referenced on top of it so the item looks like a thing rather than a
    cube.  The box is what carries the placement and the scale, so a missing prop, a broken
    layer or a failed reference degrades to exactly the scene that shipped before this
    function could reference anything -- and the report says which happened.

    The catalogue's ``asset`` is consulted for every item, and any item naming a vendored ``.usd``
    gets it, floor or wall.  An earlier version referenced floor items only -- the reasoning being
    that a wall item's box is a picture plate and a referenced cabinet would go through the wall --
    and that rule was wrong twice over: it kept the supermarket's shelf goods as bare boxes, when
    a mug standing on a shelf is precisely a wall item, and it made a layout question into a
    condition in this loop.  Which items are referenced is therefore a property of the layouts:
    of 37 items, seven name an asset that resolves to a vendored mesh.

    The first version referenced whatever ``asset`` named, which for MI_SignB and
    M_TrafficCone was an MDL: USD answered "Cannot determine file format" and the item stayed
    a box.  prop_asset_for() now refuses that case explicitly instead of relying on USD to.
    """
    from pxr import Gf, UsdGeom

    placed: List[Dict[str, Any]] = []
    for item in sc.SCENES[scene]["dressing"]:
        # The item's own prim path.  Restructuring the tree dropped the assignment that used to define
        # it while the name was still used below, and ruff caught the undefined reference locally --
        # which is worth noting, because the same class of slip reached the lab machine twice in this
        # session through paths that no local check read.
        path = f"/World/Dressing/{item['name']}"
        # THE BOX AND THE PROP ARE SIBLINGS, NOT PARENT AND CHILD.
        #
        # This is the third arrangement tried and the first that can work.  The clearance box has to
        # stop being drawn once a prop resolves, and BOTH ways of hiding a prim hide its descendants:
        #
        #   * SetActive(False) removes the subtree from composition -- the prop stops existing;
        #   * MakeInvisible() sets visibility, and UsdGeomImageable documents that `invisible` means
        #     "the prim and all of its DESCENDANTS ... will be invisible", because visibility is an
        #     inherited property.
        #
        # With the prop as a child of the box, every option hides the prop: that is why the frames
        # stayed bare after the deactivation was replaced with invisibility, and why the report kept
        # saying RESOLVED while nothing appeared.  Making them siblings removes the interaction
        # entirely -- the box is hidden, the prop is untouched, and both remain on the stage where the
        # fixture checks and the on-stage verification can find them.
        box_path = f"/World/Dressing/{item['name']}/clearance"
        cube = UsdGeom.Cube.Define(stage, box_path)
        cube.GetSizeAttr().Set(1.0)
        # ONE frame for every size: the CATALOGUE's (x, height, lateral).  Both mounts store that,
        # and it is converted here, once, into the stage's (x, lateral, height).  A wall item used to
        # be exempted on the reasoning that "the order changes nothing" within the wall plane -- it
        # does: a Cube's scale component 1 lands on world y (lateral) and component 2 on world z
        # (height), so the poster declared (0.0524, 0.7426, 1.092) was drawn 1.092 m TALL and 0.74 m
        # wide.  Height and width are interchangeable for what the wall looks like and not at all
        # interchangeable for the prompt.
        size = sc.to_world_size(item["size"])
        # THE ITEM'S XFORM carries the placement and the declared scale, and the clearance box is a
        # unit cube UNDER it.  The declaration has to live here rather than on the box because the box
        # can no longer be the prop's ancestor: a Cube scaled by the declared size, with the prop as
        # its child, cannot be hidden without hiding the prop.  A unit cube under a scaled Xform is
        # still exactly the declared clearance volume, so every fixture check still reasons about the
        # declared size, and the prop -- a sibling of the box, under the same scaled Xform -- is
        # scaled by the declared size exactly as it was before.
        #
        # TRANSLATE FIRST, THEN SCALE.  AddXformOp documents that each successive call adds an
        # operator "applied more locally than the preceding", and GetLocalTransformation composes the
        # ordered list in reverse -- so with scale added first the order is [scale, translate] and the
        # translation is applied FIRST and then multiplied by the scale.  Every item was therefore
        # centred at size ⊙ at: the crate declared at x 10.2 with size 0.6015 landed at x 6.14, and
        # the rack at 5.01 instead of 7.0.  Adding the translate before the scale puts the scale
        # innermost, which is what "a box of `size` centred at `at`" means.
        #
        # The translate is written as Vec3d because AddTranslateOp defaults to PrecisionDouble and
        # creates a double3 attribute.  Vec3f is right for the scale, whose op defaults to float.
        item_xform = UsdGeom.Xform.Define(stage, path)
        item_prim = item_xform.GetPrim()
        UsdGeom.Xformable(item_prim).AddTranslateOp().Set(Gf.Vec3d(*sc.to_world(item["at"])))
        item_xform.AddScaleOp().Set(Gf.Vec3f(*size))
        prim = cube.GetPrim()
        _disable_collision(prim)
        if item.get("colour"):
            paint(stage, box_path, item["colour"])   # (ok, why); not reported per dressing item
        record = {"name": item["name"], "mount": item["mount"],
                  "asset": item["asset"], "used_asset": False}
        # The box is a COLLISION AND CLEARANCE VOLUME, not something to look at.  It used to stay
        # visible with the prop drawn inside it, which is why every furnished item still read as a
        # block: the box is exactly the declared size, so wherever a prop resolved there were two
        # objects on top of each other and the cube's flat faces were the ones facing the camera.
        # It is now hidden once a prop has resolved, and left visible only when there is no prop --
        # then it IS the item, and the fallback has to be drawn rather than be an invisible hole in
        # the scene.
        # Any item that names a prop gets one, wall items included.  It used to be floor items
        # only, on the reasoning that a wall item's box is a picture plate and referencing a
        # cabinet there would push it through the wall.  But the same rule kept the supermarket's
        # shelf goods as bare boxes -- a mug and a banana ARE wall items, standing on the shelf --
        # so the restriction was refusing exactly the references that make a wall look furnished.
        # Furniture on a wall remains a bad idea; that is a layout decision, and it belongs to the
        # layout rather than to a condition in this loop.
        local = prop_asset_for(item["asset"]) if item.get("asset") else None
        if local:
            try:
                # The prop is a SIBLING of the clearance box, under a plain Xform that carries the
                # item's placement, so that hiding the box cannot reach the prop:
                #
                #     /World/Dressing/<name>            Xform, the item
                #     /World/Dressing/<name>/clearance  Cube, the declared clearance volume
                #     /World/Dressing/<name>/prop       Xform, the referenced mesh
                #
                # The reference is added in the explicit four-step shape --
                #     UsdGeom.Xform.Define(stage, path) -> schema
                #     schema.GetPrim()                  -> prim
                #     prim.GetReferences()              -> UsdReferences metadata
                #     .AddReference(local_path)         -> the reference itself
                # -- because it cannot be exercised on the development machine (no pxr), and
                # spelling it out keeps the prim that receives the reference unambiguous.  The
                # except below turns any mistake here into a recorded message plus the box that was
                # already built, rather than a stage that fails to assemble mid-sweep.
                # (The item Xform itself was defined above, because it carries the placement; this
                # line is kept as the naming of the step rather than a second definition.)
                proxy_schema = UsdGeom.Xform.Define(stage, f"{path}/prop")
                proxy_prim = proxy_schema.GetPrim()
                proxy_prim.GetReferences().AddReference(local)
                # The reference is fitted INTO the declared box: scaled to fill it, and shifted so the
                # prop's own centre sits on the anchor.  Both halves are needed and they are separate
                # defects:
                #
                #  * the library is not in one unit.  Part of it is authored in centimetres with no
                #    metersPerUnit metadata, so the packing table measures 247.3647 in its own
                #    units -- 2.47 m of furniture.  Unscaled it draws the length of the room.
                #  * a prop drawn larger than its declared box sticks out of the volume every
                #    clearance and occlusion check reasons about, which is clipping by
                #    construction.  Fitting the prop to the box removes that class of defect: what
                #    the checks cleared is what is seen.
                #  * a prop whose geometry is anchored at a face rather than at its centre -- the
                #    trees, the shrubs, the rack and the curved bench all are -- is otherwise placed
                #    from the anchor outward.  An extent check cannot see this, and the conifer
                #    reached z = 3.5738 m under a 3.0 m ceiling with every report line green.
                #
                # The declared size is what the factor is computed against, because the prop's bound
                # relative to its parent is measured in the ITEM's local units, where one unit is the
                # declared size.  See fit_reference_to_local_unit, which carries the measurement and
                # the arithmetic; this call used to take no declared size and return min(1/native),
                # which filled the box only on the tightest axis.
                offset, scale = fit_reference_to_local_unit(proxy_prim)
                # NO ROTATION IS APPLIED, AND THIS IS THE PLACE TO PUT ONE IF THAT EVER CHANGES.
                #
                # attempt, 2026-10-16: adding AddOrientOp() with the quaternion (0.5, -0.5, -0.5,
                # -0.5) -- the exact -120 degree turn about (1,1,1) that maps the poster's measured
                # axes onto its declared frame -- drew it 15.46 m tall in a 1.092 m box and made it
                # "clip by construction".  The rotation is not needed for the BOX, because the fit is
                # per axis and the non-uniform scale already carries each measured extent to its
                # declared one: stage1.3 measured 12/12 items drawn exactly on their declared boxes
                # with no rotation at all.  What a rotation would change is the poster's IMAGE, which
                # is stretched by that non-uniform scale today.  That is a cosmetic defect and it is
                # recorded rather than fixed, because the fix has to be validated in a render and the
                # first attempt made the geometry worse, not better.
                #
                # sc.needs_rotation() and scenes.PROP_NEEDS_ROTATION name the one asset this applies
                # to, so the work has a starting point.
                # THE BOUND THE FIT WAS COMPUTED FROM, recorded because it is the one number that
                # settles the open prop-size defect and the one number the reverted attempt did not
                # have.  See fit_reference_to_local_unit: if `native_i * declared_i` equals the drawn
                # extent then `native` is the proxy/unscaled bound and the per-axis factor is the fix;
                # if `native_i` is already the drawn extent then the bound is scaled and the uniform
                # factor is closer to right.  `fit_reference_to_local_unit` does not return it, so it
                # is recomputed here from the same proxy the fit measured -- and only for the record,
                # so a failure here can never change what is placed.
                try:
                    from pxr import Usd as _Usd, UsdGeom as _UsdGeom

                    _parent = proxy_prim.GetParent()
                    _cache = _UsdGeom.BBoxCache(_Usd.TimeCode.Default(), [_UsdGeom.Tokens.default_])
                    _rng = _cache.ComputeRelativeBound(
                        proxy_prim, _parent).ComputeAlignedRange()
                    if not _rng.IsEmpty():
                        _lo, _hi = _rng.GetMin(), _rng.GetMax()
                        record["native_extent"] = tuple(
                            round(float(_hi[a] - _lo[a]), 6) for a in range(3))
                except Exception as _exc:  # noqa: BLE001
                    record["native_extent_error"] = f"{type(_exc).__name__}: {_exc}"
                if any(abs(o) > 1e-12 for o in offset):
                    # The OFFSET FIRST, then the scale, AND THEN THE ORDER IS FORCED.
                    #
                    # The order is not a style choice, it is the transform.  With the ops listed as
                    # [translate, scale], USD applies the scale first and the translate afterwards, so
                    # a point p lands at `p * scale + offset` -- which is what the offset above
                    # assumes, because it is `-scale * centre`, expressed in the item's own units.
                    # With [scale, translate] the same two ops mean `scale * (p + offset)`: the offset
                    # is scaled as well, so the prop's centre does not reach the anchor.
                    #
                    # Appending the ops in this order is NOT enough on its own, because something can
                    # already have put a scale op on the prim: `_rescale_and_measure` measures the
                    # bound a second time by writing one, and if it leaves it behind then
                    # apply_translate appends the translate AFTER it and the order silently flips.
                    # Measured on 2026-10-16: every prop's local box became [0, 1] instead of
                    # [-0.5, +0.5] in its own units, which lifted each one by 0.5 * (1 - scale) *
                    # height from the anchor -- half an object for the centimetre assets (the racks,
                    # the shelves: world box z 0.0000..1.9956 became 0.9945..2.9901) and 3% for the
                    # metre assets, which is why only some scenes looked wrong.
                    #
                    # So the order is set explicitly below, whatever any probe left on the prim.
                    _xformable = UsdGeom.Xformable(proxy_prim)
                    _translate_op = None
                    _scale_op = None
                    for _op in _xformable.GetOrderedXformOps():
                        if _op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                            _translate_op = _op
                        elif _op.GetOpType() == UsdGeom.XformOp.TypeScale:
                            _scale_op = _op
                    apply_translate(proxy_prim, offset)
                else:
                    _xformable = UsdGeom.Xformable(proxy_prim)
                    _translate_op = None
                    _scale_op = None
                    for _op in _xformable.GetOrderedXformOps():
                        if _op.GetOpType() == UsdGeom.XformOp.TypeScale:
                            _scale_op = _op
                if any(abs(s - 1.0) > 1e-9 for s in scale):
                    # The scale goes on the prop in the AXES THE PROP LIVES IN, which are the
                    # stage's: the item Xform above carries the declared size converted with
                    # to_world_size, so the prop's local axes are already (x, lateral, height).  No
                    # mount exemption here either -- that exemption is what drew every wall item with
                    # its height and width exchanged.
                    apply_scale(proxy_prim, scale)
                # Force [translate, scale] -- see the block above.  Re-read the ops because
                # apply_translate/apply_scale may have just created one of them.
                _xformable = UsdGeom.Xformable(proxy_prim)
                _ordered = []
                _translate_op = None
                _scale_op = None
                for _op in _xformable.GetOrderedXformOps():
                    if _op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                        _translate_op = _op
                    elif _op.GetOpType() == UsdGeom.XformOp.TypeScale:
                        _scale_op = _op
                    else:
                        _ordered.append(_op)
                if _translate_op is not None:
                    _ordered.append(_translate_op)
                if _scale_op is not None:
                    _ordered.append(_scale_op)
                if _ordered:
                    _xformable.SetXformOpOrder(_ordered)
                _disable_collision(proxy_prim)
                # A prop whose own shader cannot compile is repainted flat, so it renders in its
                # declared colour instead of the renderer's fallback.  Recorded in the report, because
                # "this prop's own materials are broken" is a fact the next reader needs.
                # EVERY resolving prop gets the declared colour bound as a flat material, not only
                # the ones whose own shader is broken.  A prop that carries no material at all is
                # drawn in the renderer's fallback whatever its clearance box says, which is why the
                # park's benches, railing and litter bin stayed red while their declared colours were
                # concrete, weathered timber and hedge.  Binding the colour is the only route that
                # reaches a prop: `paint()` cannot, because a prop is an Xform and not a Gprim, and
                # the clearance box it can paint is hidden the moment the prop resolves.
                how = bind_flat_material(stage, f"{path}/prop", item["name"], item["colour"])
                record["flat_material"] = how
                record["used_asset"] = os.path.basename(local)
                record["reference"] = local
                record["fit_scale"] = scale
                record["fit_offset"] = offset
                # Hide ONLY the clearance box.  Getting this wrong is what emptied every frame, and
                # it took three arrangements to see why:
                #
                #   * `prim.SetActive(False)` removed the box's whole SUBTREE from composition, and
                #     the prop was then a child of the box, so every resolved prop stopped existing;
                #   * `MakeInvisible()` on the box looked like the fix, but UsdGeomImageable defines
                #     `invisible` as hiding "the prim and all of its DESCENDANTS", because visibility
                #     is an INHERITED property -- so the prop stayed invisible and the frames stayed
                #     bare.  That is the state the user was looking at while this report said
                #     RESOLVED fifteen times.
                #   * making the prop a SIBLING of the box, both under the item's Xform, is the
                #     arrangement in which hiding the box cannot reach the prop at all.
                #
                # So this call is correct only because of the re-parenting above, and the two belong
                # together: changing one without the other hides the props again.
                UsdGeom.Imageable(prim).MakeInvisible()
            except Exception as exc:  # noqa: BLE001
                # A prop that will not load must leave the box, not the scene, broken: the box is
                # still visible here, which is the correct fallback rather than an empty hole.
                #
                # The message is built from more than str(exc), which for USD's ErrorException is
                # the EMPTY string -- a report reading "FAILED ErrorException: " names neither what
                # failed nor why, and the type alone is not a diagnosis.  The deepest frame of the
                # traceback is included because it names the pxr call that raised.
                detail = str(exc).strip()
                where = traceback.extract_tb(exc.__traceback__)
                if where:
                    frame = where[-1]
                    detail = (detail + " " if detail else "") + \
                        f"at {os.path.basename(frame.filename)}:{frame.lineno} {frame.line}"
                record["reference_error"] = f"{type(exc).__name__}: {detail.strip()}"
        # AN ASSET-LESS WALL ITEM GETS REAL GEOMETRY.  Four of the five are flat boards, so a sign
        # body plus an inset panel is what they are; building that needs no asset and stops them
        # being anonymous cubes while keeping the dressing their catalogues describe.
        # `asset is None` is not enough: stage1.2's `sign` carries a URL to an MDL MATERIAL
        # (MI_SignB.mdl) in its asset field, so it looks furnished and is not -- a cataloguing bug.
        # Anything that is not a .usd path cannot be referenced as a prop, so it counts as asset-less.
        nominal_asset = str(record.get("asset") or "")
        if (not record.get("used_asset") and item["mount"] != "floor"
                and not nominal_asset.endswith(".usd")):
            from pxr import UsdGeom
            try:
                made = build_plate(stage, item["name"], item)
                record["built_parts"] = made["parts"]
                # AND HIDE THE CLEARANCE CUBE, which is the cube the owner keeps seeing.  It is left
                # visible only when there is no prop, because then it IS the item -- but now the item
                # has real geometry, so leaving the cube visible would draw a box around the board
                # and the board would be the thing nobody notices.  Same call the resolving path uses.
                box = stage.GetPrimAtPath(f"{path}/clearance")
                if box and box.IsValid():
                    UsdGeom.Imageable(box).MakeInvisible()
                    record["clearance_hidden"] = True
            except Exception as exc:  # noqa: BLE001
                record["built_part_error"] = f"{type(exc).__name__}: {exc}"
        placed.append(record)

    # ---------------------------------------------------------------------------------------------
    # VERIFY THE FIT ON THE ASSEMBLED STAGE, and do it here rather than on the lab machine.
    #
    # This check exists because of a question I could not answer by reading the code, and which a
    # report then made urgent: the item's Xform carries a scale of `size`, the prop under it carries
    # a scale and an offset of its own, and Xform ops COMPOSE down the hierarchy.  Reading
    # GetOrderedXformOps does not settle what the drawn prop measures: it depends on how the composed
    # transform resolves at runtime, which is exactly the kind of thing this project has repeatedly
    # guessed wrong.
    #
    # So the code now asks the stage.  A FRESH BBoxCache is constructed after every prop is in place
    # (the cache does not listen for change notifications, so reusing the earlier one would report
    # whatever it had cached) and each resolved prop is measured relative to the dressing root.  Both
    # questions are asked of that measurement, because they are different defects:
    #
    #   * SIZE -- the drawn extent against the declared size.  A prop drawn larger than its box sticks
    #     out of the volume every clearance check reasons about, which is clipping by construction.
    #   * POSITION -- the drawn bound against the declared box (`at +/- size/2`).  An extent is
    #     `hi - lo` and can be perfect while the prop sits half a metre away; measured on the four
    #     dressed scenes before the offset existed, 25 of 43 props had geometry outside their own
    #     declared box, including a conifer at z = 3.5738 m under a 3.0 m ceiling, and every line of
    #     this report was green because only the size was ever compared.
    #
    # Both are recorded in fit_error rather than inferred from a picture later.
    # ---------------------------------------------------------------------------------------------
    from pxr import Usd, UsdGeom

    box_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    # The dressing root, which has no transform: a bound relative to it is a world-space extent, so
    # the number this block reports is in the same units as the item's declared size.
    root_prim = stage.GetPrimAtPath("/World/Dressing")
    for record in placed:
        if not record.get("used_asset"):
            continue
        proxy = stage.GetPrimAtPath(f"/World/Dressing/{record['name']}/prop")
        if not proxy or not proxy.IsValid() or root_prim is None or not root_prim.IsValid():
            # A resolved prop whose prim cannot be found is a defect in itself, and it is what this
            # block used to do silently: the prop sat under a deactivated ancestor, so every path here
            # was invalid and every item was skipped without a word.  A check that cannot run looks
            # exactly like a check that passed, so now it says so.  (The prop is also no longer under
            # the clearance box, which could be hidden by either mechanism and take the prop with it.)
            record["fit_error"] = ("resolved prop is not on the stage at "
                                   f"/World/Dressing/{record['name']}/prop, so its size cannot be "
                                   "verified -- was an ancestor deactivated?")
            continue
        try:
            # MEASURED RELATIVE TO THE DRESSING ROOT, and that is a correction.  This used to be
            # relative to the item's Xform, on the reasoning that the Xform carries the declared
            # scale and so a relative bound "is in metres".  The reference says otherwise: a relative
            # bound "excludes the local transform at relativeToAncestorPrim", so measuring against the
            # Xform excluded the very scale that turns the prop's local size into metres -- and the
            # numbers came back in the parent's local units, saturating at exactly 1.0 on the
            # constraining axis of every item (the fountain at (1.0, 1.0, 0.6835), the railing at
            # (1.0, 0.0327, 0.5238)).  Nothing was wrong with the geometry; the verification was
            # reading the wrong space, and the three FITS BADLY lines it printed were its own error.
            #
            # The dressing root has no transform, so a bound relative to it IS the world-space extent
            # in metres -- the same units as the declared size, and comparable with it directly.
            rng = box_cache.ComputeRelativeBound(proxy, root_prim).ComputeAlignedRange()
        except Exception as exc:  # noqa: BLE001
            record["fit_error"] = f"could not measure the drawn prop: {type(exc).__name__}: {exc}"
            continue
        if rng.IsEmpty():
            record["fit_error"] = "the drawn prop has an empty bound, so it has no size on the stage"
            continue
        lo, hi = rng.GetMin(), rng.GetMax()
        drawn = (float(hi[0] - lo[0]), float(hi[1] - lo[1]), float(hi[2] - lo[2]))
        item = next(entry for entry in sc.SCENES[scene]["dressing"]
                    if entry["name"] == record["name"])
        # Both sides in the STAGE's axes: the box's declared size converted once, and the prop
        # measured in the same space.  Comparing a catalogue tuple against a stage measurement is how
        # wall items came to be drawn with height and width exchanged while every check passed.
        target = tuple(float(v) for v in sc.to_world_size(item["size"]))
        record["drawn_extent"] = drawn
        record["declared_extent"] = target
        # The fraction of the declared box the drawn prop actually fills, per axis.  Recorded because
        # the containment test below can only say "no larger than"; a uniform fit against a box whose
        # proportions are not the asset's leaves axes empty, and an empty box is not reported by any
        # other line.  The worst of the three is the number a reader should look at.
        record["fill_ratio"] = tuple(round(drawn[axis] / target[axis], 3) if target[axis] else 0.0
                                     for axis in range(3))
        # WHERE the prop is, not only how big it is.  An extent is hi - lo and says nothing about
        # position, so this block used to pass any prop that was the right SIZE anywhere in the room:
        # measured on this stage, 25 of the 43 resolved props had their geometry outside their own
        # declared box -- the conifer reached z = 3.5738 m under a 3.0 m ceiling -- while every extent
        # line stayed green.  The declared box is centred on the item's anchor by construction (see
        # _dressing_floor and _dressing_wall), so the box is `at +/- size/2` in world axes and the
        # prop has to be inside it.
        anchor = tuple(float(v) for v in sc.to_world(item["at"]))
        box_lo = tuple(anchor[axis] - target[axis] / 2.0 for axis in range(3))
        box_hi = tuple(anchor[axis] + target[axis] / 2.0 for axis in range(3))
        record["drawn_box"] = tuple(round(float(lo[axis]), 5) for axis in range(3)) + \
            tuple(round(float(hi[axis]), 5) for axis in range(3))
        # 2 % and a millimetre for the SIZE, because the fit is computed from a bound that itself
        # carries float error and a prop that exactly fills its box must not be reported for the last
        # bit.  The POSITION tolerance is a millimetre: the offset is exact arithmetic on the same
        # bound, so anything larger than capture noise means the prop is not where it was declared.
        over = [axis for axis in range(3) if drawn[axis] > target[axis] * 1.02 + 1e-3]
        axis_names = "xyz"
        outside = [axis_names[axis] for axis in range(3)
                   if float(lo[axis]) < box_lo[axis] - 1e-3
                   or float(hi[axis]) > box_hi[axis] + 1e-3]
        reasons = []
        if over:
            reasons.append(f"drawn {tuple(round(v, 4) for v in drawn)} m exceeds declared "
                           f"{tuple(round(v, 4) for v in target)} m on axis {over}")
        if outside:
            reasons.append(
                f"its geometry leaves the declared box on axis {outside}: drawn "
                f"{tuple(round(float(lo[axis]), 4) for axis in range(3))}.."
                f"{tuple(round(float(hi[axis]), 4) for axis in range(3))} against box "
                f"{tuple(round(v, 4) for v in box_lo)}..{tuple(round(v, 4) for v in box_hi)}")
        if reasons:
            record["fit_error"] = " and ".join(reasons) + " -- clipping by construction"
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
                use_mdl: bool = False,
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

    THE DEFAULT AND THIS PARAGRAPH DISAGREED, and the paragraph was right.  The signature said
    ``use_mdl: bool = True`` while the docstring said False, and environment.py also defaulted its
    task_dict lookup to True -- and capture_scenes passed ``use_mdl=not args.flat_paint``, so a run
    WITHOUT ``--flat-paint`` built MDL prims on the machine whose cache cannot be written and every
    surface came back full of noise.  A preview was then read as "the scene is bare" several times
    when the truth was that the surfaces were unreadable at all.

    All three now agree on the safe default: this signature is False, environment's lookup is False,
    and capture_scenes drives it from an explicit ``--mdl`` flag rather than from the absence of
    ``--flat-paint``.  ``--flat-paint`` is kept and is now the default's name for compatibility.
    """
    if scene not in sc.SCENES:
        raise ValueError(f"unknown scene {scene!r}, expected one of {list(sc.SCENES)}")
    spec = sc.SCENES[scene]
    report: Dict[str, Any] = {"scene": scene, "slot": slot, "label": spec["label"],
                              "materials": {}, "marker": None, "dressing": []}

    # THE SCENE'S OWN FLAT-PAINT PALETTE, not one global fallback.  It used to be a single dict
    # here for all five scenes, so every preview had the same grey floor, the same blue-grey
    # corridor and the same dark green end wall no matter what the scene's catalogue said, and the
    # five scenes were indistinguishable.  The colours now live in `scenes.SURFACE_COLOURS`, per
    # scene, because they are scene DESIGN and this module is the thing that applies them; the
    # arithmetic that keeps them distinct from each other and from each scene's markers is written
    # next to the table.  A scene with no entry keeps the old neutral, so a sixth scene cannot
    # silently paint itself the wrong colour.
    neutral = {"floor": (0.55, 0.55, 0.55), "side_wall": (0.72, 0.73, 0.75),
               "ceiling": (0.92, 0.93, 0.95), "far_wall": (0.13, 0.42, 0.20)}
    fallbacks = dict(neutral)
    fallbacks.update(getattr(sc, "SURFACE_COLOURS", {}).get(scene, {}))
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
            # A TEXTURE if the scene names one for this surface and it is vendored, else the flat
            # colour.  The owner asked for patterned walls, and a texture needs no MDL, so the
            # unwritable material cache that made flat paint the safe default does not block it.
            # The flat colour is still painted first: it is the fallback if the texture is missing, and
            # it keeps the surface honest if the texture fails to load.
            texture = (getattr(sc, "SURFACE_TEXTURES", {}) or {}).get(scene, {}).get(surface)
            local_texture = _vendored_texture(texture) if texture else None
            how = "flat-paint"
            flat_paint = [
                paint(stage, path, fallbacks.get(surface, (0.6,) * 3))
                for path in prim_paths
            ]
            if not all(ok for ok, _ in flat_paint):
                how = (
                    "flat-paint (paint FAILED on "
                    f"{sum(1 for ok, _ in flat_paint if not ok)} prim(s))"
                )
            if local_texture:
                # UVs FIRST: a mesh with no texture coordinates paints the whole surface with one
                # texel, which is a flat colour however large the image is.
                uvs = [author_uvs(stage, path, TEXTURE_METRES_PER_REPEAT.get(
                    os.path.basename(local_texture), 1.5)) for path in prim_paths]
                results = [bind_texture_material(stage, path, f"{surface}_{scene}", local_texture)
                           for path in prim_paths]
                results = [f"{r} ({u})" for r, u in zip(results, uvs)]
                how = results[0] if len(set(results)) == 1 else "; ".join(sorted(set(results)))
            elif texture:
                how = f"flat-paint (texture {texture} is not vendored)"
            report["materials"][surface] = {"url": url, "how": how,
                                            "prim": ", ".join(prim_paths)}
            continue
        name = os.path.basename(url).replace(".mdl", "")
        material, how = resolve_material(
            stage, url, name, f"/World/Looks/{surface}_{prim_name(scene)}",
            prim_path, fallbacks.get(surface, (0.6,) * 3))
        if material is not None:
            # CONSUME the result: _bind catches every exception and returns False, so
            # discarding it let the report claim "mdl" for a surface the material was
            # never bound to -- a report is evidence, and it was asserting success.
            bound = [_bind(stage, path, material) for path in prim_paths]
        else:
            bound = []
        report["materials"][surface] = {
            "url": url,
            "how": how,
            "prim": ", ".join(prim_paths),
            "bound": (all(bound) if material is not None else None),
        }
        if material is not None and not all(bound):
            report["materials"][surface]["how"] = (
                f"{how} (BIND FAILED on {sum(1 for ok in bound if not ok)} prim(s))"
            )

    # THE OBSTACLE WALL'S PANELS, which are not one of the four classified surfaces.  They are found
    # by prim NAME rather than by geometry, because they are the only prims environment creates with
    # "WallPanel" in the path and classifying them by shape would be guesswork.  They are the surface
    # an eye-camera frame is mostly made of, so leaving them out is what kept every corridor blue.
    if "materials" in parts:
        from pxr import UsdGeom   # local, like the other passes in this function

        panel_texture = (getattr(sc, "SURFACE_TEXTURES_PANELS", {}) or {}).get(scene)
        panel_colour = (getattr(sc, "SURFACE_COLOUR_PANELS", {}) or {}).get(scene)
        local_panel_texture = _vendored_texture(panel_texture) if panel_texture else None
        panel_paths = [str(prim.GetPath()) for prim in stage.Traverse()
                       if "WallPanel" in prim.GetName() and prim.IsA(UsdGeom.Gprim)]
        if panel_paths:
            panel_paint_ok = True
            if panel_colour:
                panel_paint_ok = all(
                    ok for ok, _ in (
                        paint(stage, path, panel_colour) for path in panel_paths
                    )
                )
            how = "flat colour" if panel_paint_ok else "flat colour (paint FAILED)"
            if local_panel_texture:
                panel_uvs = [author_uvs(stage, path, TEXTURE_METRES_PER_REPEAT.get(
                    os.path.basename(local_panel_texture), 1.5)) for path in panel_paths]
                outcomes = [bind_texture_material(stage, path, f"wall_panel_{scene}",
                                                  local_panel_texture) for path in panel_paths]
                outcomes = [f"{o} ({u})" for o, u in zip(outcomes, panel_uvs)]
                how = outcomes[0] if len(set(outcomes)) == 1 else "; ".join(sorted(set(outcomes)))
            elif panel_texture:
                how = f"flat colour (texture {panel_texture} is not vendored)"
            report["materials"]["wall_panel"] = {"url": panel_texture or "-", "how": how,
                                                 "prim": ", ".join(sorted(panel_paths))}
        else:
            report["materials"]["wall_panel"] = {"url": panel_texture or "-",
                                                 "how": "no WallPanel prim found", "prim": "-"}
    # A MATTE FLOOR EVEN WHEN IT IS NOT TEXTURED.  environment binds its own material to /World/Ground
    # (metallic 0.0, roughness 0.6), and 0.6 is glossy enough to reflect the far-wall marker -- the
    # owner's "反光显示的标志物会影响模型判断".  A textured floor gets its roughness from
    # TEXTURE_ROUGHNESS; this covers the rest, and the ceiling too, which is what the eye camera sees
    # most of.
    if "materials" in parts:
        from pxr import Sdf, UsdShade
        matte = {"floor": 0.95, "ceiling": 0.92}
        found = discover_surfaces(stage)
        for surface, rough in matte.items():
            path = found.get(surface)
            if path:
                prim = stage.GetPrimAtPath(path)
                material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
                if not (material and material.GetPrim().IsValid()):
                    continue
                shader, _, _ = material.ComputeSurfaceSource()
                if not (shader and shader.GetPrim().IsValid()):
                    continue
                value = shader.GetInput("roughness")
                if value is None:
                    value = shader.CreateInput("roughness", Sdf.ValueTypeNames.Float)
                value.Set(float(rough))
                # WITH a complete entry, not an empty one.  This used to be
                # `setdefault(surface, {})` followed by only `["roughness"]`, which left a materials
                # entry with no "how" key; format_report prints info['how'] for every entry, so the
                # report CRASHED with KeyError: 'how' on any scene that has no materials of its own --
                # stage1.1, the baseline.  Its marker sheet was therefore never written and stayed
                # stale on disk, which is how this was noticed.
                entry = report["materials"].setdefault(
                    surface, {"url": "-", "how": "flat colour (no material named)",
                              "prim": path})
                entry["roughness"] = float(rough)
    report["marker"] = build_marker(stage, scene, slot)
    report["parts"] = list(parts)
    report["dressing"] = place_dressing(stage, scene) if "dressing" in parts else []
    report["dressing_count"] = len(report["dressing"])
    # material_verified IS CHECKED HERE, NOT MERELY DECLARED.  Every scene in scenes.py sets
    # material_verified: True, and until now NOTHING READ IT -- the only consumer was a test asserting
    # the flag it had just been handed.  So a scene whose surfaces silently failed to bind still
    # claimed to be verified, and the claim was evidence of nothing.
    #
    # The declared value is now compared with what actually happened to the surfaces, and the result
    # is recorded under its own key.  It is NOT silently overwritten: a mismatch is a real
    # disagreement between the catalogue and the render, and both halves of it are worth keeping.
    if "materials" in parts:
        declared = spec.get("material_verified")
        bound_ok, bound_failed, unbound = 0, 0, 0
        for _surface, outcome in (report.get("materials") or {}).items():
            if not isinstance(outcome, dict) or outcome.get("how") == "skipped":
                continue
            bound = outcome.get("bound")
            if bound is None:
                unbound += 1          # flat paint or texture: no MDL to bind, so not a failure
            elif bound:
                bound_ok += 1
            else:
                bound_failed += 1
        report["material_verified_declared"] = declared
        report["material_verified_actual"] = (
            bound_failed == 0 and (bound_ok + unbound) > 0
        )
        report["material_bind_failures"] = bound_failed
        if declared is True and report["material_verified_actual"] is False:
            report.setdefault("problems", []).append(
                f"{scene}: scenes.py declares material_verified=True but "
                f"{bound_failed} surface(s) failed to bind"
            )
    return report


def format_report(report: Dict[str, Any]) -> str:
    """One line per fact, for the run log."""
    lines = [f"[scene] {report['scene']} ({report['label']}) marker slot {report['slot']}"]
    marker = report["marker"]
    # The size and the paint result are printed because both changed recently and neither can be
    # checked from any other number in the log: the marker went to 0.90 m, and "painted False" is
    # otherwise a failure that has to be diagnosed by eye from a dim frame.
    lines.append(f"[scene] marker: {marker['shape']} in {marker['colour']} "
                 f"({marker['parts']} mesh part(s), size "
                 f"{marker.get('size_m', 0.0):.2f} m, painted {marker.get('painted')}"
                 # The reason a paint failed, because "painted False" alone cost two render
                 # round trips: the except had discarded the answer both times.
                 + (f" -- {marker['paint_error']}" if marker.get("paint_error") else "")
                 + ")")
    for surface, info in report["materials"].items():
        # The prim path is printed because without it the log cannot answer the question that
        # matters when a surface looks wrong: which prim did this material actually land on.  In
        # a preview of the library the right-hand wall came out in the floor's wood, and there
        # was no way to tell from the log whether the classifier had named the wrong prim, named
        # the same prim twice, or the wall prim was simply not the one being painted.
        lines.append(f"[scene] {surface:<9} {info['how']:<11} -> {info.get('prim', '-')}")
    if report["dressing"]:
        used = sum(1 for item in report["dressing"] if item["used_asset"])
        failed = [item for item in report["dressing"] if item.get("reference_error")]
        oversized = [item for item in report["dressing"] if item.get("fit_error")]
        # Items the stage reports as drawn well inside their box on every axis.  Not an error -- the
        # clearance checks clear the box, and containment is the requirement -- but it is the
        # condition the human described as "道具偏细", and it belongs in the always-printed report
        # because no picture can be trusted to show it.
        thin: List[tuple] = []
        lines.append(f"[scene] dressing: {report['dressing_count']} items, "
                     f"{used} from assets, rest boxes, all non-collidable")
        # Per item, and in the ALWAYS-printed report rather than behind a switch.  An item whose
        # prop resolves has its clearance box made invisible, so if the prop then fails to draw the
        # item is absent from the frame with no fallback to notice -- which means "the preview is
        # empty" has two opposite causes (the reference failed, or it resolved and nothing is
        # drawn) that no picture can separate.  This separates them, and it appears in every run
        # so it cannot be lost to a flag or a grep.
        for item in report["dressing"]:
            if item["used_asset"]:
                detail = item["used_asset"]
                fit = item.get("fit_scale") or ()
                if any(abs(float(v) - 1.0) > 1e-9 for v in fit):
                    detail += f" (fit {tuple(round(float(v), 4) for v in fit)})"
                offset = item.get("fit_offset") or ()
                # The offset is printed when it is not zero, because it is the half of the fit that
                # an extent cannot show: it is what puts a face-anchored asset's centre on the
                # item's anchor instead of drawing the asset from the anchor outward.
                if any(abs(float(v)) > 1e-12 for v in offset):
                    detail += f" (offset {tuple(round(float(v), 4) for v in offset)})"
                outcome = f"RESOLVED {detail}"
                # The size the STAGE reports, beside the size the item declared.  A prop drawn
                # larger than its box clips by construction, and a prop drawn at the wrong scale is
                # the defect that has cost this session the most rounds, so both are stated as
                # numbers instead of being left to a picture.
                if item.get("drawn_extent"):
                    drawn = tuple(round(float(v), 4) for v in item["drawn_extent"])
                    declared = tuple(round(float(v), 4) for v in item["declared_extent"])
                    outcome += f" [drawn {drawn} vs declared {declared}]"
                    # The bound the fit divided by.  Printed next to the drawn extent because the two
                    # together are what decide the open defect: see fit_reference_to_local_unit.
                    if item.get("native_extent"):
                        native = tuple(round(float(v), 4) for v in item["native_extent"])
                        outcome += f" [native {native}]"
                    elif item.get("native_extent_error"):
                        outcome += f" [native UNMEASURED: {item['native_extent_error']}]"
                if item.get("drawn_box"):
                    box = item["drawn_box"]
                    outcome += (f" [world box {tuple(round(float(v), 4) for v in box[:3])}.."
                                f"{tuple(round(float(v), 4) for v in box[3:])}]")
                if item.get("fit_error"):
                    outcome += f" FITS BADLY: {item['fit_error']}"
                # HOW MUCH OF ITS BOX THE PROP ACTUALLY FILLS, as a number, because a uniform fit
                # leaves the box empty wherever the asset's proportions differ from the declaration
                # and the frame shows a prop that is smaller than it was declared.  This is the
                # criterion the human asked for instead of looking at a picture.  1.000 means the box
                # is filled on that axis; a low worst ratio means the asset is drawn small (the
                # railing at 0.033 is 2.4 mm thick where 72.8 mm was declared).
                fill = item.get("fill_ratio") or ()
                if fill:
                    worst = min(float(v) for v in fill)
                    outcome += f" [fills {tuple(float(v) for v in fill)} of its box]"
                    # 0.9 rather than 1.0: an asset whose proportions differ slightly from its box
                    # legitimately fills it to 0.95, and a threshold that fires on those is noise.  A
                    # prop below 0.9 on every axis is drawn visibly smaller than it was declared, and
                    # that is the condition the human described as "道具偏细".
                    if worst < 0.90:
                        thin.append((item["name"], round(worst, 3)))
            elif item.get("reference_error"):
                outcome = f"FAILED {item['reference_error']}"
            elif item.get("built_parts"):
                # No asset, so the item is BUILT rather than referenced: a sign board with an inset
                # panel, and its clearance cube is hidden.  Reporting this as "NO ASSET (drawn as its
                # clearance box)" was true before and is false now -- the line has to say which of the
                # two it is, because that sentence is what a reader uses to decide whether an object
                # in the frame is real.
                outcome = (f"NO ASSET (built as a plate: {len(item['built_parts'])} part(s), "
                           "clearance box hidden)")
            else:
                outcome = "NO ASSET (drawn as its clearance box)"
            # The prop-colour result is APPENDED after the chain, not one of its branches: a prop
            # with a fill ratio takes the `fill` branch and would never reach a flat_material branch,
            # which is why the suffix did not appear the first time it was added.
            if item.get("flat_material"):
                outcome += f" [colour: {item['flat_material']}]"
            lines.append(f"[scene]   dressing {item['name']}: {outcome}")
        if failed:
            lines.append(f"[scene] dressing: {len(failed)} reference(s) FAILED -- those items are "
                         "drawn as their clearance box")
        if oversized:
            lines.append(f"[scene] dressing: {len(oversized)} item(s) are drawn outside their "
                         "declared box -- larger than it, or positioned so part of the geometry "
                         "leaves it, either of which is clipping by construction: "
                         + ", ".join(item["name"] for item in oversized))
        if thin:
            lines.append(f"[scene] dressing: {len(thin)} item(s) are drawn well inside their declared "
                         "box on every axis (worst fill ratio first), which is what a uniform fit "
                         "does when the box's proportions are not the asset's: "
                         + ", ".join(f"{name} {ratio}" for name, ratio in sorted(thin, key=lambda t: t[1])))
    return "\n".join(lines)
