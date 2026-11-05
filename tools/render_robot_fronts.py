"""Render H1, G1 and R1 individually from the same frontal camera, then combine.

Each robot is rendered alone (others hidden), under identical dome/key/fill
lighting and with the same camera.  The background alpha is produced by a second
render with all robot bodies hidden.  Output:
  reports/robot_front_render/H1_front.png
  reports/robot_front_render/G1_front.png
  reports/robot_front_render/R1_front.png
  reports/robot_front_render/robot_fronts_H1_G1_R1.png  (combined, RGBA)
"""
from __future__ import annotations
import math, os
from isaacsim import SimulationApp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "reports", "robot_front_render")
os.makedirs(OUT, exist_ok=True)
app = SimulationApp({"headless": True, "renderer": "RaytracedLighting"})
import numpy as np
import omni.usd
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade
from isaacsim.core.api import World
from isaacsim.core.prims import XFormPrim
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.sensors.camera import Camera
from PIL import Image, ImageDraw, ImageFont

stage = omni.usd.get_context().get_stage()
world = World(stage_units_in_meters=1.0)
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)

def look_at_quaternion(eye, target, up=(0.0, 0.0, 1.0)):
    eye = np.asarray(eye, float); target = np.asarray(target, float)
    f = target - eye; f /= np.linalg.norm(f)
    u = np.asarray(up, float); u /= np.linalg.norm(u)
    l = np.cross(u, f); l /= np.linalg.norm(l)
    u2 = np.cross(f, l); u2 /= np.linalg.norm(u2)
    m = np.column_stack((f, l, u2))
    tr = m[0,0] + m[1,1] + m[2,2]
    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2; w = 0.25*S
        x = (m[2,1]-m[1,2])/S; y = (m[0,2]-m[2,0])/S; z = (m[1,0]-m[0,1])/S
    elif m[0,0] > m[1,1] and m[0,0] > m[2,2]:
        S = math.sqrt(1.0+m[0,0]-m[1,1]-m[2,2])*2
        w = (m[2,1]-m[1,2])/S; x = 0.25*S
        y = (m[0,1]+m[1,0])/S; z = (m[0,2]+m[2,0])/S
    elif m[1,1] > m[2,2]:
        S = math.sqrt(1.0+m[1,1]-m[0,0]-m[2,2])*2
        w = (m[0,2]-m[2,0])/S; x = (m[0,1]+m[1,0])/S
        y = 0.25*S; z = (m[1,2]+m[2,1])/S
    else:
        S = math.sqrt(1.0+m[2,2]-m[0,0]-m[1,1])*2
        w = (m[1,0]-m[0,1])/S; x = (m[0,2]+m[2,0])/S
        y = (m[1,2]+m[2,1])/S; z = 0.25*S
    return np.array([w,x,y,z], float)

def bbox_z(path):
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    lo = float("inf"); hi = float("-inf")
    for p in stage.Traverse():
        if not str(p.GetPath()).startswith(path): continue
        b = cache.ComputeWorldBound(p).ComputeAlignedRange(); a,z = b.GetMin(), b.GetMax()
        if not all(abs(v) < 1e12 for v in [a[0],a[1],a[2],z[0],z[1],z[2]]): continue
        lo = min(lo, a[2]); hi = max(hi, z[2])
    return lo, hi

specs = [
    ("H1", os.path.join(ROOT, "assets/H1/h1.usd")),
    ("G1", os.path.join(ROOT, "assets/G1/configuration/g1_base.usd")),
    ("R1", os.path.join(ROOT, "assets/R1/configuration/r1_base.usd")),
]
paths = {}
for name, path in specs:
    prim = f"/World/{name}"
    add_reference_to_stage(usd_path=path, prim_path=prim)
    paths[name] = prim
    lo, hi = bbox_z(prim)
    lift = -lo if lo != float("inf") else 0.0
    print(f"[render] {name} bbox z [{lo:.3f},{hi:.3f}] lift={lift:.3f}", flush=True)
    XFormPrim(prim_paths_expr=prim).set_world_poses(
        positions=np.array([[0.0, 0.0, lift]]),
        orientations=np.array([[1.0, 0.0, 0.0, 0.0]]),
    )

# Lights: identical for every isolated render.
dome = UsdLux.DomeLight.Define(stage, "/World/Dome"); dome.GetIntensityAttr().Set(900.0)
for p, loc, rot, intensity in [
    ("/World/Key", (3.0, -3.0, 4.0), (50.0, 0.0, 20.0), 2500.0),
    ("/World/Fill", (3.0, 3.0, 2.5), (60.0, 0.0, -20.0), 900.0),
]:
    L = UsdLux.RectLight.Define(stage, p); L.GetIntensityAttr().Set(intensity)
    L.GetWidthAttr().Set(4.0); L.GetHeightAttr().Set(4.0)
    api = UsdGeom.XformCommonAPI(L)
    api.SetTranslate(Gf.Vec3d(*loc)); api.SetRotate(Gf.Vec3f(*rot))

# Chroma backdrop behind the robots.  It gives the transparent alpha a
# well-defined colour even for the white H1 body, which the "background minus
# robots" difference could not separate from the white renderer background.
backdrop = UsdGeom.Cube.Define(stage, "/World/Backdrop")
backdrop.GetSizeAttr().Set(2.0)
UsdGeom.XformCommonAPI(backdrop).SetTranslate(Gf.Vec3d(-3.0, 0.0, 1.0))
UsdGeom.XformCommonAPI(backdrop).SetScale(Gf.Vec3f(0.02, 4.0, 4.0))
bmat = UsdShade.Material.Define(stage, "/World/BackdropMat")
bsh = UsdShade.Shader.Define(stage, "/World/BackdropMat/Shader")
bsh.CreateIdAttr("UsdPreviewSurface")
bsh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0, 0.0, 1.0))
bsh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0, 0.0, 1.0))
bsh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
bmat.CreateSurfaceOutput().ConnectToSource(bsh.ConnectableAPI(), "surface")
UsdShade.MaterialBindingAPI(backdrop).Bind(bmat)

world.play(); world.reset()
camera = Camera(prim_path="/World/Camera", frequency=30, resolution=(1200, 1200))
camera.initialize()
camera.set_focal_length(35.0)
eye = np.array([10.0, 0.0, 1.00], float)
target = np.array([0.0, 0.0, 0.95], float)
quat = look_at_quaternion(eye, target)
camera.set_world_pose(position=eye.tolist(), orientation=quat.tolist(), camera_axes="world")
for _ in range(80):
    world.step(render=True); app.update()

ROBOT_ROOTS = [stage.GetPrimAtPath(f"/World/{n}") for n in ("H1", "G1", "R1")]

def set_visible(name):
    # Toggle only the three root references.  Toggling every descendant while
    # iterating the stage makes a parent inactive before its children are
    # visited and segfaults the USD stage (measured in the first attempt).
    for root in ROBOT_ROOTS:
        try:
            root.SetActive(str(root.GetPath()) == f"/World/{name}")
        except Exception:
            pass

def render_rgba():
    for _ in range(30):
        world.step(render=True); app.update()
    a = np.asarray(camera.get_rgba(), dtype=np.uint8)
    if a.ndim == 1:
        a = a.reshape(1200, 1200, 4)
    return a

images = {}
MAGENTA = np.array([255.0, 0.0, 255.0], dtype=np.float32)
for name in ("H1", "G1", "R1"):
    set_visible(name)
    front = render_rgba()
    d = np.linalg.norm(front[:,:,:3].astype(np.float32) - MAGENTA, axis=2)
    front[:,:,3] = np.where(d > 18.0, 255, 0).astype(np.uint8)
    path = os.path.join(OUT, f"{name}_front.png")
    Image.fromarray(front, "RGBA").save(path)
    images[name] = Image.fromarray(front, "RGBA")
    print(f"[render] {name}: alpha nonzero={int((front[:,:,3]>0).sum())} -> {path}", flush=True)
# Trim to alpha bounds, preserve identical scale, align feet, and combine.
def trim(im):
    a = np.asarray(im)[:,:,3]
    ys, xs = np.where(a > 0)
    if len(xs) == 0: return im
    return im.crop((int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1))
trimmed = {k: trim(v) for k, v in images.items()}
max_h = max(v.height for v in trimmed.values())
pad = 60
canvas_w = sum(v.width for v in trimmed.values()) + pad * 4
canvas_h = max_h + pad * 3 + 80
canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
draw = ImageDraw.Draw(canvas)
try: font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 54)
except Exception: font = ImageFont.load_default()
x = pad
for name in ("H1", "G1", "R1"):
    im = trimmed[name]
    y = pad + (max_h - im.height)
    canvas.alpha_composite(im, (x, y))
    draw.text((x + 10, y + 10), name, fill=(30,30,30,255), font=font)
    x += im.width + pad
combined = os.path.join(OUT, "robot_fronts_H1_G1_R1.png")
canvas.save(combined)
print("[render] wrote", combined, canvas.size, flush=True)
