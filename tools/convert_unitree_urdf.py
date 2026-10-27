"""Convert the downloaded Unitree G1/R1 URDFs into Isaac Sim USD assets.

Run with /home/ybh/isaacsim/python.sh tools/convert_unitree_urdf.py.
This is offline asset preparation, not an experiment.
"""
from __future__ import annotations
import os
import sys
from isaacsim import SimulationApp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAIRS = [
    ("G1", os.path.join(ROOT, "assets", "G1", "g1_23dof.urdf"), os.path.join(ROOT, "assets", "G1", "g1.usd")),
    ("R1", os.path.join(ROOT, "assets", "R1", "R1.urdf"), os.path.join(ROOT, "assets", "R1", "r1.usd")),
]

app = SimulationApp({"headless": True})
import omni.kit.commands
from pxr import Usd, UsdGeom

for name, urdf, usd_path in PAIRS:
    if not os.path.exists(urdf):
        print(f"[{name}] missing URDF {urdf}", flush=True)
        continue
    os.makedirs(os.path.dirname(usd_path), exist_ok=True)
    status, cfg = omni.kit.commands.execute("URDFCreateImportConfig")
    cfg.merge_fixed_joints = False
    cfg.convex_decomp = False
    cfg.import_inertia_tensor = True
    cfg.fix_base = False
    cfg.distance_scale = 1.0
    print(f"[{name}] importing {urdf} -> {usd_path}", flush=True)
    status, prim_path = omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=urdf,
        import_config=cfg,
        dest_path=usd_path,
        get_articulation_root=True,
    )
    print(f"[{name}] status={status} prim_path={prim_path} usd_exists={os.path.exists(usd_path)}", flush=True)
    if os.path.exists(usd_path):
        stage = Usd.Stage.Open(usd_path)
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        lo = [float("inf")] * 3
        hi = [float("-inf")] * 3
        for p in stage.Traverse():
            b = cache.ComputeWorldBound(p).ComputeAlignedRange()
            a, z = b.GetMin(), b.GetMax()
            if all(abs(v) < 1e12 for v in (a[0], a[1], a[2], z[0], z[1], z[2])):
                for i in range(3):
                    lo[i] = min(lo[i], a[i]); hi[i] = max(hi[i], z[i])
        print(f"[{name}] bbox min={lo} max={hi} size={[hi[i]-lo[i] for i in range(3)]} up_axis={UsdGeom.GetStageUpAxis(stage)}", flush=True)

app.close()
