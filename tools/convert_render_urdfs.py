from __future__ import annotations
import os
from isaacsim import SimulationApp
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
app=SimulationApp({'headless':True})
import omni.kit.commands
pairs=[('H1','assets/H1_urdf/render_h1.urdf','assets/H1/render_h1.usd'),('G1','assets/G1/render_g1.urdf','assets/G1/render_g1.usd'),('R1','assets/R1/render_r1.urdf','assets/R1/render_r1.usd')]
for name,rel,out in pairs:
 urdf=os.path.join(ROOT,rel); usd=os.path.join(ROOT,out)
 st,cfg=omni.kit.commands.execute('URDFCreateImportConfig'); cfg.merge_fixed_joints=False; cfg.convex_decomp=False; cfg.import_inertia_tensor=True; cfg.fix_base=False; cfg.distance_scale=1.0
 print('convert',name,flush=True)
 st,prim=omni.kit.commands.execute('URDFParseAndImportFile',urdf_path=urdf,import_config=cfg,dest_path=usd,get_articulation_root=True)
 print('done',name,st,prim,os.path.exists(usd),flush=True)
