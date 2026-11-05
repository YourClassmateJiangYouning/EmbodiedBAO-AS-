from __future__ import annotations
import os
from isaacsim import SimulationApp
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
app=SimulationApp({'headless':True})
import omni.kit.commands
urdf=os.path.join(ROOT,'assets','H1_urdf','h1.urdf'); usd=os.path.join(ROOT,'assets','H1','h1_urdf.usd')
status,cfg=omni.kit.commands.execute('URDFCreateImportConfig')
cfg.merge_fixed_joints=False; cfg.convex_decomp=False; cfg.import_inertia_tensor=True; cfg.fix_base=False; cfg.distance_scale=1.0
print('import',urdf,'->',usd,flush=True)
st,prim=omni.kit.commands.execute('URDFParseAndImportFile',urdf_path=urdf,import_config=cfg,dest_path=usd,get_articulation_root=True)
print('status',st,'prim',prim,'exists',os.path.exists(usd),flush=True)
