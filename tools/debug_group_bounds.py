from __future__ import annotations
import os
from isaacsim import SimulationApp
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
app=SimulationApp({'headless':True})
from pxr import Gf,Usd,UsdGeom
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
stage=omni.usd.get_context().get_stage(); world=World(stage_units_in_meters=1.0)
UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z)
def bounds(path):
 c=UsdGeom.BBoxCache(Usd.TimeCode.Default(),[UsdGeom.Tokens.default_]); lo=[1e30]*3; hi=[-1e30]*3
 for p in stage.Traverse():
  if not str(p.GetPath()).startswith(path): continue
  b=c.ComputeWorldBound(p).ComputeAlignedRange(); a,z=b.GetMin(),b.GetMax()
  if not all(abs(v)<1e12 for v in [a[0],a[1],a[2],z[0],z[1],z[2]]): continue
  for i in range(3): lo[i]=min(lo[i],a[i]); hi[i]=max(hi[i],z[i])
 return lo,hi
for name,path,z in [('H1',os.path.join(ROOT,'assets/H1/configuration/h1_urdf_base.usd'),-1.45),('G1',os.path.join(ROOT,'assets/G1/configuration/g1_base.usd'),0.0),('R1',os.path.join(ROOT,'assets/R1/configuration/r1_base.usd'),1.45)]:
 prim=f'/World/{name}'; add_reference_to_stage(usd_path=path,prim_path=prim)
 lo,hi=bounds(prim); lift=-lo[2] if lo[0]<1e20 else 0
 p=stage.GetPrimAtPath(prim)
 print('\n',name,'path',prim,'type',p.GetTypeName(),'active',p.IsActive(),'valid',p.IsValid())
 print(' before lo',lo,'hi',hi)
 UsdGeom.XformCommonAPI(p).SetTranslate(Gf.Vec3d(0.0,z,lift))
 print(' translate',p.GetAttribute('xformOp:translate').Get())
 lo2,hi2=bounds(prim); print(' after lo',lo2,'hi',hi2)
