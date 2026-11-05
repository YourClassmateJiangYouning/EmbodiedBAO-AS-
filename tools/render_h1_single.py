"""H1-only frontal render, same camera/lighting as the G1/R1 renders."""
from __future__ import annotations
import math, os
from isaacsim import SimulationApp
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT=os.path.join(ROOT,'reports','robot_front_render')
app=SimulationApp({'headless':True,'renderer':'RaytracedLighting'})
import numpy as np, omni.usd
from pxr import Gf,Usd,UsdGeom,UsdLux,UsdShade,Sdf
from isaacsim.core.api import World
from isaacsim.core.prims import XFormPrim
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.sensors.camera import Camera
from PIL import Image
stage=omni.usd.get_context().get_stage(); world=World(stage_units_in_meters=1.0)
UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z); UsdGeom.SetStageMetersPerUnit(stage,1.0)
def look_at(eye,target,up=(0,0,1)):
 e=np.array(eye,float); t=np.array(target,float); f=t-e; f/=np.linalg.norm(f); u=np.array(up,float); u/=np.linalg.norm(u)
 l=np.cross(u,f); l/=np.linalg.norm(l); u2=np.cross(f,l); u2/=np.linalg.norm(u2); m=np.column_stack((f,l,u2))
 tr=m[0,0]+m[1,1]+m[2,2]
 if tr>0:
  S=math.sqrt(tr+1)*2; w=.25*S; x=(m[2,1]-m[1,2])/S; y=(m[0,2]-m[2,0])/S; z=(m[1,0]-m[0,1])/S
 elif m[0,0]>m[1,1] and m[0,0]>m[2,2]:
  S=math.sqrt(1+m[0,0]-m[1,1]-m[2,2])*2; w=(m[2,1]-m[1,2])/S; x=.25*S; y=(m[0,1]+m[1,0])/S; z=(m[0,2]+m[2,0])/S
 elif m[1,1]>m[2,2]:
  S=math.sqrt(1+m[1,1]-m[0,0]-m[2,2])*2; w=(m[0,2]-m[2,0])/S; x=(m[0,1]+m[1,0])/S; y=.25*S; z=(m[1,2]+m[2,1])/S
 else:
  S=math.sqrt(1+m[2,2]-m[0,0]-m[1,1])*2; w=(m[1,0]-m[0,1])/S; x=(m[0,2]+m[2,0])/S; y=(m[1,2]+m[2,1])/S; z=.25*S
 return np.array([w,x,y,z])
add_reference_to_stage(usd_path=os.path.join(ROOT,'assets/H1/h1.usd'),prim_path='/World/H1')
cache=UsdGeom.BBoxCache(Usd.TimeCode.Default(),[UsdGeom.Tokens.default_]); lo=float('inf'); hi=float('-inf')
for p in stage.Traverse():
 if not str(p.GetPath()).startswith('/World/H1'): continue
 b=cache.ComputeWorldBound(p).ComputeAlignedRange(); a,z=b.GetMin(),b.GetMax()
 if all(abs(v)<1e12 for v in [a[0],a[1],a[2],z[0],z[1],z[2]]): lo=min(lo,a[2]); hi=max(hi,z[2])
lift=-lo
for _p in stage.Traverse():
    if str(_p.GetPath()).startswith('/World/H1'):
        try:
            _p.GetVisibilityAttr().Set(UsdGeom.Tokens.inherited)
        except Exception:
            pass
XFormPrim(prim_paths_expr='/World/H1').set_world_poses(positions=np.array([[0,-1.45,lift]]),orientations=np.array([[1,0,0,0]]))
UsdLux.DomeLight.Define(stage,'/World/Dome').GetIntensityAttr().Set(900)
for p,loc,rot,inten in [('/World/Key',(3,-3,4),(50,0,20),2500),('/World/Fill',(3,3,2.5),(60,0,-20),900)]:
 L=UsdLux.RectLight.Define(stage,p); L.GetIntensityAttr().Set(inten); L.GetWidthAttr().Set(4); L.GetHeightAttr().Set(4); api=UsdGeom.XformCommonAPI(L); api.SetTranslate(Gf.Vec3d(*loc)); api.SetRotate(Gf.Vec3f(*rot))
world.play(); cam=Camera(prim_path='/World/Camera',frequency=30,resolution=(1200,1200)); cam.initialize(); cam.set_focal_length(28.0)
cam.set_world_pose(position=[6.5,0,1],orientation=look_at([6.5,0,1],[0,0,.95]).tolist(),camera_axes='world')
for _ in range(80):
    world.step(render=True); app.update()
a=np.asarray(cam.get_rgba(),dtype=np.uint8)
if a.ndim==1: a=a.reshape(1200,1200,4)
front=a.copy()
for p in stage.Traverse():
    if str(p.GetPath()).startswith('/World/H1'):
        try: p.SetActive(False)
        except Exception: pass
for _ in range(8):
    world.step(render=True); app.update()
b=np.asarray(cam.get_rgba(),dtype=np.uint8)
if b.ndim==1: b=b.reshape(1200,1200,4)
diff=np.abs(front[:,:,:3].astype(np.int16)-b[:,:,:3].astype(np.int16)).max(axis=2)
front[:,:,3]=np.where(diff>4,255,0).astype(np.uint8)
Image.fromarray(front,'RGBA').save(os.path.join(OUT,'H1_front.png'))
print('[render] H1 alpha nonzero',int((front[:,:,3]>0).sum()),'bbox z',lo,hi,flush=True)
