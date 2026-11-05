"""One group shot: H1, G1, R1 side by side, far frontal camera, transparent."""
from __future__ import annotations
import math, os
from isaacsim import SimulationApp
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT=os.path.join(ROOT,'reports','robot_front_render'); os.makedirs(OUT,exist_ok=True)
app=SimulationApp({'headless':True,'renderer':'RaytracedLighting'})
import numpy as np, omni.usd
from pxr import Gf,Sdf,Usd,UsdGeom,UsdLux,UsdShade
from isaacsim.core.api import World
from isaacsim.core.prims import XFormPrim
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.sensors.camera import Camera
from PIL import Image,ImageDraw,ImageFont
stage=omni.usd.get_context().get_stage(); world=World(stage_units_in_meters=1.0)
UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z); UsdGeom.SetStageMetersPerUnit(stage,1.0)
def look(eye,target):
 e=np.array(eye,float); t=np.array(target,float); f=t-e; f/=np.linalg.norm(f); u=np.array([0,0,1.],float); l=np.cross(u,f); l/=np.linalg.norm(l); u2=np.cross(f,l); u2/=np.linalg.norm(u2); m=np.column_stack((f,l,u2)); tr=m[0,0]+m[1,1]+m[2,2]
 if tr>0: S=math.sqrt(tr+1)*2; w=.25*S; x=(m[2,1]-m[1,2])/S; y=(m[0,2]-m[2,0])/S; z=(m[1,0]-m[0,1])/S
 elif m[0,0]>m[1,1] and m[0,0]>m[2,2]: S=math.sqrt(1+m[0,0]-m[1,1]-m[2,2])*2; w=(m[2,1]-m[1,2])/S; x=.25*S; y=(m[0,1]+m[1,0])/S; z=(m[0,2]+m[2,0])/S
 elif m[1,1]>m[2,2]: S=math.sqrt(1+m[1,1]-m[0,0]-m[2,2])*2; w=(m[0,2]-m[2,0])/S; x=(m[0,1]+m[1,0])/S; y=.25*S; z=(m[1,2]+m[2,1])/S
 else: S=math.sqrt(1+m[2,2]-m[0,0]-m[1,1])*2; w=(m[1,0]-m[0,1])/S; x=(m[0,2]+m[2,0])/S; y=(m[1,2]+m[2,1])/S; z=.25*S
 return np.array([w,x,y,z])
def bz(path):
 c=UsdGeom.BBoxCache(Usd.TimeCode.Default(),[UsdGeom.Tokens.default_]); lo=float('inf'); hi=float('-inf')
 for p in stage.Traverse():
  if not str(p.GetPath()).startswith(path): continue
  b=c.ComputeWorldBound(p).ComputeAlignedRange(); a,z=b.GetMin(),b.GetMax()
  if all(abs(v)<1e12 for v in [a[0],a[1],a[2],z[0],z[1],z[2]]): lo=min(lo,a[2]); hi=max(hi,z[2])
 return lo,hi
for name,path,z in [('H1',os.path.join(ROOT,'assets/H1/configuration/h1_urdf_base.usd'),-1.45),('G1',os.path.join(ROOT,'assets/G1/configuration/g1_base.usd'),0.0),('R1',os.path.join(ROOT,'assets/R1/configuration/r1_base.usd'),1.45)]:
 prim=f'/World/{name}'; add_reference_to_stage(usd_path=path,prim_path=prim); lo,hi=bz(prim); lift=-lo
 print(name,'bbox',lo,hi,'lift',lift,flush=True)
 UsdGeom.XformCommonAPI(stage.GetPrimAtPath(prim)).SetTranslate(Gf.Vec3d(0.0, z, lift))
UsdLux.DomeLight.Define(stage,'/World/Dome').GetIntensityAttr().Set(900)
for p,loc,rot,inten in [('/World/Key',(3,-3,4),(50,0,20),2500),('/World/Fill',(3,3,2.5),(60,0,-20),900)]:
 L=UsdLux.RectLight.Define(stage,p); L.GetIntensityAttr().Set(inten); L.GetWidthAttr().Set(4); L.GetHeightAttr().Set(4); api=UsdGeom.XformCommonAPI(L); api.SetTranslate(Gf.Vec3d(*loc)); api.SetRotate(Gf.Vec3f(*rot))
world.play(); world.reset(); cam=Camera(prim_path='/World/Camera',frequency=30,resolution=(2400,1000)); cam.initialize(); cam.set_focal_length(35.0)
cam.set_world_pose(position=[15,0,1.0],orientation=look([15,0,1.0],[0,0,1.0]).tolist(),camera_axes='world')
for _ in range(120): world.step(render=True); app.update()
a=np.asarray(cam.get_rgba(),dtype=np.uint8)
if a.ndim==1: a=a.reshape(1000,2400,4)
a[:,:,3]=255
img=Image.fromarray(a,'RGBA'); p=os.path.join(OUT,'robot_group_H1_G1_R1.png'); img.save(p); print('wrote',p,img.size,'alpha nz',int((a[:,:,3]>0).sum()),flush=True)
