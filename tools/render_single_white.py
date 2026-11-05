"""Render one robot full-body on white, with the same camera for all robots."""
from __future__ import annotations
import argparse, math, os
from isaacsim import SimulationApp
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
p=argparse.ArgumentParser(); p.add_argument('--robot',choices=['H1','G1','R1'],required=True); args=p.parse_args()
out=os.path.join(ROOT,'reports','robot_front_render'); os.makedirs(out,exist_ok=True)
app=SimulationApp({'headless':True,'renderer':'RaytracedLighting'})
import numpy as np, omni.usd
from pxr import Gf,Usd,UsdGeom,UsdLux
from isaacsim.core.api import World
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.sensors.camera import Camera
from PIL import Image
stage=omni.usd.get_context().get_stage(); world=World(stage_units_in_meters=1.0)
UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z); UsdGeom.SetStageMetersPerUnit(stage,1.0)
def look(eye,target):
 e=np.array(eye,float); t=np.array(target,float); f=t-e; f/=np.linalg.norm(f); u=np.array([0,0,1.],float); l=np.cross(u,f); l/=np.linalg.norm(l); u2=np.cross(f,l); u2/=np.linalg.norm(u2); m=np.column_stack((f,l,u2)); tr=m[0,0]+m[1,1]+m[2,2]
 if tr>0: S=math.sqrt(tr+1)*2; w=.25*S; x=(m[2,1]-m[1,2])/S; y=(m[0,2]-m[2,0])/S; z=(m[1,0]-m[0,1])/S
 elif m[0,0]>m[1,1] and m[0,0]>m[2,2]: S=math.sqrt(1+m[0,0]-m[1,1]-m[2,2])*2; w=(m[2,1]-m[1,2])/S; x=.25*S; y=(m[0,1]+m[1,0])/S; z=(m[0,2]+m[2,0])/S
 elif m[1,1]>m[2,2]: S=math.sqrt(1+m[1,1]-m[0,0]-m[2,2])*2; w=(m[0,2]-m[2,0])/S; x=(m[0,1]+m[1,0])/S; y=.25*S; z=(m[1,2]+m[2,1])/S
 else: S=math.sqrt(1+m[2,2]-m[0,0]-m[1,1])*2; w=(m[1,0]-m[0,1])/S; x=(m[0,2]+m[2,0])/S; y=(m[1,2]+m[2,1])/S; z=.25*S
 return np.array([w,x,y,z])
def bounds(path):
 c=UsdGeom.BBoxCache(Usd.TimeCode.Default(),[UsdGeom.Tokens.default_]); lo=float('inf'); hi=float('-inf')
 for prim in stage.Traverse():
  if not str(prim.GetPath()).startswith(path): continue
  b=c.ComputeWorldBound(prim).ComputeAlignedRange(); a,z=b.GetMin(),b.GetMax()
  if all(abs(v)<1e12 for v in [a[0],a[1],a[2],z[0],z[1],z[2]]): lo=min(lo,a[2]); hi=max(hi,z[2])
 return lo,hi
assets={'H1':os.path.join(ROOT,'assets/H1/configuration/h1_urdf_base.usd'),
        'G1':os.path.join(ROOT,'assets/G1/configuration/g1_base.usd'),
        'R1':os.path.join(ROOT,'assets/R1/configuration/r1_base.usd')}
prim=f'/World/{args.robot}'; add_reference_to_stage(usd_path=assets[args.robot],prim_path=prim)
lo,hi=bounds(prim); lift=-lo; print(args.robot,'bbox',lo,hi,'lift',lift,flush=True)
UsdGeom.XformCommonAPI(stage.GetPrimAtPath(prim)).SetTranslate(Gf.Vec3d(0,0,lift))
UsdLux.DomeLight.Define(stage,'/World/Dome').GetIntensityAttr().Set(900)
for pth,loc,rot,inten in [('/World/Key',(3,-3,4),(50,0,20),2500),('/World/Fill',(3,3,2.5),(60,0,-20),900)]:
 L=UsdLux.RectLight.Define(stage,pth); L.GetIntensityAttr().Set(inten); L.GetWidthAttr().Set(4); L.GetHeightAttr().Set(4); api=UsdGeom.XformCommonAPI(L); api.SetTranslate(Gf.Vec3d(*loc)); api.SetRotate(Gf.Vec3f(*rot))
world.play(); world.reset()
art = Articulation(prim_paths_expr=prim)
try:
    art.initialize()
    names = list(art.dof_names)
    targets = []
    for n in names:
        low = n.lower()
        if 'shoulder_pitch' in low:
            targets.append(0.0)
        elif 'elbow' in low:
            targets.append(1.5708)
        elif 'shoulder_roll' in low or 'hip' in low or 'knee' in low:
            targets.append(0.0)
        else:
            targets.append(0.0)
    art.set_joint_positions(np.asarray(targets, dtype=float))
    for _ in range(20):
        world.step(render=True)
    print('posed', args.robot, 'dofs', len(names), flush=True)
except Exception as e:
    print('pose error', args.robot, type(e).__name__, e, flush=True)
cam=Camera(prim_path='/World/Camera',frequency=30,resolution=(1200,1200)); cam.initialize(); cam.set_focal_length(35.0)
cam.set_world_pose(position=[30,0,0.95],orientation=look([30,0,0.95],[0,0,0.90]).tolist(),camera_axes='world')
for _ in range(100): world.step(render=True); app.update()
a=np.asarray(cam.get_rgba(),dtype=np.uint8)
if a.ndim==1: a=a.reshape(1200,1200,4)
a[:,:,3]=255
path=os.path.join(out,f'{args.robot}_white.png'); Image.fromarray(a,'RGBA').save(path); print('wrote',path,flush=True)
