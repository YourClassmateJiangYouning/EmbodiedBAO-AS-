from isaacsim import SimulationApp
import os
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
app=SimulationApp({'headless':True})
import omni.kit.commands
urdf=os.path.join(ROOT,'assets/H1_urdf/render_h1.urdf'); usd=os.path.join(ROOT,'assets/H1/render_h1.usd')
st,cfg=omni.kit.commands.execute('URDFCreateImportConfig'); cfg.merge_fixed_joints=False; cfg.convex_decomp=False; cfg.fix_base=False
st,p=omni.kit.commands.execute('URDFParseAndImportFile',urdf_path=urdf,import_config=cfg,dest_path=usd,get_articulation_root=True)
print('converted',st,p,os.path.exists(usd),flush=True)
