# Why the room's walls cannot be re-coloured, and what the park's red actually is

Diagnosis from the session on 2026-10-12, left as a work order because the fix was NOT applied.
Everything below is a measurement, with the log it came from. No item on this page is fixed.

## The room surfaces carry materials that override `displayColor`

`scene_builder.paint()` works by authoring `primvars:displayColor`.  A bound material wins over
`displayColor`, so paint only reaches a surface that has NO material bound.

`environment._add_box` binds a `UsdPreviewSurface` to every box it creates, through
`_create_and_bind_material(..., metallic=0.0, roughness=0.6)` (`environment.py:930`).  The four room
surfaces are created with `_add_box`, so all four already carry a material by the time
`scene_builder.apply_scene` runs.  The only surface that responds to the palette is the one WITHOUT
such a binding:

    /World/Ground                is_gprim=True displayColor=(0.33, 0.46, 0.24)   <- takes the paint
    /World/room_side_left        is_gprim=True displayColor=(0.6, 0.63, 0.56)    <- ignored
    /World/room_side_right       is_gprim=True displayColor=(0.6, 0.63, 0.56)    <- ignored
    /World/room_ceiling          is_gprim=True displayColor=(0.88, 0.9, 0.94)    <- ignored
    /World/room_far              is_gprim=True displayColor=(0.2, 0.44, 0.24)    <- ignored
    /World/room_near             is_gprim=True displayColor=None

The displayColor IS authored on all four (`diag_logs/wallpaint.log`); it is simply not what is
drawn.  Proof by painting everything red (`diag_logs/allworld.log`, `diag_logs/allworld_red.png`):
paint every Gprim under `/World` red, render the eye frame, and see which pixels obey -- 46.34 % of
the frame turns red, and the parts that stay grey are the four room surfaces and the obstacle-wall
panels.  The floor turns red.  The walls do not.

**Consequence.**  No value in `scenes.SURFACE_COLOURS` can change the corridor walls, the ceiling or
the far wall while the preview is rendered this way, and that is why the walls stayed the same
colour through the `872b069` palette change.  The floors DID change, because `/World/Ground` takes
the paint.

**The fix, not yet applied.**  `paint()` must reach the bound material: find the material bound to
the prim (`UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()`), get its surface shader
(`material.ComputeSurfaceSource()`), and set the shader's `inputs:diffuseColor`.  That is a change to
`scene_builder.paint()` and it would make `SURFACE_COLOURS` effective for every scene.  It was not
applied because it must be followed by a render that measures the wall hue actually changing, and
the session ran out of room before that could be done.

**Rejected along the way.**  Lowering the wall albedo (to 0.26-0.42) was tried and REVERTED: the
bright wall band stayed (185,187,187)-(187,185,185) with normalised ratios (0.98-1.00), i.e. white,
so the change produced no measurable effect and was not kept.  Those numbers are what first showed
the albedo was not the lever, because it was not reaching the render at all.

## The far wall is visible, but not where it was measured

The far wall at x = 16 IS in the eye frame: in `allworld_red.png` the green area between the two blue
panels is `/World/room_far` still showing its `environment` colour.  It reads as a green band rather
than the scene's colour because the obstacle wall hides most of it.  Measuring it at the frame centre
gives the same numbers in every scene -- `(138,175,153)`, `(140,174,150)`, `(136,175,149)`,
`(144,180,159)` -- because the red marker disc sits in that window and dominates it.  Any future
measurement of the far wall has to avoid the marker's projection or mask it out.

## The park's red is the vegetation's MDL shaders failing to compile

A `stage1.4` render prints exactly eight:

    MDLC comp error: C120 could not find module '::.::materials::bark3'
                                                    ... 'TreeBark_10'
                                                    ... 'Pine_needles'
                                                    ... 'Barberry_Mat'
                                                    ... 'American_Beech_leaf'
                                                    ... '::..::..::..::..::nv_core::materials::SimPBR'

Those are the park's trees and shrubs (`Chinese_Juniper`, `Douglas_Fir`, `Elm_Sapling`, `Barberry`,
`Goldflame_Spirea`).  The modules are not vendored: scanning each asset's own file for the files it
mentions finds `Bark_10.mdl`, `TreeBark_10.mdl`, `Barberry_Mat.mdl`, `Pine_needles.mdl` and texture
PNGs that are absent from `assets/isaac` and `assets/bucket`.  With the shader missing the renderer
falls back, and the fallback is what the owner sees as "全部都是红色" -- the same failure the project
already recorded for MDL surfaces, now landing on the vegetation instead of the walls.

**Consequence.**  The declared `DRESSING_HEDGE` / `DRESSING_GRASS` colours on the park's plants do
nothing, because they are `displayColor` on the clearance box and the prop has its own material.
"没有可看见风化木，草和绿篱" is the same cause: what should be a hedge renders as a red mass.

**Two candidate fixes, neither applied.**
  1. Bind a working material over the vegetation, the way `OmniPBR` already works for `dolly` and
     `011_banana`.  A first test bound a flat `UsdPreviewSurface` to `tree_conifer` and the measured
     colour moved from `(126,126,119)` to `(140,139,132)` -- a 10-level shift that is mostly render
     settling, NOT evidence that the green took, because the probe's sampling window was dominated by
     the room behind the prop.  This must be re-measured with a tight window on the prop alone before
     it is believed.
  2. Vendor the missing MDL modules and textures from the public bucket, as `fetch_bucket_set.py`
     does for props.  That is the "correct" fix but it needs the module paths, which the USD files
     encode in binary and this session did not extract.

## The remaining box-only items, checked against the eye frame

Projecting each item's centre through the eye camera (`sc.project(at, eye_x=0.5)`, in frame when u
and v are both within [-1, 1]):

  * `stage1.2/bay_sign` -- u=+0.18 v=+0.59, IN FRAME, no asset.  No substitute exists in the library
    or in the bucket: the only sign-like asset vendored is `sign_plaza_fountain_01`, whose folder has
    no separate sign layer, and `SM_BooksSet_11_167` cannot load (missing `SM_BooksSet_11.usd`).
  * `stage1.3/reading_lamp` -- v=-3.36, NOT in frame.
  * `stage1.2/forklift` -- u=+0.08 v=+0.81, IN FRAME, no asset.
  * `stage1.5/shelf_left/right`, `price_strip`, `promo_banner` -- not measured against the eye frame
    in this session.
  * `stage1.4/park_sign` -- not measured against the eye frame in this session.

So "不允许再出现方块" needs, at minimum, a decision for `bay_sign` and `forklift`, which have no
asset to be given.  Building them from primitives with materials (a sign plate, a fork carriage) is
possible and is not the same thing as leaving them as plain cubes, but it is new geometry rather
than a substitution and was not attempted.
