"""Robot body profiles for Stage 3.

Stage 1/2 were all run with the Unitree H1 (0.570 m shoulder width).  Stage 3
repeats the same body-as-obstacle setting with two smaller Unitree humanoids:

* G1  : official standing size 1320 x 450 x 200 mm, 35 kg
* R1  : official standing size 1230 x 357 x 190 mm, 27-29 kg

The profile changes only the robot's physical body, camera anchor, and the A/S
ladder.  It does not change the room, actions, step length, scene markers,
scoring plane or prompt wording.

DEFINITION OF "SHOULDER WIDTH": this is the manufacturer's official standing
body WIDTH (the Y dimension in the standing table), exactly the quantity Stage 1
used for the H1 (0.570 m from the official 570 mm width).  It is NOT the span
between the two shoulder-pitch joints: those are about 0.200 m apart on G1 and
0.171 m on R1, which is a joint-layout quantity, not the body width that has to
clear a doorway.

`eye_height` is the same 93% eye-height convention used for the H1
(1.68 / 1.806); `head_height` is the authored head anchor.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class RobotProfile:
    key: str
    name: str
    usd_path: str
    prim_name: str
    shoulder_width: float
    torso_thickness: float
    body_height: float
    head_height: float
    eye_height: float
    source_note: str


PROFILES: Dict[str, RobotProfile] = {
    "H1": RobotProfile(
        key="H1",
        name="Unitree H1",
        usd_path="assets/H1/h1.usd",
        prim_name="H1",
        shoulder_width=0.570,
        torso_thickness=0.220,
        body_height=1.806,
        head_height=1.550,
        eye_height=1.680,
        source_note="Existing Stage 1/2 H1 profile; official width 0.570 m, thickness 0.220 m, measured height 1.806 m.",
    ),
    "G1": RobotProfile(
        key="G1",
        name="Unitree G1",
        usd_path="assets/G1/configuration/g1_base.usd",
        prim_name="G1",
        shoulder_width=0.450,
        torso_thickness=0.200,
        body_height=1.323,
        head_height=1.135,
        eye_height=1.230,
        source_note="Official G1 1320x450x200 mm (width 0.450, thickness 0.200); converted 23-DOF URDF bbox height 1.323 m.",
    ),
    "R1": RobotProfile(
        key="R1",
        name="Unitree R1",
        usd_path="assets/R1/configuration/r1_base.usd",
        prim_name="R1",
        shoulder_width=0.357,
        torso_thickness=0.190,
        body_height=1.230,
        head_height=1.055,
        eye_height=1.144,
        source_note="Official R1 1230x357x190 mm (width 0.357, thickness 0.190); converted URDF bbox height 1.230 m.",
    ),
}

DEFAULT_PROFILE = "H1"


def get_profile(key: str | None = None) -> RobotProfile:
    if not key:
        return PROFILES[DEFAULT_PROFILE]
    k = str(key).upper()
    if k not in PROFILES:
        raise ValueError(f"unknown robot profile {key!r}; expected one of {sorted(PROFILES)}")
    return PROFILES[k]


def as_width(profile: RobotProfile, a_s: float) -> float:
    """Channel width for a requested A/S ratio and this robot body."""
    return round(float(profile.shoulder_width) * float(a_s), 6)
