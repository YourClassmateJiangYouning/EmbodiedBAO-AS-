| Model | Level | A/S | Width (m) | Pass % | Turned % | Sideways % | 1st Turn | Rot@Pass | Avg Pass Steps |
|---|---|---|---|---|---|---|---|---|---|

| Model | Turned Threshold (A/S) | Sideways Threshold (A/S) | Human Reference | Gap vs Human | Class | Graded? |
|---|---|---|---|---|---|---|
| kimi-kimi-k2.5 | - | - | 1.30 | - | no_threshold | - |

Turned Threshold is the widest A/S at which the torso was rotated at all;
Sideways Threshold additionally requires the episode to have succeeded.
The human reference of 1.30 measures whether the shoulders were rotated,
so Turned Threshold is the closer analogue.
Graded? is flat_always / flat_never when the rotation rate does not vary
with A/S at all: a fixed policy has no threshold, whatever the number says.
