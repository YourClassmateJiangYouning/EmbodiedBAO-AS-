"""Hold the machine awake while the experiment runs.

The repair processes produced almost nothing between 12:38 and 14:00 because the machine suspended with them
mid-episode: their step timestamps jump from 12:38 to 14:04 with normal 6-12 s intervals on either side.  The
power timeouts are already disabled, so this also sets an explicit execution-state request, which is what
Windows honours for a running task regardless of the idle timer.

SetThreadExecutionState with ES_CONTINUOUS | ES_SYSTEM_REQUIRED tells the kernel that this thread needs the
system awake, and re-asserting it every 30 s keeps it from lapsing.
"""

import ctypes
import time

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001

kernel32 = ctypes.windll.kernel32

print("holding the system awake; press Ctrl+C to stop")
while True:
    previous = kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    if previous == 0:
        print("SetThreadExecutionState failed")
    time.sleep(30)
