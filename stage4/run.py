"""Launcher for the Stage 4 operator window.

    python stage4/run.py            # port 8765, opens the browser
    python stage4/run.py 8790       # another port, when 8765 is taken

Set the API key once in the environment if you do not want to paste it into the window every session:

    setx BOYUE_API_KEY "sk-..."        (Windows, new terminals)
    export BOYUE_API_KEY="sk-..."      (bash)
"""

from __future__ import annotations

import os
import sys
import threading
import webbrowser

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def main() -> int:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    from server import serve

    httpd = serve(port=port)
    url = "http://127.0.0.1:%d/" % port
    print("=" * 68)
    print(" Stage 4 操作台已启动： %s" % url)
    print(" 关闭窗口：在本终端按 Ctrl+C")
    print(" 会话数据目录： %s" % os.path.join(HERE, "sessions"))
    print("=" * 68)
    threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
