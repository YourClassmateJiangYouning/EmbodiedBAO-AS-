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

    # A stale instance still holding the port would otherwise make this one exit with "address in use",
    # leaving the operator looking at an old window that is missing the newest code.  Walk forward until a
    # port is free and use that instead; the printed URL and the opened tab always point at the real one.
    import socket

    chosen = None
    for candidate in [port] + list(range(port + 1, port + 15)):
        # A plain bind, with SO_REUSEADDR deliberately NOT set: on Windows SO_REUSEADDR lets a second
        # socket bind a port that is already actively listened on, so the probe would report every port as
        # free and two servers would end up sharing one address.
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            probe.bind(("127.0.0.1", candidate))
            chosen = candidate
            break
        except OSError:
            continue
        finally:
            probe.close()

    if chosen is None:
        print("端口 %d 到 %d 都被占用，请先关闭旧的窗口再试。" % (port, port + 14))
        return 1
    if chosen != port:
        print("提示：端口 %d 已被占用（可能是之前没关干净的实例），改用 %d。" % (port, chosen))

    httpd = serve(port=chosen)
    url = "http://127.0.0.1:%d/" % chosen
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
