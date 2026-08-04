"""実験機サーバの入口。  python -m substrate_probe.serve [port]"""
import sys

from .server import DEFAULT_PORT, serve

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PORT
    serve(port)
