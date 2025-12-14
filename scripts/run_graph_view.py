"""
Convenience launcher for the KG graph view.

Steps performed:
1) Starts a simple HTTP server from the repo root (default port 8000).
2) Prints the URL to open the interactive viewer.

Usage:
  python scripts/run_graph_view.py
  python scripts/run_graph_view.py --port 9000
"""

import argparse
import http.server
import os
import socketserver
import webbrowser


def main():
    parser = argparse.ArgumentParser(description="Run a simple HTTP server for the KG graph viewer.")
    parser.add_argument("--port", type=int, default=8000, help="Port to serve on (default: 8000).")
    args = parser.parse_args()

    port = args.port
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", port), handler) as httpd:
        url = f"http://localhost:{port}/scripts/graph_view.html"
        print(f"Serving KG graph viewer at {url}")
        print("Hint: Load data/interim/file_with_relationships.json via the UI.")
        try:
            webbrowser.open(url)
        except Exception:
            pass
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server.")


if __name__ == "__main__":
    main()
