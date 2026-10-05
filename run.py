"""One-command launcher: python run.py  (generates data + trains models on first run, then serves http://localhost:8000)"""
import os, sys
D = os.path.dirname(os.path.abspath(__file__)); os.chdir(D); sys.path.insert(0, D)
if not os.path.exists("models/models.pkl"):
    print("First run: generating synthetic data and training models (~1 min)..."); import ml; ml.train()
import server; server.seed_all(); import webbrowser, threading
port = int(os.environ.get("PORT", 8000)); threading.Timer(1.2, lambda: webbrowser.open(f"http://localhost:{port}")).start()
print(f"DICE prototype (synthetic data) -> http://localhost:{port}"); server.ThreadingHTTPServer(("127.0.0.1", port), server.H).serve_forever()
