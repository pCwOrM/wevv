"""
werr.server: Production-Grade Zero-Dependency HTTP Decision Server.
Implements the TypeSafe-compatible wire format:
  POST /v1/systemone
  POST /decide
  GET /health

Enables live interactive benchmark evaluation with zero external dependencies
(runs entirely on Python standard library + numpy).

Air-Gapped & Telemetry Invariant
--------------------------------
100% Air-Gapped by default. Zero outbound telemetry or network calls are made
during evaluation. Telemetry is strictly opt-in (set WERR_TELEMETRY=1 to enable).
"""
import os
import sys
import json
import time
import argparse
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, Any

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from werr.engine import WerrEngine
from werr.adapters.wire_adapter import JevWireAdapter


class WerrJevWireHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):
        if getattr(self.server, "verbose", False):
            sys.stderr.write(
                "%s - - [%s] %s\n"
                % (self.address_string(), self.log_date_time_string(), format % args)
            )

    def _send_json(self, status_code: int, data: Dict[str, Any]):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.end_headers()

    def do_GET(self):
        if self.path in ["/health", "/"]:
            self._send_json(200, {
                "status": "healthy",
                "system": "werr",
                "version": "0.5.1",
                "engine": "System-One Zero-Memory Fractal Kernel",
                "wire_format": "TypeSafe /v1/systemone Compatible",
            })
        else:
            self._send_json(404, {"error": "Not Found"})

    def do_POST(self):
        content_len = int(self.headers.get("Content-Length", 0))
        if content_len == 0:
            self._send_json(400, {"error": "Empty request body"})
            return

        raw_body = self.rfile.read(content_len)
        try:
            req_data = json.loads(raw_body.decode("utf-8"))
        except Exception as e:
            self._send_json(400, {"error": f"Invalid JSON body: {str(e)}"})
            return

        t0 = time.perf_counter()

        # ── TypeSafe wire format: POST /v1/systemone ──────────────────────────
        if self.path.rstrip("/") == "/v1/systemone":
            state = req_data.get("state", "")
            questions = req_data.get("questions", {})
            model_name = req_data.get("model", "werr-system-one")

            answers = {}
            for q_id, q_def in questions.items():
                res = self.server.engine.decide({
                    "id": q_id,
                    "state": state,
                    "question": q_def,
                    "labels": q_def.get("labels", []),
                    "expected": None,
                })
                q_type = q_def.get("type", "choice")
                if q_type == "noul":
                    answers[q_id] = {
                        "type": "noul",
                        "noul": res["probs"].get("yes", 0.5),
                        "probabilities": res["probs"],
                    }
                elif q_type == "choice":
                    answers[q_id] = {
                        "type": "choice",
                        "choice": res["predicted"],
                        "probabilities": res["probs"],
                    }
                elif q_type == "score":
                    answers[q_id] = {
                        "type": "score",
                        "score": res["predicted"],
                        "probabilities": res["probs"],
                    }

            lat_ms = (time.perf_counter() - t0) * 1000.0
            self._send_json(200, {
                "model": model_name,
                "usage": {
                    "input_tokens": max(1, len(str(state).split()) + 15),
                    "output_tokens": 1,
                },
                "answers": answers,
                "latency_ms": round(lat_ms, 3),
            })

        # ── Direct endpoint: POST /decide ──────────────────────────────────────
        elif self.path.rstrip("/") == "/decide":
            res = self.server.engine.decide(req_data)
            lat_ms = (time.perf_counter() - t0) * 1000.0
            self._send_json(200, {**res, "latency_ms": round(lat_ms, 3)})

        else:
            self._send_json(404, {"error": f"Endpoint not found: {self.path}"})


def run_server(
    host: str = "0.0.0.0",
    port: int = 8443,
    verbose: bool = False,
    enable_telemetry: bool = False,
    domain_mode: str = "none",
    mode: str = None,
    enable_domain: bool = None,
    enable_lexical: bool = None,
    enable_resonance: bool = None,
):
    if not enable_telemetry:
        os.environ["WERR_TELEMETRY"] = "0"
    else:
        os.environ["WERR_TELEMETRY"] = "1"

    resolved_mode = mode if mode is not None else ("hybrid" if (enable_domain or domain_mode == "multi") else "pure_fractal")
    engine = WerrEngine(
        mode=resolved_mode,
        domain_mode=domain_mode,
        enable_domain=enable_domain,
        enable_lexical=enable_lexical,
        enable_resonance=enable_resonance,
    )
    adapter = JevWireAdapter(engine=engine)

    server = HTTPServer((host, port), WerrJevWireHandler)
    server.engine = adapter
    server.verbose = verbose

    print(f"[*] Werr System-One Decision Server v0.5.1 running at http://{host}:{port}")
    print(
        f"[*] Architecture : domain_mode='{engine.domain_mode}', mode='{engine.mode}' "
        f"[domain={engine.enable_domain}, lexical={engine.enable_lexical}, resonance={engine.enable_resonance}]"
    )
    print(f"[*] Wire Formats : POST /v1/systemone, POST /decide, GET /health")
    print(
        f"[*] Telemetry   : "
        + ("ON (research telemetry enabled)" if enable_telemetry else "OFF (100% air-gapped, zero outbound requests)")
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[*] Shutting down Werr server...")
        server.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Werr Decision HTTP Server (v0.5.1)")
    parser.add_argument("--host", default="0.0.0.0", help="Binding host")
    parser.add_argument("--port", type=int, default=8443, help="Binding port")
    parser.add_argument("--domain-mode", default="none", choices=["none", "multi"], help="Domain routing: 'none' (domainless monolithic default) or 'multi' (domain gates)")
    parser.add_argument("--mode", default=None, choices=["pure_fractal", "lexical", "resonance", "hybrid", "production"], help="Engine dictionary mode (auto-aligned with domain-mode if omitted)")
    parser.add_argument("--enable-domain", dest="enable_domain", action="store_true", default=None, help="Explicitly enable multi-domain routing")
    parser.add_argument("--enable-lexical", dest="enable_lexical", action="store_true", default=None, help="Explicitly enable lexical dictionary")
    parser.add_argument("--enable-resonance", dest="enable_resonance", action="store_true", default=None, help="Explicitly enable resonance dictionary")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    parser.add_argument(
        "--enable-telemetry",
        action="store_true",
        help="Explicitly enable research telemetry dispatch. Default is strictly OFF (air-gapped).",
    )
    parser.add_argument(
        "--no-telemetry",
        action="store_true",
        help="Explicitly disable all telemetry (default behavior).",
    )
    args = parser.parse_args()
    telemetry_flag = args.enable_telemetry and not args.no_telemetry
    run_server(
        args.host,
        args.port,
        args.verbose,
        telemetry_flag,
        args.domain_mode,
        args.mode,
        args.enable_domain,
        args.enable_lexical,
        args.enable_resonance,
    )
