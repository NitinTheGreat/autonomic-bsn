"""Mock HTTP backends for the Phase 1 client test suite.

Every mode below reproduces a REAL response shape (or a real failure shape)
from the provider it names. Nothing here is a research result.

Modes:
  llamacpp_good      non-uniform completion_probabilities      -> PASS
  llamacpp_uniform   exactly uniform logprobs                  -> FAIL (noise)
  llamacpp_nofield   200 OK, no logprob field                  -> FAIL loudly
  vllm_good          OpenAI-style top_logprobs with echo       -> PASS
  ollama_nologprob   native shape, no logprobs key             -> FAIL loudly
  gemini_good        logprobsResult present                    -> PASS
  gemini_nologprobs  Gemini 3.x shape: no logprobsResult       -> FAIL loudly
  openai_good        chat/completions logprobs.content[0]      -> PASS
  openai_nologprobs  200 OK, `logprobs` absent                 -> FAIL loudly
  openai_nolabels    top_logprobs holds no label letters       -> FAIL loudly
  openai_uniform     exactly uniform logprobs                  -> FAIL (noise)
  anthropic_uniform  k votes spread evenly over 6 labels       -> FAIL (noise)
  anthropic_votes    k responses cycling a fixed vote pattern  -> self-consistency
  anthropic_garbage  k responses with no valid letter          -> FAIL loudly
"""
import json
import math
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

MODE = sys.argv[1] if len(sys.argv) > 1 else "openai_good"
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8099

# Shared fixture: the SAME logprobs for every logprob backend, so all of them
# must produce byte-identical distributions through the shared softmax.
LP = {"D": -0.223, "E": -1.897, "F": -3.100,
      "C": -4.050, "B": -5.200, "A": -6.010}
UNIFORM = {k: math.log(1.0 / 6.0) for k in "ABCDEF"}
DESC = sorted(LP.items(), key=lambda x: -x[1])

# 7 D + 2 E + 1 A  -> D 0.7, E 0.2, A 0.1 at k=10
VOTE_PATTERN = ["D"] * 7 + ["E"] * 2 + ["A"]
_call_n = {"i": 0}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        sys.stderr.write("REQ " + json.dumps({
            "auth": bool(self.headers.get("Authorization")),
            "x_api_key": bool(self.headers.get("x-api-key")),
            "anthropic_version": self.headers.get("anthropic-version"),
            "body_keys": sorted(req),
            "top_logprobs": req.get("top_logprobs"),
            "logprobs": req.get("logprobs"),
            "temperature": req.get("temperature"),
            "max_tokens": req.get("max_tokens"),
        }) + "\n")
        sys.stderr.flush()

        # ---------------- llama.cpp ----------------
        if MODE.startswith("llamacpp"):
            if MODE == "llamacpp_nofield":
                return self._send({"content": "D", "stop": True})
            lp = UNIFORM if MODE == "llamacpp_uniform" else LP
            return self._send({"content": "D", "completion_probabilities": [{
                "id": 396, "token": "D", "logprob": lp["D"],
                "top_logprobs": [{"id": 0, "token": k, "logprob": v}
                                 for k, v in sorted(lp.items(),
                                                    key=lambda x: -x[1])]}]})

        # ---------------- vLLM ----------------
        if MODE == "vllm_good":
            return self._send({"choices": [{"text": "D", "logprobs": {
                "tokens": ["<prompt>", "D"],
                "top_logprobs": [None, dict(LP)]}}]})

        # ---------------- Ollama ----------------
        if MODE == "ollama_nologprob":
            return self._send({"model": "x", "response": "D", "done": True})

        # ---------------- Gemini / Vertex ----------------
        if MODE.startswith("gemini"):
            if MODE == "gemini_nologprobs":
                return self._send({"candidates": [{
                    "content": {"parts": [{"text": "D"}], "role": "model"},
                    "finishReason": "MAX_TOKENS"}]})
            return self._send({"candidates": [{
                "content": {"parts": [{"text": "D"}], "role": "model"},
                "finishReason": "MAX_TOKENS", "avgLogprobs": LP["D"],
                "logprobsResult": {
                    "chosenCandidates": [{"token": "D",
                                          "logProbability": LP["D"]}],
                    "topCandidates": [{"candidates": [
                        {"token": k, "logProbability": v}
                        for k, v in DESC]}]}}]})

        # ---------------- OpenAI ----------------
        if MODE.startswith("openai"):
            if MODE == "openai_nologprobs":
                # 200 OK with a normal message and NO logprobs field.
                return self._send({"choices": [{
                    "index": 0, "finish_reason": "length",
                    "message": {"role": "assistant", "content": "D"}}]})
            if MODE == "openai_uniform":
                toks = [(k, v) for k, v in UNIFORM.items()]
            elif MODE == "openai_nolabels":
                # Model answered with punctuation/whitespace, no label letter.
                toks = [("\n", -0.10), (" ", -1.20), ("The", -2.30),
                        ("Answer", -3.10), (":", -4.00)]
            else:
                toks = DESC
            return self._send({"choices": [{
                "index": 0, "finish_reason": "length",
                "message": {"role": "assistant", "content": "D"},
                "logprobs": {"content": [{
                    "token": toks[0][0], "logprob": toks[0][1], "bytes": [],
                    "top_logprobs": [{"token": t, "logprob": v, "bytes": []}
                                     for t, v in toks]}]}}]})

        # ---------------- Anthropic ----------------
        if MODE.startswith("anthropic"):
            if MODE == "anthropic_garbage":
                txt = "?"
            elif MODE == "anthropic_uniform":
                # 6 labels, one vote each at k=6 -> perfectly uniform
                txt = "ABCDEF"[_call_n["i"] % 6]
            else:
                txt = VOTE_PATTERN[_call_n["i"] % len(VOTE_PATTERN)]
            _call_n["i"] += 1
            return self._send({
                "id": "msg_x", "type": "message", "role": "assistant",
                "model": "mock", "stop_reason": "max_tokens",
                "content": [{"type": "text", "text": txt}]})

        return self._send({})


if __name__ == "__main__":
    print("mock %s on :%d" % (MODE, PORT), flush=True)
    HTTPServer(("127.0.0.1", PORT), H).serve_forever()
