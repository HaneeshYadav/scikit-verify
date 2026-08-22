# skverify-mcp

An MCP server exposing scikit-verify's `to_sympy` to coding agents.

Two tools:

* `trace(source, function, args)`: translate a Python+NumPy function
  to symbolic mathematics. Returns the certificate: `formula`,
  `latex`, `preconditions`, `definitions`, `checked` (compiled-call
  contract verdicts), `value`. Refusals return one sentence in
  `refused`, never a stack trace.
* `derivation()`: the step-by-step derivation of the most recent
  trace.

Run:

```bash
python skverify-mcp/server.py
```

Or register with a client:

```json
{"mcpServers": {"skverify": {"command": "python",
  "args": ["/path/to/skverify-mcp/server.py"]}}}
```

Requires `mcp` and `scikit-verify`.
