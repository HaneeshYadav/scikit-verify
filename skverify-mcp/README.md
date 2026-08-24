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

Install and register with a client (Claude Code, Cursor, any other):

```bash
pip install "scikit-verify[mcp]"
```

```json
{"mcpServers": {"skverify": {"command": "skverify-mcp"}}}
```

