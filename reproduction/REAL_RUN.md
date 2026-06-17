# Running the real DeepSeek comparison (new session)

The environment's network allowlist now includes `api.siliconflow.cn`, but that
change **only applies to new sessions**. This session was started before the
change, so it still can't reach the host. To get real numbers:

## Steps

1. **Start a new Claude Code session** on this repo + branch
   (`claude/affectionate-bohr-dtbfvo`). The new session inherits the updated
   allowlist and clones this code.

2. **Provide the SiliconFlow key at runtime** (do NOT put it in the environment's
   "Environment variables" box — that field is shown to anyone using the
   environment). Either export it in the shell or paste it to the new session and
   ask it to run the command below.

   ```bash
   export SILICONFLOW_API_KEY=sk-...your-key...
   cd reproduction
   python realrun.py --backend siliconflow --agent both --n 12 --m 100
   ```

   - Model defaults to `Pro/deepseek-ai/DeepSeek-V3.2-Exp`; override with
     `--model` or `SF_MODEL=...`.
   - `--n 12 --m 100` is ~50 calls total (cheap). Scale up with `--n`/`--m`.

3. **Rotate the key afterwards** — it was shared in plaintext in chat.

## What it prints

For EHRAgent (edit distance, k=4) and RAP (cosine, k=3):

- `EN / RN / EE` for **MEXTRA** (the authors' own generated prompts) vs
  **MEXTRA++** (adaptive content-seeded, keyword-free, reversibly-encoded).
- a **defense-evasion table** (none / input-keyword / verbatim-echo / output-shape)
  computed post-hoc from the same responses.

## Validate offline first (no network, no key)

```bash
cd reproduction && python realrun.py --backend mock --agent both
```

This exercises the entire pipeline (real memory, retrieval, prompt construction,
decoding, scoring, defenses). In mock mode only `RN` is meaningful — the absolute
`EN` uses the crude single-run `SimulatedAgent`, not a real LLM.
