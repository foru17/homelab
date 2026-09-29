# AGENTS.md — cross-border-network

Instructions for any coding / chat agent (Claude Code, Codex, Cursor, Kimi, ChatGPT, …) asked to review or
design someone's China ↔ overseas home network with this folder. Thin pointer; the full workflow is in `SKILL.md`.

## What you're doing

Help the user measure their current setup, find the weak hop, and produce a layered plan: entry → dedicated
cross-border line(s) → one Hong Kong hub → cheap landing / residential exits → per-service routing → Tailscale
mesh with relays → security baseline → cost table.

## How to run

1. Read `SKILL.md` (core ideas + workflow), then `references/measurement.md` and `references/pitfalls.md`.
2. Ask the questions in SKILL.md → 「了解现状」 in one message. Missing answers: mark 待补 and continue.
3. Measure with warm-connection latency and single-connection bandwidth, ≥ 3 rounds, medians
   (`scripts/line-bench.sh` if they can run it on a Linux box or OpenWrt router).
4. Output the plan document in the format listed in SKILL.md → 「输出格式」, with 2–3 tiers and a cost table.

## Rules (do not skip)

- **Never ask for or print** IPs, domains, ports, UUIDs, passwords, subscription URLs, or home addresses / cities.
  Use placeholders like `<HK_HUB>`. If the user pastes secrets, tell them to rotate and don't repeat them.
- Don't overbuild: casual users get "an ordinary network plan or one CN2 GIA US server".
- Prices and lines change; tell the user to verify on the vendor's page and start monthly before paying yearly.
