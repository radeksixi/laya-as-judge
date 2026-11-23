---
title: Laya-as-a-Judge Interactive Demo
emoji: ⚖️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.20.0
python_version: 3.11
app_file: app.py
pinned: false
license: apache-2.0
short_description: Zero-output-token Decision, Judge & Agent Governance API
---

# Laya-as-a-Judge v1.1.0 ⚡⚖️

Interactive CPU playground for **[`radeksixi/laya-as-judge`](https://github.com/radeksixi/laya-as-judge)** (Apache-2.0) — the zero-output-token Decision, Judge, Ecosystem Drop-In, and Hybrid Agent Governance (`COMMIT / FALLBACK / ABSTAIN / BLOCK`) API built alongside [Sixi Lab R&D](https://sixi.ai) / [Sixi Assure](https://sixi.ai/en/assure/).

- **Zero output tokens, 0% JSON parse failures**
- **Three unified modes:** Decision API (`/v1/decisions`, `/v1/systemone`), Judge API (`/v1/judge`), and Hybrid **The Judged Decision** (`/v1/judged-decision`)
- **Ecosystem drop-ins:** `/v1/decisions/batch`, OpenAI `/v1/moderations`, Cohere/vLLM `/v1/rerank`, Best-of-N `/v1/rank`, Prometheus `/metrics`, plus **FastMCP** & **LangGraph** middleware.
- **Note:** This free CPU Space runs the instant zero-VRAM `EmulatorBackend` + deterministic security policy guards (`spoofed_audit_guard_rule`). For full local neural inference (`78.7%` 6-benchmark average, `12/12` red-team, `$0.00` cloud), deploy locally with `laya-judge serve --profile rtx-16gb`.
