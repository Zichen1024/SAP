# SAP: State-Guided Data Synthesis with Argument Provenance for Multi-Turn Tool Use

**Venue:** EMNLP 2026 

**Paper:** [SAP: State-Guided Data Synthesis with Argument Provenance for Multi-Turn Tool Use](https://arxiv.org/abs/2609.06124)

**Data & Model:** [HuggingFace](https://huggingface.co/collections/Zichen1024/sap)

---

## 1. Introduction

High-quality multi-turn tool-use data is the key bottleneck for training agentic models. Existing data-synthesis methods focus almost exclusively on **tool-selection dependencies** (which tool to call after which), but largely overlook **argument-level dependencies** — the fact that a tool argument often must carry a value that originated from the initial environment state, a prior tool-call return, or an earlier user message. As a result, models trained on such data may select the correct tool yet fail the task because they fill arguments with fabricated, stale, or weakly grounded values.

This paper makes three contributions:

1. **Argument-provenance diagnostics.** We define cross-turn argument dependency-chain metrics (mean/max chain length, fraction of dependent arguments) and show that existing open-source multi-turn training sets (ToolACE, APIGen-MT) are consistently shallow on both — leaving models prone to errors in long-horizon multi-turn tasks.
2. **SAP** — a state-guided synthesis framework that:
   - dynamically builds a **provenance-annotated FSM** skeleton from raw tool documentation (no hand-curated tool-dependency graph),
   - treats **argument provenance as an explicit constraint**: every argument must declare its causal source (initial state / prior tool output / user message / self-created) before being bound, and every declaration is resolved against the **actually executed history**,
   - validates every tool call **turn by turn against a live executor**, retrying only the offending call instead of discarding the whole trajectory,
   - synthesizes user/assistant messages **post-hoc** once the tool-call sequence is fixed and verified, so all dialogue text is grounded in real returns,
   - injects challenge scenarios (MissParam / MissFunc rewrites) with executor-replay guards to cover deployment failure modes.
3. **SAP-4B**, a 4B model trained by SFT only (no RL) on ~9k SAP-synthesized trajectories, which is highly competitive on both BFCL v4 Multi-Turn and τ²-bench, matching or beating several 4B-8B RL-trained baselines on τ²-bench.

---

## 2. Main Figure (placeholder)
<img width="3985" height="2009" alt="main_1" src="https://github.com/user-attachments/assets/36f5a5d2-097d-4f79-b07d-57d5d6ff8d42" />

- Three stages: (1) FSM skeleton synthesis by $\mathcal{A}_{\text{FSM}}$ with provenance tags;
- (2) per-call planning + executor execution by $\mathcal{A}_{\text{plan}} + \varepsilon$ (two-track output: executor args $\theta^{\text{exec}}$ + provenance metadata $\theta^{\text{prov}}$, parallel grouping, per-call retry);
- (3) post-hoc dialogue synthesis by $\mathcal{A}_{\text{msg}}$.
---

## 3. Method at a Glance

SAP casts multi-turn tool-use data synthesis as a closed loop of **three LLM agents + one executor**:

| Component | Role |
|---|---|
| $\mathcal{A}_{\text{FSM}}$ | Reads summarized tool documentation + initial-state summary → emits a dialogue-phase **FSM** whose edges carry ordered tool lists and per-argument **Provenance Tags** |
| $\mathcal{A}_{\text{plan}}$ | Samples a path over the FSM (weighted random walk, long-tail emphasis), binds each argument against the executed history with provenance metadata, executes via $\varepsilon$ group by group; on failure only the offending call is retried (executor error fed back in-context) |
| $\mathcal{A}_{\text{msg}}$ | After the call sequence is frozen, generates user messages (via must-mention markers) and assistant text responses, all grounded in real returns |
| $\varepsilon$ | Real BFCL v4 / τ²-bench executor backends — every synthesized trajectory is validated against the same simulator used at evaluation |

**Provenance Tag type system** (`S_decl` + runtime): `initial_state` (from $c_0$), `prev_output` (from a prior turn's tool return, bound at runtime), `self_create` (new value introduced this turn), `prev_user_msg` (introduced in an earlier user message via virtual-history remap), and runtime-only `fallback` (recovery binding when a declared upstream is unavailable; original source kept in `fallback_from`).

**Key invariant:** before any argument is committed, its causal source must be declared and — for the three upstream-grounded types — the referenced upstream must exist (or be pre-registered). This shifts cross-turn argument dependencies from a passive side-effect into an explicitly planned, turn-by-turn validated property.

---

## 4. Experimental Results & Explanation

**Setup.** 
- SFT (AdamW, lr 1e-6, batch 128, 10 epochs, verl) on Qwen3-4B-Instruct-2507; ~9k trajectories; 
- synthesis agents: $\mathcal{A}\_{\text{FSM}}$ = Gemini 3.1-pro, $\mathcal{A}\_{\text{plan}}$ = Gemini 3-pro, $\mathcal{A}\_{\text{msg}}$ = Qwen3-235B-A22B.

### 4.1 Main results (Table 1 in paper)

| | BFCL v4 MT Avg | Base | MFn | MPm | LCtx | τ² Avg | Retail | Airline |
|---|---|---|---|---|---|---|---|---|
| Qwen3-4B backbone | 22.1 | 26.5 | 21.0 | 15.5 | 25.5 | 32.2 | 40.4 | 24.0 |
| **SAP-4B (ours)** | **30.4** | **38.0** | **23.0** | **24.5** | **36.0** | **35.1** | **42.1** | **28.0** |
| Δ vs backbone | **+8.3** | +11.5 | +2.0 | +9.0 | +10.5 | **+2.9** | +1.7 | +4.0 |

Key takeaways:

- **SFT only, no RL:** a pure 4B SFT model improves BFCL v4 multi-turn Avg by **+8.3** and τ²-bench Avg by **+2.9** over its backbone. Gains concentrate on Base (+11.5), LongCtx (+10.5), and MissParam (+9.0) — exactly the subsets that stress long-range argument tracking; MissFunc, which requires no argument grounding, shows a small lift (+2.0).
- **Same-scale comparison:** on BFCL v4 MT, SAP-4B is on par with AWM-4B (30.4 vs 30.3) while **substantially outperforming it on τ²-bench (35.1 vs 24.7)**, where long-horizon state-tracking is the bottleneck — the exact capability the provenance constraint targets.
- **Cross-scale comparison:** SAP-4B trails 8B baselines (CM2-8B-RL, ToolACE-2-8B, BitAgent-8B) on BFCL v4 — attributed primarily to scale, not data quality — but on τ²-bench it **matches or exceeds most of them** (35.1 vs 31.7 / 17.5 / 21.7) using only SFT on a 4B backbone with 9k trajectories.
- **Frontier headroom:** closed-source frontier models reach BFCL v4 MT Avg ~60 and τ² Avg ~78, so the open-source 4B regime still has substantial headroom.
- **Out-of-distribution probe:** multi-turn-only training does not regress single-turn performance; SAP-4B yields small consistent gains on both BFCL v4 single-turn splits (Non-live +0.52, Live +0.54).

### 4.2 Ablations (Table 2 in paper)

| Variant | Avg | Base | MFn | MPm | LCtx |
|---|---|---|---|---|---|
| SAP-4B (Full) | **30.4** | 38.0 | 23.0 | 24.5 | 36.0 |
| A1: w/o Provenance Tags | 24.0 (−6.4) | 29.5 | 19.5 | 21.5 | 25.5 |
| A2: w/o rewrite operators Ψ | 28.8 (−1.6) | 37.0 | 21.0 | 22.5 | 34.5 |

- **A1 (Provenance Tags)** is the dominant component: removing the tag declarations costs **−6.4 Avg**, with the sharpest hits on LongCtx (−10.5) and Base (−8.5), confirming that the tag declarations act as a structural prior forcing $\mathcal{A}_{\text{plan}}$ to ground each argument in a verifiable upstream, most useful precisely where long-horizon argument tracking matters.
- **A2 (rewrite operators)** is modest but consistent (−1.6): the operators specifically target deployment failure modes (MissFunc −2.0, MissParam −2.0) while leaving standard multi-turn performance largely intact.

### 4.3 Pipeline-level diagnostics (Appendix)

- Removing the FSM skeleton cuts the executor **pass-rate from 89% → 50%** and raises the fallback rate to 15.75%, the FSM's structure-first prior is the primary guarantor of causal-chain existence.
- Removing only the Provenance Tags keeps pass-rate high (92%) but produces **zero `prev_output` arguments** , all cross-turn dependency structure disappears.
- Disabling per-call retry drops pass-rate to **54%** — the executor-in-the-loop refill is essential.
- Cross-backbone fallback study (500 identical skeletons): Gemini-3.1 Pro lowest fallback 2.02% (chosen for the main pipeline); Qwen3.5-Plus highest 4.82%; DeepSeek-V4-Flash tends to fabricate literals (prev_output 9.2% / self_create 59.4%).


