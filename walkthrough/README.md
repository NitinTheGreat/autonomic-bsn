# Walkthrough

One document per phase, written **at the end of that phase**. Each is a
handoff: enough context for a fresh reader (human or a new Claude session with
no memory of the work) to understand what was built, what was actually
verified, what is still open, and how to start the next phase.

## Convention

| File | Phase |
|---|---|
| [`phase1.md`](phase1.md) | Phase 1 — foundational de-risking |
| [`phase2.md`](phase2.md) | Phase 2 — pluggable data layer & dataset explorer |
| [`phase3.md`](phase3.md) | Phase 3 — failure injection lab |
| … | … |

## What every phase document must contain

1. **Goal** — the question the phase existed to answer.
2. **What was built** — files, with what each is responsible for.
3. **What was verified, and how** — separating *code verified against mocks or
   fixtures* from *claims verified against real data or a live model*. Never
   blur the two.
4. **Results** — the actual numbers, including failures.
5. **Decisions and their rationale** — especially anywhere the implementation
   deliberately departs from the original brief.
6. **Known gaps / open questions** — what is still unproven.
7. **How to start the next phase** — concrete entry points and constraints
   carried forward.

## Ground rule

These documents must stay **honest about what has not been proven**. A
walkthrough that reads as though everything succeeded, when a check was
actually blocked or skipped, is worse than no walkthrough — the next session
will build on a false premise. State blocked work as blocked.
