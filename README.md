# VoxLens

**VoxLens** is an experimental computer vision project that converts visible lip movements in video into text. It analyzes close-up footage of a speaker's mouth and uses visual speech recognition to infer what is being said — without relying on audio.

The goal is to explore practical, real-time lip-reading using modern AI and vision models.

## Status

Early and experimental. The repository currently holds project and agent configuration only — there is no implementation yet, and no stack has been committed to. Expect the structure below to grow.

## Repository layout

```
/
├── README.md
├── CONTEXT.md             ← the domain glossary
├── AGENTS.md              ← conventions for agents working in this repo
└── docs/
    └── agents/
        ├── issue-tracker.md
        ├── triage-labels.md
        └── domain.md
```

[`CONTEXT.md`](CONTEXT.md) fixes the project's vocabulary. Terms like _Clip_, _Occlusion_, and _Inferred Text_ mean something specific here — use them as defined rather than reaching for a synonym.

One directory will appear as the project grows, created lazily rather than upfront:

- `docs/adr/` — architecture decision records, one file per decision.

## Working in this repo

[`AGENTS.md`](AGENTS.md) is the entry point for both humans and coding agents. It points at three documents under `docs/agents/`:

| Document | What it settles |
| --- | --- |
| [`issue-tracker.md`](docs/agents/issue-tracker.md) | Work is tracked in this repo's **GitHub Issues** via the `gh` CLI |
| [`triage-labels.md`](docs/agents/triage-labels.md) | The five triage states: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix` |
| [`domain.md`](docs/agents/domain.md) | Single-context layout, and the rule to read `CONTEXT.md` and relevant ADRs before exploring |

When naming a domain concept in an issue, a test, or a proposal, use the term as defined in `CONTEXT.md` rather than a synonym — consistent vocabulary is what keeps the issue tracker searchable as the project grows.

## Getting started

Nothing to run yet. This section will cover setup, dependencies, and how to point the pipeline at a video once there's an implementation to describe.
