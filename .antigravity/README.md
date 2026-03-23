# Agent Configuration

Directory containing configuration files for the AI agent (Antigravity/Gemini).

## General Description

This directory stores the rules and contracts that define the agent's behavior when interacting with the repository. It allows for standardizing commits, documentation, and workflows.

## Contents

| File | Type | Description |
| :--- | :--- | :--- |
| `rules.md` | Configuration | Contract of rules for commits and documentation. |

## Defined Rules

The `rules.md` file establishes:

1. **Commit Format**: Semantic tags (`[MODEL]`, `[ARCH]`, `[DOCS]`, etc.).
2. **Auto-Tagging**: Automatic creation of Git tags for model commits.
3. **Language**: All repository content (code, documentation, READMEs) must be in **ENGLISH**.
4. **README Standard**: Structure based on `training/README.md`.

## Usage

The agent automatically reads these rules when starting a session (if configured in `.gemini/rules.md`) or can be referenced manually:

```
Read the rules in .antigravity/rules.md and follow them.
```
