# Coding Style

Follow `DESIGN.md`. When in doubt, read the existing code.

## Principles

1. **Deep modules.** Hide complexity behind an API with a small surface
2. **Design errors out of existence.** Idempotent ops, never panic, always return error codes.
3. **Stateless** No global state

## Functions

Prefer short functions (10–50 lines)

## Comments

Try hard to explain *why*, not *what*
