# Skill routing

- CRITICAL: Always prefer the LSP tool over Grep/Read for code navigation. Use it to find definitions, references, and workspace symbols.

When the user's request matches an available skill, ALWAYS invoke it using the Skill
tool as your FIRST action. Do NOT answer directly, do NOT use other tools first.
The skill has specialized workflows that produce better results than ad-hoc answers.

# Design & Purpose

- README.md -- entry level generic information
- plans/DESIGN.md -- purpose and design of the library
- plans/STYLE.md -- code style conventions
- plans/PLAN.md -- initial plan iteration 0 before further improvements added
- plans/DONE.md -- keep updated with implementation progress
- plans/IMPROVEMENTS.md -- future improvements to consider when implementing, if opportune pick from there
- plans/TODO.md -- long term TODO list of features that may or not be implemented

Follow docs/DESIGN.md principles and docs/STYLE.md conventions in all code.

# Build / lint / test (required before marking done)

- Format: `cargo fmt`
- Lint: `cargo clippy --workspace --all-targets --all-features -- -D warnings`
- Test: `cargo test --workspace`

# Tracking Work Done

Please keep track of what is implemented in plans/DONE.md. Keep updating it for all code changes.

# Documentation

Create and maintain documentation under docs/. Easy to follow by average tech person, with well separated topics.
- Use mdbook 
- Add mermaid diagrams when necessary
- Add examples when it becomes hard to follow
- Especially note the edge cases

# Examples

Create and maintain a sub-dir examples/<lang> populated with examples of caller code showing how to use interfaces:
- create sub-dir per language of the caller: Rust, C++, Python
- examplify the most common cases
- show how to use all API functions
