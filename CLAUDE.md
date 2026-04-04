# Skill routing

- CRITICAL: Always prefer the LSP tool over Grep/Read for code navigation. 
    - Use it to find definitions, references, and workspace symbols.
- When the user's request matches an available skill:
    - ALWAYS invoke it using the Skill tool as your FIRST action. 
    - Do NOT answer directly, do NOT use other tools first. 
    - The skill has specialized workflows that produce better results than ad-hoc answers.
- IMPORTANT: Before you do any work, mention how you would verify that work
- NOTE: When the user asks for a "second pass", "third pass", treat it as shorthand for:
  - simplification opportunities,
  - naming/comment/doc quality review,
  - edge-case/logical regression scan,
  - no panics in rust code,
  - all documentation up-to-date with changes,
  - and running required formatter/lint/tests.

# Design & Purpose

- README.md -- entry level generic information
- plans/DESIGN.md -- purpose and design of the library
- plans/STYLE.md -- code style conventions
- plans/PLAN.md -- initial plan iteration 0 before further improvements added
- plans/DONE.md -- keep updated with implementation progress
- plans/TODO.md -- long term TODO list of features that may or not be implemented

Follow docs/DESIGN.md principles and docs/STYLE.md conventions in all code.

# Build / lint / test (required before marking done)

- Build: `cargo build --workspace`
- Format: `cargo fmt`
- Lint: `cargo clippy --workspace --all-targets --all-features -- -D warnings`
- Test: `cargo test --workspace`

# Version control
- Git
- IMPORTANT: versions are tagged using Semantic Versioning form 'MAJOR.MINOR.MICRO', NEVER update MAJOR unless users says so. Increment MINOR for new features. MICRO for bugfixes and documentation updates.
- NEVER prepend git tag or releases with 'v'

# Tracking Work Done

Keep track of implementations in plans/DONE.md for all code changes.

# Documentation

Create and maintain documentation under docs/. 
- Easy to follow by average tech person, with well separated topics.
- Use mdbook 
- Add mermaid diagrams when necessary
- Add examples when it becomes hard to follow
- Especially note the edge cases

# Examples

Create and maintain a sub-dir examples/<lang> 
- 1 sub-dir per supported language of the caller Rust, C++, Python
- Populate with examples of caller code showing how to use interfaces
- examplify the most common cases
- show how to use all API functions
