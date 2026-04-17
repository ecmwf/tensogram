# Address Pull Request Comments

Review and resolve all feedback on a GitHub pull request.

## Arguments

Provide the PR number as argument, e.g. `/address-pr-comments 32`
If no number given, detect the current branch's open PR.

## Process

### 1. Fetch Reviews
```bash
gh pr view <NUMBER> --json reviews,comments
gh api repos/{owner}/{repo}/pulls/<NUMBER>/comments
gh api repos/{owner}/{repo}/pulls/<NUMBER>/reviews
```

### 2. Analyze Each Comment

For each review comment or inline comment:
- Read the full context of the code being discussed
- Understand the reviewer's concern in light of this project's philosophy:
  - Simplicity over complexity
  - No backwards compatibility constraints (pre-public software)
  - Proper fixes over workarounds
- If the intent is ambiguous, ask the user for clarification before acting

### 3. Address Issues

For each actionable comment:
- Fix the code as requested (or propose a better alternative with justification)
- Update documentation if the change affects public API or behavior
- Add or update tests if the comment identified a gap

### 4. Push and Verify

- Commit changes with a clear message referencing the review feedback
- Push to update the PR
- Wait for CI to finish:
  ```bash
  gh pr checks <NUMBER> --watch
  ```
- If CI fails, fix and push again

### 5. Iterate

Continue the loop until:
- All review comments have been addressed
- CI is green
- Summarize what was changed in response to each comment

### 6. Report

Provide a summary mapping each reviewer comment to the action taken.
