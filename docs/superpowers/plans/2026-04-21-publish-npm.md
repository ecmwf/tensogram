# Publish npm Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename the TypeScript package from `@ecmwf/tensogram` to `@ecmwf.int/tensogram` and add a `publish-npm.yml` GitHub Actions workflow that publishes it to npmjs.com on manual dispatch, skipping idempotently if the version is already published.

**Architecture:** Two tasks: (1) rename the package across the repo, regenerating lock files; (2) create the workflow file. The workflow runs on the ECMWF self-hosted Linux runner using the existing CI Docker image (Rust + wasm-pack already installed), builds WASM then TypeScript, checks npmjs.com for the current version, and publishes only if absent. Auth via `NPMJS_API_TOKEN` secret in the `npm` GitHub environment.

**Tech Stack:** GitHub Actions, wasm-pack, Node.js, npm, TypeScript.

---

### Task 1: Rename package from `@ecmwf/tensogram` to `@ecmwf.int/tensogram`

**Files:**
- Modify: `typescript/package.json`
- Modify: `tensoscope/package.json`
- Modify: `examples/typescript/package.json`
- Regenerate: `typescript/package-lock.json`, `tensoscope/package-lock.json`, `examples/typescript/package-lock.json`
- Bulk find-replace: all 40 files containing `@ecmwf/tensogram`

- [x] **Step 1: Rename via sed across the repo**

Run from the repo root (dry-run first to review):
```bash
grep -rl '@ecmwf/tensogram' . \
  --exclude-dir='.git' \
  --exclude-dir='target' \
  --exclude-dir='node_modules' \
  | sort
```
Then apply:
```bash
grep -rl '@ecmwf/tensogram' . \
  --exclude-dir='.git' \
  --exclude-dir='target' \
  --exclude-dir='node_modules' \
  | xargs sed -i 's|@ecmwf/tensogram|@ecmwf.int/tensogram|g'
```

- [x] **Step 2: Verify the rename in the three package.json files**

```bash
grep '"name"' typescript/package.json tensoscope/package.json examples/typescript/package.json
```
Expected:
```
typescript/package.json:  "name": "@ecmwf.int/tensogram",
tensoscope/package.json:  "name": "tensoscope",
examples/typescript/package.json:  "name": "tensogram-typescript-examples",
```
(Only `typescript/package.json` carries the scoped name; the others reference it as a dependency.)

Check that dependencies were updated:
```bash
grep 'ecmwf' tensoscope/package.json examples/typescript/package.json
```
Both should now show `@ecmwf.int/tensogram`.

- [x] **Step 3: Regenerate lock files**

```bash
cd typescript && npm install --package-lock-only && cd ..
cd tensoscope && npm install --package-lock-only && cd ..
cd examples/typescript && npm install --package-lock-only && cd ..
```

- [x] **Step 4: Verify no stale references remain**

```bash
grep -r '@ecmwf.int/tensogram' . \
  --exclude-dir='.git' \
  --exclude-dir='target' \
  --exclude-dir='node_modules'
```
Expected: no output.

- [x] **Step 5: Commit**

```bash
git add \
  typescript/package.json typescript/package-lock.json \
  tensoscope/package.json tensoscope/package-lock.json \
  examples/typescript/package.json examples/typescript/package-lock.json
git add $(git diff --name-only)
git commit -m "feat: rename npm package to @ecmwf.int/tensogram"
```

---

### Task 2: Create the publish-npm workflow file

**Files:**
- Create: `.github/workflows/publish-npm.yml`

- [ ] **Step 1: Create the workflow file**

Create `.github/workflows/publish-npm.yml` with this content:

```yaml
name: Publish TypeScript package to npm

on:
  workflow_dispatch:

env:
  CARGO_TERM_COLOR: always
  TMPDIR: ${{ github.workspace }}/.tmp
  NODE_VERSION: "22.22.2"

jobs:
  publish:
    name: Publish @ecmwf.int/tensogram
    runs-on: [self-hosted, Linux, platform-builder-docker-xl, platform-builder-Ubuntu-22.04]
    environment: npm
    container:
      image: eccr.ecmwf.int/tensogram/ci:1.2.0
      credentials:
        username: ${{ secrets.ECMWF_DOCKER_REGISTRY_USERNAME }}
        password: ${{ secrets.ECMWF_DOCKER_REGISTRY_ACCESS_TOKEN }}
    steps:
      - uses: actions/checkout@v4
        with:
          submodules: true

      - run: mkdir -p "$TMPDIR"

      - name: Install Node.js
        run: |
          curl -fsSL "https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-x64.tar.xz" \
            | tar xJ --strip-components=1 -C /usr/local

      - name: Build WASM
        run: |
          wasm-pack build rust/tensogram-wasm --release \
            --target web --out-dir ../../typescript/wasm --out-name tensogram_wasm

      - name: Install TypeScript deps
        working-directory: typescript
        run: npm ci || npm install --no-audit --no-fund

      - name: Build distributable
        working-directory: typescript
        run: npx tsc

      - name: Check if version already published
        id: version-check
        run: |
          VERSION=$(node -e "process.stdout.write(require('./typescript/package.json').version)")
          echo "version=$VERSION" >> "$GITHUB_OUTPUT"
          HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" \
            "https://registry.npmjs.org/%40ecmwf.int%2Ftensogram/$VERSION")
          if [ "$HTTP_STATUS" = "200" ]; then
            echo "already_published=true" >> "$GITHUB_OUTPUT"
            echo "@ecmwf.int/tensogram@$VERSION is already on npmjs.com — skipping"
          else
            echo "already_published=false" >> "$GITHUB_OUTPUT"
            echo "@ecmwf.int/tensogram@$VERSION not found on npmjs.com — will publish"
          fi

      - name: Publish to npm
        if: steps.version-check.outputs.already_published == 'false'
        working-directory: typescript
        env:
          NODE_AUTH_TOKEN: ${{ secrets.NPMJS_API_TOKEN }}
        run: |
          echo "//registry.npmjs.org/:_authToken=${NODE_AUTH_TOKEN}" > .npmrc
          npm publish --access public
```

- [ ] **Step 2: Verify YAML is valid**

```bash
python3 -c "import yaml; yaml.safe_load(open('.github/workflows/publish-npm.yml'))" && echo "YAML OK"
```
Expected: `YAML OK`

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/publish-npm.yml
git commit -m "feat: add GitHub Actions workflow to publish @ecmwf.int/tensogram to npm"
```

---

## Prerequisites (human action required before first run)

1. Verify the `@ecmwf.int` scope is registered and owned on npmjs.com.
2. In the GitHub repo settings, create a new **Environment** named `npm`.
3. Add a secret named `NPMJS_API_TOKEN` to that environment — value is a publish-scoped npm access token with rights to publish under the `@ecmwf.int` scope on npmjs.com.
