# Publish npm Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `publish-npm.yml` GitHub Actions workflow that publishes `@ecmwf/tensogram` to the public npm registry on manual dispatch, idempotently skipping if the version is already published.

**Architecture:** Single-job workflow on the ECMWF self-hosted Linux runner using the existing CI Docker image (which already has Rust and wasm-pack). Builds WASM then TypeScript, checks npmjs.com for the current version, and publishes only if not already present. Auth via `NPMJS_API_TOKEN` secret in the `npm` GitHub environment.

**Tech Stack:** GitHub Actions, wasm-pack, Node.js, npm, TypeScript.

---

### Task 1: Create the publish-npm workflow file

**Files:**
- Create: `.github/workflows/publish-npm.yml`

- [ ] **Step 1: Create the workflow file**

Create `.github/workflows/publish-npm.yml` with the following content:

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
    name: Publish @ecmwf/tensogram
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
            "https://registry.npmjs.org/%40ecmwf%2Ftensogram/$VERSION")
          if [ "$HTTP_STATUS" = "200" ]; then
            echo "already_published=true" >> "$GITHUB_OUTPUT"
            echo "@ecmwf/tensogram@$VERSION is already on npmjs.com — skipping"
          else
            echo "already_published=false" >> "$GITHUB_OUTPUT"
            echo "@ecmwf/tensogram@$VERSION not found on npmjs.com — will publish"
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

Note on the URL encoding: `@ecmwf/tensogram` becomes `%40ecmwf%2Ftensogram` in the registry URL. The `@` and `/` in the scope must be percent-encoded when used in a path segment.

- [ ] **Step 2: Verify YAML is valid**

Run:
```bash
python3 -c "import yaml; yaml.safe_load(open('.github/workflows/publish-npm.yml'))" && echo "YAML OK"
```
Expected output: `YAML OK`

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/publish-npm.yml
git commit -m "feat: add GitHub Actions workflow to publish @ecmwf/tensogram to npm"
```

---

## Prerequisites (human action required before first run)

1. In the GitHub repo settings, create a new **Environment** named `npm`.
2. Add a secret named `NPMJS_API_TOKEN` to that environment — value is a publish-scoped npm access token with rights to publish under the `@ecmwf` scope on npmjs.com.
