# Codex Windows/WSL Configuration Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give Windows Codex and WSL Codex independent `config.toml` files so Windows-only MCP and plugin paths are never launched by Linux Codex.

**Architecture:** Preserve `/mnt/c/Users/Cyd/.codex/config.toml` as the Windows authority and replace only the WSL-side symlink `/home/cyd/.codex/config.toml` with a regular Linux file. The Linux file retains working model, provider, trust, hook, feature, and TUI settings while omitting Windows-only notification, `node_repl`, desktop marketplace, plugin, Windows sandbox, and browser environment entries.

**Tech Stack:** TOML, Codex CLI 0.147.0, WSL2, SHA-256 checksums.

## Global Constraints

- Back up the Windows configuration before changing the WSL link.
- Do not modify the contents of the Windows configuration.
- Use an explicit timestamped backup and verify checksums.
- The WSL configuration must be a regular file, not a symlink.
- Validation must confirm that WSL reports no configured Windows-only MCP server.

---

### Task 1: Snapshot and split the configuration

**Files:**
- Preserve: `/mnt/c/Users/Cyd/.codex/config.toml`
- Create: `/mnt/c/Users/Cyd/.codex/config.toml.backup-<timestamp>`
- Replace symlink with regular file: `/home/cyd/.codex/config.toml`

**Interfaces:**
- Consumes: the current shared Windows TOML configuration.
- Produces: independent Windows and WSL TOML configuration files.

- [x] **Step 1: Record the source checksum and symlink target**

Run `sha256sum /mnt/c/Users/Cyd/.codex/config.toml` and `readlink /home/cyd/.codex/config.toml`.

- [x] **Step 2: Create a timestamped Windows-side backup**

Copy the Windows file with metadata preserved, then verify the source and backup SHA-256 values are identical.

- [x] **Step 3: Replace only the WSL symlink**

Move the symlink itself aside as `/home/cyd/.codex/config.toml.windows-link-<timestamp>` and create `/home/cyd/.codex/config.toml` as a regular file.

- [x] **Step 4: Populate Linux-safe settings**

Retain the working model/provider, project trust, hook trust state, `js_repl` feature flag, and TUI state. Omit all values containing `C:\\`, `\\\\.\\pipe`, Windows marketplace paths, `[mcp_servers.node_repl]`, `[desktop]`, `[windows]`, Windows-only plugin enablement, and the browser-only shell environment block.

### Task 2: Validate both sides and rollback assets

**Files:**
- Inspect: `/home/cyd/.codex/config.toml`
- Inspect: `/mnt/c/Users/Cyd/.codex/config.toml`
- Inspect: timestamped backup and saved symlink

**Interfaces:**
- Consumes: the split configuration from Task 1.
- Produces: evidence that Linux uses no Windows MCP paths and Windows stayed byte-identical.

- [x] **Step 1: Parse the WSL configuration through Codex**

Run `codex mcp list`; expect `No MCP servers configured yet` and no TOML parse error.

- [x] **Step 2: Check filesystem separation**

Verify `/home/cyd/.codex/config.toml` is a regular file and `/mnt/c/Users/Cyd/.codex/config.toml` remains a regular Windows-side file.

- [x] **Step 3: Verify Windows immutability**

Compare the post-split Windows checksum to the recorded checksum and its timestamped backup.

- [x] **Step 4: Scan the WSL configuration**

Confirm it contains no `C:\\`, named-pipe, `node_repl`, or `openai-bundled` Windows path entries.

- [x] **Step 5: Report restart boundary and rollback path**

Explain that already-running Codex sessions keep their startup tool inventory; validate the clean startup in a new WSL Codex session. Rollback consists of moving the regular WSL config aside and restoring the saved symlink.
