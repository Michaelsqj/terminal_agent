# 🤓 Terminal-Agent-RL: Terminal Agent evaluation and training based on CAMEL




## Getting Started 🎯
### Installation

```bash
# Clone the repository
git clone --recurse-submodules https://github.com/Michaelsqj/terminal_agent.git

pip install uv
# Install submodules and dependencies
uv pip install -e external/camel
uv pip install -e external/terminal-bench

uv pip install -e external/rllm/verl

# Before installing rllm, making sure rust compiler is available
rustc --version
# [Optional], install rust compiler
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
rustc --version

# Install rllm
uv pip install -e external/rllm
```

🚧 **TODO:** Create an all-in-one installation method with `pyproject.toml` and `uv.lock`

### Make Changes for `<submodule>`

#### 1. Go into the `<submodule>` folder:

```bash
cd external/<submodule>
```
#### 2. Make changes and push them to the `<submodule>`:

```bash
git checkout -b my-feature-branch  # optional, if you want to use a new branch
git add .
git commit -m "My changes to submodule"
git push origin my-feature-branch  # or push to the appropriate branch
```

#### 3. Go back to the main repo and commit the submodule pointer update:

```bash
cd ../..  # back to the root
git add external/<submodule>
git commit -m "Updated submodule pointer"
git push origin main  # or your main branch
```