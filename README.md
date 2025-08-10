# 🤓 Terminal-Agent-RL: Terminal Agent evaluation and training based on CAMEL




## Getting Started 🎯
### Installation

```bash
# Clone the repository
git clone --recurse-submodules https://github.com/Michaelsqj/terminal_agent.git

```


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