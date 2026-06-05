# 如何向 vLLM Ascend 提交代码与文档

本文介绍从注册 GitHub 账号到创建 Pull Request（PR）的完整流程，适用于首次向 [vllm-ascend](https://github.com/vllm-project/vllm-ascend) 仓库贡献代码或文档的开发者。

## 1. 自行准备 GitHub 账号

vLLM Ascend 使用 GitHub 托管代码并接收社区贡献。开始前请完成以下准备：

1. 打开 [https://github.com/signup](https://github.com/signup) 注册 GitHub 账号。
2. 验证邮箱，并建议开启两步验证（Settings → Password and authentication → Two-factor authentication）。
3. 完善个人资料（可选）：头像、Display name，便于维护者在 Review 时识别贡献者。
4. 阅读并了解 [Developer Certificate of Origin (DCO)](https://developercertificate.org/)：向本仓库提交 PR 时，**每个 commit 必须包含 `Signed-off-by` 行**，表示你同意 DCO 条款。使用 `git commit -s` 可自动添加该签名。

:::{note}
本文中的「GitHub 账号」即参与开源协作所需的代码托管账号。若你已有 GitHub 账号，可直接进入下一步。
:::

## 2. Fork vllm-ascend 代码仓

Fork 会在你的 GitHub 账号下创建一份 vllm-ascend 的独立副本，所有本地修改先推送到 **你自己的 Fork**，再通过 PR 合并到上游仓库。

1. 在浏览器中打开上游仓库：[https://github.com/vllm-project/vllm-ascend](https://github.com/vllm-project/vllm-ascend)
2. 点击页面右上角的 **Fork** 按钮。
3. 选择 Fork 到的目标账号（通常为你自己的 GitHub 账号），确认创建。
4. Fork 完成后，你的仓库地址形如：

   ```text
   https://github.com/<你的用户名>/vllm-ascend
   ```

:::{important}
请勿直接向上游 `vllm-project/vllm-ascend` 的 `main` 分支 push。社区约定是：**先 push 到自己的 Fork，再向上游发起 PR**。
:::

## 3. 本地获取 Fork 的代码并配置 GitHub SSH 公钥

### 3.1 生成 SSH 密钥（若尚未配置）

在本地终端执行：

```bash
# 将邮箱替换为你的 GitHub 注册邮箱
ssh-keygen -t ed25519 -C "your_email@example.com"
```

按提示选择密钥保存路径（默认 `~/.ssh/id_ed25519`）并设置 passphrase（可选但推荐）。

启动 ssh-agent 并添加私钥：

```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
```

### 3.2 将公钥添加到 GitHub

1. 查看并复制公钥内容：

   ```bash
   cat ~/.ssh/id_ed25519.pub
   ```

2. 打开 GitHub：**Settings → SSH and GPG keys → New SSH key**
3. Title 填写便于识别的名称（如 `work-laptop`），Key 粘贴公钥全文，保存。

验证 SSH 连接：

```bash
ssh -T git@github.com
```

首次连接时输入 `yes` 确认；成功时会看到类似 `Hi <username>! You've successfully authenticated...` 的提示。

### 3.3 Clone 自己的 Fork

```bash
# 将 <你的用户名> 替换为实际 GitHub 用户名
git clone git@github.com:<你的用户名>/vllm-ascend.git
cd vllm-ascend
```

建议添加上游远程仓库，便于后续同步最新代码：

```bash
git remote add upstream git@github.com:vllm-project/vllm-ascend.git
git remote -v
```

预期输出包含 `origin`（你的 Fork）和 `upstream`（官方仓库）。

### 3.4 创建功能分支

不要直接在 `main` 上开发。从最新的上游 `main` 拉取并创建分支：

```bash
git fetch upstream
git checkout main
git merge upstream/main
git checkout -b feature/my-change
```

将 `feature/my-change` 替换为有意义的英文分支名，例如 `docs/add-tuning-guide` 或 `fix/model-runner-padding`。

### 3.5 配置开发环境（可选但推荐）

贡献代码或文档前，建议参考 [Contributing](./index.md) 配置本地 lint 与测试环境：

```bash
pip install -r requirements-lint.txt
bash format.sh
```

文档贡献还可参考 [Doc writing guide](./doc_writing.md)。

## 4. 修改内容后提交到自己的代码仓

### 4.1 修改并检查

完成代码或文档修改后，在提交前运行格式与 lint 检查：

```bash
bash format.sh ci
```

若 `markdownlint` 等工具改动了文件，需重新 `git add` 后再提交。

### 4.2 提交 Commit（必须 Sign-off）

本项目要求 commit 遵循 [Conventional Commits](https://www.conventionalcommits.org/) 格式，并带 `-s` 签名：

```bash
git add .
git commit -s -m "docs: add contribution submission guide" -m "Describe what changed and why in the body."
```

**Commit 示例：**

```text
fix(model_runner): correct padding token handling

- Fixes token padding that caused incorrect attention masks
- Addresses issue #1234

Signed-off-by: Your Name <your.email@example.com>
```

常用 type：`feat`、`fix`、`perf`、`refactor`、`test`、`docs`、`chore`。

### 4.3 推送到自己的 Fork

```bash
git push -u origin feature/my-change
```

若分支已存在，后续修改可执行：

```bash
git push origin feature/my-change
```

### 4.4 与上游保持同步（可选）

PR 评审期间若上游 `main` 有更新，可在本地 rebase 或 merge：

```bash
git fetch upstream
git checkout feature/my-change
git rebase upstream/main
# 若有冲突，解决后：git add . && git rebase --continue
git push -f origin feature/my-change   # rebase 后需 force push 到你的 Fork
```

:::{warning}
仅对你的 **Fork** 使用 `git push -f`。不要对 `vllm-project/vllm-ascend` 强制推送。
:::

## 5. 如何创建 Pull Request

### 5.1 在 GitHub 网页上创建 PR

1. 打开你的 Fork：`https://github.com/<你的用户名>/vllm-ascend`
2. 推送分支后，页面通常会显示 **Compare & pull request**，点击即可；或手动进入 **Pull requests → New pull request**。
3. 设置 PR 方向：
   - **base repository**：`vllm-project/vllm-ascend`，**base** 分支一般为 `main`
   - **head repository**：`<你的用户名>/vllm-ascend`，**compare** 分支为你刚 push 的功能分支
4. 填写 PR 标题与描述。

### 5.2 PR 标题格式

标题建议使用：`[Type][Module] Description`

| 前缀 | 含义 |
| --- | --- |
| `[Doc]` | 文档 |
| `[BugFix]` | 缺陷修复 |
| `[Feat]` | 新功能 |
| `[CI]` | CI / 构建 |
| `[Test]` | 测试 |
| `[Misc]` | 其他 |

示例：`[Doc][Misc] Add contribution submission guide`

### 5.3 填写 PR 描述

请按仓库 [PR 模板](https://github.com/vllm-project/vllm-ascend/blob/main/.github/PULL_REQUEST_TEMPLATE.md) 填写，至少包括：

1. **What this PR does / why we need it?** — 说明改动内容与动机；若修复 Issue，可写 `Fixes #123`。
2. **Does this PR introduce _any_ user-facing change?** — 纯文档更新一般填「否」或说明仅文档变更。
3. **How was this patch tested?** — 说明测试方式，例如：
   - 运行 `bash format.sh ci`
   - 单元测试：`pytest -sv tests/ut/...`
   - 文档构建：本地 Sphinx / 依赖 CI 通过

### 5.4 等待 CI 与 Code Review

PR 创建后，GitHub Actions 会自动运行 lint、测试等检查。请确保：

- 所有 required checks 通过（绿色）
- 按 Review 意见修改后，在同一分支继续 commit 并 push，PR 会自动更新
- 不要自行 merge；由维护者在 Review 通过后合并

### 5.5 使用 GitHub CLI 创建 PR（可选）

若已安装 [GitHub CLI](https://cli.github.com/)：

```bash
gh auth login
gh pr create \
  --repo vllm-project/vllm-ascend \
  --base main \
  --head <你的用户名>:feature/my-change \
  --title "[Doc][Misc] Add contribution submission guide" \
  --body-file pr_body.md
```

其中 `pr_body.md` 为按模板写好的描述文件。

## 常见问题

| 问题 | 建议 |
| --- | --- |
| DCO 检查失败 | 使用 `git commit -s` 重新提交，或对已有 commit 执行 `git commit --amend -s` 后 force push 到 Fork |
| markdownlint / ruff 失败 | 本地运行 `bash format.sh ci` 修复后重新提交 |
| PR 包含过多无关 commit | 使用 `git rebase -i upstream/main` 整理 commit，或 squash 为一个清晰 commit |
| 没有权限 push 到上游 | 正常情况；只需 push 到自己的 Fork 并创建 PR |

## 延伸阅读

- [Contributing（开发与测试）](./index.md)
- [Doc writing guide（文档写作）](./doc_writing.md)
- [AGENTS.md（贡献规范)](https://github.com/vllm-project/vllm-ascend/blob/main/AGENTS.md)
- [vLLM 上游贡献说明](https://docs.vllm.ai/en/latest/contributing/overview.html)
