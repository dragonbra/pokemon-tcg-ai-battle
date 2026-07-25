# Git LFS 全历史迁移与成果级备份设计

**日期：** 2026-07-25  
**仓库：** `dragonbra/pokemon-tcg-ai-battle`  
**状态：** 已批准

## 1. 背景

仓库目前约 1 GiB，普通 Git 对象库约 520 MiB。`submission/` 和 `evaluation/` 中存在多个版本的模型权重与正式提交压缩包；仓库尚未配置 `.gitattributes`，本机 Git LFS 当前也不可用。部分二进制制品已经进入普通 Git 历史，因此仅从当前版本启用 LFS 无法释放普通 Git 历史负担。

本仓库当前只有仓库所有者使用。所有者已明确授权重写完整历史，并在验证通过后 force-push GitHub 分支和 tags。

## 2. 目标与非目标

### 2.1 目标

1. 将所有历史版本中的以下文件迁移到 Git LFS：
   - `*.bin`
   - `*.pt`
   - `*.ckpt`
   - `submission/dist/*.tar.gz`
2. 保留原有文件路径及基于 commit/tag 的版本管理体验。
3. 使普通 Git 历史只保存 LFS pointer，二进制内容由 GitHub LFS 保存。
4. 在重写历史前创建仓库外的完整恢复备份。
5. 验证迁移前后当前有效制品的内容哈希一致，并验证全部分支、tags 和 LFS 对象可恢复。
6. force-push 后执行一次干净 clone 恢复演练。

### 2.2 非目标

本次不备份或迁移以下内容：

- 完整训练过程的所有 checkpoint；只有已经纳入仓库且匹配上述规则的 checkpoint 才迁移。
- dataset、official data、replay、trace、cache 或其他被忽略的临时文件。
- 淘汰且从未提交到 Git 的候选制品。
- 将 GitHub LFS 当成唯一灾难恢复介质。
- 自动删除迁移前备份。

## 3. 成果级恢复边界

采用“成果级恢复”策略。灾难恢复后必须能够取得：

- Git 完整历史、保留的分支和 tags；
- 源代码、配置、环境定义和正式报告；
- 已提交的最终模型权重；
- 已提交的正式 Kaggle submission 压缩包；
- 与正式成果关联的 commit/tag、指标和校验信息。

不承诺从全部中间状态恢复训练，也不承诺在原始数据不可重新获得时重现训练过程。

## 4. 存储分层

### 4.1 普通 Git

保存：

- 源代码和测试；
- 配置和环境文件；
- 文档、评测报告和小型指标文件；
- `.gitattributes`；
- LFS pointer；
- 可审计的 SHA-256 清单和迁移记录（不得包含凭据）。

### 4.2 GitHub Git LFS

保存匹配迁移规则的历史二进制对象。GitHub LFS 提供与 Git commit/tag 绑定的版本检出能力，但受账户存储和下载带宽配额约束；同一大文件每个不同版本都按完整对象占用存储。

### 4.3 仓库外迁移备份

force-push 前在仓库目录之外创建带时间戳的迁移备份目录，至少包含：

- 迁移前的 `git bundle --all` 或等价 mirror；
- 所有 refs 的文本快照；
- 当前 HEAD 与远端 URL；
- 迁移目标文件清单及当前工作树中有效文件的 SHA-256；
- Git LFS 迁移前后的审计输出。

备份不得放回本仓库，也不得在本次操作中自动删除。若本机剩余空间不足以安全容纳备份和迁移临时空间，操作必须停止。

## 5. 迁移规则

`.gitattributes` 使用 Git LFS 标准属性：

```gitattributes
*.bin filter=lfs diff=lfs merge=lfs -text
*.pt filter=lfs diff=lfs merge=lfs -text
*.ckpt filter=lfs diff=lfs merge=lfs -text
submission/dist/*.tar.gz filter=lfs diff=lfs merge=lfs -text
```

迁移采用 `git lfs migrate import --everything`，include 范围与上述模式保持一致。执行前应核对 `git lfs migrate info --everything` 估算对象数量与体积。

不使用覆盖所有 `*.tar.gz` 的宽泛规则；只有正式 submission 路径中的压缩包进入 LFS。

## 6. 执行流程

1. **前置审计**
   - 确认工作树状态，并保护现有未跟踪文件；
   - fetch 全部远端 refs；
   - 记录本地/远端分支、tags 和 HEAD；
   - 确认远端仍为预期 GitHub 仓库；
   - 检查磁盘空间和 GitHub LFS 可用条件；
   - 安装并验证 `git-lfs`。

2. **建立迁移前备份**
   - 在仓库外生成 bundle/mirror、refs 快照和 SHA-256 清单；
   - 使用 `git bundle verify` 验证备份；
   - 从 bundle 临时恢复或至少验证关键 refs 可读。

3. **隔离现有未跟踪文件**
   - 不提交、不删除、不覆盖已有的未跟踪设计/计划文件；
   - 历史迁移前将工作树整理到 Git LFS 可以安全执行的状态；
   - 如需临时移动未跟踪文件，记录原路径并在迁移后原样恢复。

4. **配置并迁移历史**
   - 配置 `.gitattributes`；
   - 对所有 refs 执行 LFS 历史迁移；
   - 不在验证通过前推送。

5. **本地验证**
   - `git fsck --full` 通过；
   - `git lfs fsck` 通过；
   - `git lfs ls-files --all` 覆盖目标制品；
   - 历史中目标路径的 blob 为 LFS pointer；
   - 当前检出文件仍为真实二进制内容，而非裸 pointer；
   - 当前关键制品迁移前后 SHA-256 一致；
   - 预期分支和 tags 均存在；
   - 已知未跟踪文件保持原样。

6. **发布迁移结果**
   - 先推送全部 LFS 对象；
   - 使用 `--force-with-lease` 或基于已记录远端状态的等价保护机制改写分支；
   - force-push tags；
   - 若远端状态相对审计时发生变化，立即停止，不覆盖新引用。

7. **远端恢复演练**
   - 在仓库外新建干净目录，从 GitHub clone；
   - 验证默认分支、远端分支和 tags；
   - 验证 LFS 下载、`git lfs fsck` 与关键文件 SHA-256；
   - 记录恢复演练结果。

## 7. 错误处理与停止条件

遇到以下任一情况必须停止，且不得 force-push：

- 远端 URL、所有者或仓库名与预期不符；
- 存在未预期的协作者活动、远端 refs 变化或分支保护阻止安全迁移；
- Git LFS 安装、认证或上传失败；
- 预计 LFS 数据量明显超出可接受额度；
- 磁盘空间不足；
- bundle、Git fsck、LFS fsck 或哈希校验失败；
- 迁移丢失分支或 tags；
- 已知未跟踪文件发生内容变化。

若 force-push 后干净 clone 验证失败，应保留远端现场和迁移前备份，诊断原因；未经新的明确判断，不进行第二次破坏性改写。迁移前 bundle 是最终回滚来源。

## 8. 验收标准

实施完成必须同时满足：

1. `.gitattributes` 精确覆盖四类目标路径。
2. 所有历史 refs 中匹配目标模式的大文件均由 Git LFS 管理。
3. 当前版本关键制品迁移前后 SHA-256 一致。
4. Git 和 LFS 完整性检查均通过。
5. 原有预期分支与 tags 均保留。
6. 已知未跟踪文件未被提交、删除或修改。
7. GitHub 上改写后的 `main` 和其他既有远端分支可正常 clone。
8. 干净 clone 能下载并读取 LFS 制品。
9. 迁移前完整备份仍保留在仓库之外，且经过验证。
10. 最终报告列出新 HEAD、备份位置、迁移对象规模、验证命令及结果。

## 9. 后续操作约定

- 新增正式模型或 submission 包时正常执行 `git add`、`git commit`、`git push`，Git LFS 自动处理匹配文件。
- 提交前用 `git check-attr filter -- <path>` 确认大文件匹配 LFS。
- 不把所有训练 checkpoint 默认加入仓库；只有明确晋升为成果制品的 checkpoint 才进入受版本控制路径。
- 定期检查 GitHub LFS 存储和带宽使用情况。
- GitHub/Git LFS 之外的第二份长期备份属于后续独立工作；本次迁移前 bundle 仅作为迁移安全网，不替代长期异地备份。
