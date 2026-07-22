# Kaggle CLI 快速开始

Kaggle CLI 安装在本机 Homebrew Python 的真实环境中，不需要激活仓库虚拟环境。

```bash
cd /Users/hejinyu/Documents/repos/pokemon-tcg-ai-battle
kaggle --version
```

当前验证版本为 `Kaggle CLI 2.2.3`，可执行文件位于 `/opt/homebrew/bin/kaggle`。

Kaggle 登录不会自动执行。准备好后使用官方 OAuth：

```bash
kaggle auth login
```

登录后先做只读验证，不下载、不提交：

```bash
kaggle competitions list --group entered
kaggle competitions pages pokemon-tcg-ai-battle
kaggle competitions topics list pokemon-tcg-ai-battle -s hot --page-size 10
```

加入比赛并接受规则后，才下载比赛数据：

```bash
mkdir -p data/competition
kaggle competitions download pokemon-tcg-ai-battle -p data/competition
```

Simulation Competition 常用命令：

```bash
kaggle competitions topics list pokemon-tcg-ai-battle -s top --page-size 20
kaggle competitions topic-messages pokemon-tcg-ai-battle <topic-id>
kaggle competitions pages pokemon-tcg-ai-battle --content
kaggle competitions submissions pokemon-tcg-ai-battle
kaggle competitions episodes <submission-id>
kaggle competitions replay <episode-id> -p replays
kaggle competitions logs <episode-id> <agent-index> -p logs
```

不要把凭据放进仓库。`.env`、`.kaggle/` 和比赛原始数据已经加入 Git 忽略规则。
