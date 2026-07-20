# Replay Launcher 同 Tab 跳转设计

## 目标

消除 Chrome 打开 replay launcher 时因自动提交 `target="_blank"` 而触发的弹窗拦截。launcher 仍然使用必要的 HTML form `POST` 将完整 replay JSON 交给外部 viewer，但提交结果替换当前标签页。

## 设计

`visualization/replay/core.py` 生成的 form 显式使用 `target="_self"`。页面加载后自动提交时，浏览器会在当前标签页导航到 viewer，不创建脚本弹窗。`show_replay()`、CLI 参数和外部 viewer endpoint 保持不变。

不采用新 tab 点击按钮，因为它增加用户操作；不采用本地 POST 代理，因为它引入额外服务和资源路径风险。由于 viewer 接收的是 POST body，launcher 仍是必要的临时中间页，但用户不会看到需要手动处理的弹窗。

## 测试与文档

- launcher 测试断言 form 使用 `target="_self"`，且不含 `target="_blank"`。
- 保留 POST method、viewer action、`json` 字段和完整 payload 的现有断言。
- `visualization/README.md`、`AGENTS.md` 和 README 说明 launcher 会在当前标签页完成跳转，不再创建新窗口。
