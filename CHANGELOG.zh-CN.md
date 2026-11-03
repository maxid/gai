[English](CHANGELOG.md) | 简体中文

# 变更日志

本文件记录本项目的重要变更。

## [Unreleased]

- `anthropic`：Messages API 的 `gai.Provider`，支持 chat 与流式；system 提升到顶层、工具调用是 content block、工具结果是 user 消息，流式解析按事件类型（`content_block_start/delta/stop`、`message_delta`、`message_stop`）。工具参数由 `input_json_delta` 片段拼装，只在 block 关闭后才交出——半截却被当成完整调用比没有更糟。
- `openai`：流式循环末尾的不可达代码已消除，解析移入 `recvEvent`，由它报告该事件是否产生了 chunk。

- 初始脚手架：provider、message/tool 类型、registry、toolset 与流式收集。
