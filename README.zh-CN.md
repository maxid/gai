[English](README.md) | 简体中文

# gai

面向大语言模型调用、工具与流式响应的小型供应商中立接口。它只定义契约，供应商适配器放在旁边。

> 状态：**实验性脚手架**（v0.0.0-dev）。API 尚不稳定。

## 特性

- `Message`、`ToolCall`、`Tool`、`Request`、`Response`、`Usage` 类型。
- `Provider` 接口，提供 `Chat` 与 `Stream`。
- `Registry`：按名称注册 provider，输出稳定排序。
- `Toolset`：注册工具并按 JSON 参数分发调用。
- `Collect`：把流式响应收拢为一次响应。

## 示例

```bash
go run ./examples/echo
```

## 安装

```bash
go get github.com/maxid/gai
```

## 规划中

- OpenAI-compatible provider 适配器（覆盖 OpenAI、DeepSeek、Qwen 及兼容网关）。
- Anthropic provider 适配器。
- 重试、超时、限流与用量统计助手。
- 结构化输出助手。

## 许可证

MIT。见 [LICENSE](LICENSE)。
