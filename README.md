English | [简体中文](README.zh-CN.md)

# gai

A small, provider-neutral interface for large language model calls, tools and
streaming responses. It defines the contract; provider adapters live beside it.

> Status: **experimental scaffold** (v0.0.0-dev). The API is not stable yet.

## Features

- `Message`, `ToolCall`, `Tool`, `Request`, `Response` and `Usage` types.
- `Provider` with `Chat` and `Stream` methods.
- `Registry` for named providers, with deterministic name order.
- `Toolset` for registering tools and dispatching calls with JSON arguments.
- `Collect` to drain a stream into a response.

## Example

```bash
go run ./examples/echo
```

## Install

```bash
go get github.com/maxid/gai
```

## Providers

| package | API | stream |
|---|---|---|
| `openai` | OpenAI chat completions, and every server that copies the wire format (DeepSeek, Qwen, Moonshot, Ollama, vLLM) | ✅ |
| `anthropic` | Anthropic Messages (`x-api-key`, `anthropic-version`), with the system prompt at the top level and tool calls as content blocks | ✅ |

The two adapters are separate packages, not flags on one: the wire formats diverge where it
matters (system prompt placement, tool-call shape, tool results as user messages, typed
streaming events). A vendor that diverges from both gets its own adapter.

## Planned

- Retry, timeout, rate-limit and usage accounting helpers.
- Structured output helpers.

## License

MIT. See [LICENSE](LICENSE).
