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

## Planned

- OpenAI-compatible provider adapter (covers OpenAI, DeepSeek, Qwen and
  compatible gateways).
- Anthropic provider adapter.
- Retry, timeout, rate-limit and usage accounting helpers.
- Structured output helpers.

## License

MIT. See [LICENSE](LICENSE).
