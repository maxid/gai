# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

- `anthropic`: a `gai.Provider` for the Messages API — chat and streaming, with the system
  prompt hoisted, tool calls as content blocks, tool results as user messages, and typed SSE
  events (`content_block_start/delta/stop`, `message_delta`, `message_stop`). Tool arguments
  are assembled from `input_json_delta` fragments and only handed out once their block closes,
  because a half-written call looking complete is worse than none.
- `openai`: the streaming loop no longer ends with unreachable code; parsing moved to
  `recvEvent`, which reports whether an event produced a chunk.

- Initial scaffold: provider, message/tool types, registry, toolset and stream
  collection.
