package openai

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"sort"
	"strings"

	"github.com/maxid/gai"
)

// stream decodes the server-sent events of a streaming completion.
//
// The API sends text deltas as they arrive but tool calls *incrementally*: the first
// fragment carries the id and name, later fragments append argument text. This type
// therefore emits text immediately and holds each tool call until the stream ends,
// then hands them out one per Recv before the final Done.
type stream struct {
	body   io.ReadCloser
	reader *bufio.Reader

	// pending tool calls keyed by the API's index.
	toolCalls map[int]*gai.ToolCall
	// queued are finished tool calls waiting to be returned.
	queued []gai.ToolCall
	done   bool
	closed bool
}

func newStream(body io.ReadCloser) *stream {
	return &stream{body: body, reader: bufio.NewReader(body), toolCalls: map[int]*gai.ToolCall{}}
}

// Recv implements gai.Stream.
func (s *stream) Recv() (gai.Chunk, error) {
	if s.closed {
		return gai.Chunk{}, io.EOF
	}
	// Hand out the finished tool calls before reporting completion.
	if len(s.queued) > 0 {
		chunk := gai.Chunk{ToolCall: &s.queued[0]}
		s.queued = s.queued[1:]
		return chunk, nil
	}

	for {
		line, err := s.reader.ReadString('\n')
		if err != nil {
			if err == io.EOF {
				if s.done {
					s.closed = true
					return gai.Chunk{}, io.EOF
				}
				// The server closed without [DONE]: treat accumulated calls as final.
				return s.finish()
			}
			return gai.Chunk{}, err
		}

		line = strings.TrimRight(line, "\r\n")
		if line == "" || strings.HasPrefix(line, ":") {
			continue
		}
		if !strings.HasPrefix(line, "data:") {
			// event:, id: and retry: fields carry no content for this API.
			continue
		}

		payload := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if payload == "[DONE]" {
			// [DONE] is where the accumulated tool calls become deliverable: they are
			// only complete once the stream says it is finished.
			s.done = true
			s.flushToolCalls()
			return s.recvQueued()
		}

		var event streamEvent
		if err := json.Unmarshal([]byte(payload), &event); err != nil {
			return gai.Chunk{}, fmt.Errorf("gai/openai: decode stream chunk: %w", err)
		}

		if len(event.Choices) == 0 {
			continue
		}
		delta := event.Choices[0].Delta
		for _, call := range delta.ToolCalls {
			s.accumulate(call)
		}

		if delta.Content != "" {
			return gai.Chunk{Delta: delta.Content}, nil
		}
		if event.Usage != nil {
			converted := event.Usage.toUsage()
			return gai.Chunk{Usage: &converted}, nil
		}
		// A tool-call-only delta produces no chunk: the call is not complete yet.
	}
	return s.recvQueued()
}

// recvQueued returns the next accumulated tool call.
func (s *stream) recvQueued() (gai.Chunk, error) {
	if len(s.queued) > 0 {
		chunk := gai.Chunk{ToolCall: &s.queued[0]}
		s.queued = s.queued[1:]
		return chunk, nil
	}
	s.closed = true
	return gai.Chunk{Done: true}, nil
}

// finish flushes the accumulated calls and ends the stream.
func (s *stream) finish() (gai.Chunk, error) {
	s.done = true
	s.flushToolCalls()
	if len(s.queued) > 0 {
		chunk := gai.Chunk{ToolCall: &s.queued[0]}
		s.queued = s.queued[1:]
		return chunk, nil
	}
	s.closed = true
	return gai.Chunk{Done: true}, nil
}

// accumulate merges one delta fragment into the pending call.
func (s *stream) accumulate(call wireToolCall) {
	index := 0
	if call.Index != nil {
		index = *call.Index
	}

	pending, ok := s.toolCalls[index]
	if !ok {
		pending = &gai.ToolCall{Arguments: json.RawMessage{}}
		s.toolCalls[index] = pending
	}
	if call.ID != "" {
		pending.ID = call.ID
	}
	if call.Function.Name != "" {
		pending.Name = call.Function.Name
	}
	// Arguments arrive as text fragments, not as JSON values.
	if call.Function.Arguments != "" {
		pending.Arguments = append(pending.Arguments, []byte(call.Function.Arguments)...)
	}
}

// flushToolCalls moves the accumulated calls into the queue in index order, so a
// multi-tool answer is delivered deterministically.
func (s *stream) flushToolCalls() {
	if len(s.toolCalls) == 0 {
		return
	}
	indexes := make([]int, 0, len(s.toolCalls))
	for index := range s.toolCalls {
		indexes = append(indexes, index)
	}
	sort.Ints(indexes)

	for _, index := range indexes {
		call := s.toolCalls[index]
		if len(call.Arguments) == 0 || !json.Valid(call.Arguments) {
			call.Arguments = json.RawMessage(`{}`)
		}
		s.queued = append(s.queued, *call)
	}
	s.toolCalls = map[int]*gai.ToolCall{}
}

// Close implements gai.Stream.
func (s *stream) Close() error {
	s.closed = true
	return s.body.Close()
}

type streamEvent struct {
	Choices []struct {
		Delta struct {
			Content   string         `json:"content"`
			ToolCalls []wireToolCall `json:"tool_calls"`
		} `json:"delta"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage *usage `json:"usage"`
}
