package anthropic

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"strings"

	"github.com/maxid/gai"
)

// stream decodes the Messages API's typed events.
//
// The event names carry the meaning here: content_block_start opens a tool call, its
// input_json_delta fragments are concatenated, and content_block_stop is when the call is
// complete. Text deltas are emitted as they arrive; tool calls are held until their block
// closes, because a half-parsed argument object is not something a caller can act on.
type stream struct {
	body   io.ReadCloser
	reader *bufio.Reader

	// pending is the tool call being assembled, keyed by content block index.
	pending map[int]*pendingCall
	// queued are finished tool calls waiting to be handed out.
	queued []gai.ToolCall
	// usage arrives on message_delta, after the content.
	usage  *gai.Usage
	done   bool
	closed bool
}

type pendingCall struct {
	id    string
	name  string
	input strings.Builder
}

func newStream(body io.ReadCloser) *stream {
	return &stream{body: body, reader: bufio.NewReader(body), pending: map[int]*pendingCall{}}
}

// sseEvent is one decoded server-sent event.
type sseEvent struct {
	name string
	data string
}

// next reads one event. It returns io.EOF when the body ends.
func (s *stream) next() (sseEvent, error) {
	var event sseEvent
	var data []string
	for {
		line, err := s.reader.ReadString('\n')
		if err != nil {
			if err == io.EOF && (event.name != "" || len(data) > 0) {
				event.data = strings.Join(data, "\n")
				return event, nil
			}
			return sseEvent{}, err
		}
		line = strings.TrimRight(line, "\r\n")
		switch {
		case line == "":
			if event.name != "" || len(data) > 0 {
				event.data = strings.Join(data, "\n")
				return event, nil
			}
		case strings.HasPrefix(line, "event:"):
			event.name = strings.TrimSpace(strings.TrimPrefix(line, "event:"))
		case strings.HasPrefix(line, "data:"):
			data = append(data, strings.TrimSpace(strings.TrimPrefix(line, "data:")))
		}
	}
}

// Recv implements gai.Stream.
func (s *stream) Recv() (gai.Chunk, error) {
	if s.closed {
		return gai.Chunk{}, io.EOF
	}
	if len(s.queued) > 0 {
		chunk := gai.Chunk{ToolCall: &s.queued[0]}
		s.queued = s.queued[1:]
		return chunk, nil
	}

	for {
		event, err := s.next()
		if err != nil {
			if err == io.EOF {
				s.closed = true
				if !s.done {
					// The server closed without message_stop: report what we have rather than
					// pretending the response completed.
					return s.finish(), nil
				}
				return gai.Chunk{}, io.EOF
			}
			return gai.Chunk{}, err
		}

		chunk, ok, err := s.handle(event)
		if err != nil {
			return gai.Chunk{}, err
		}
		if ok {
			return chunk, nil
		}
	}
}

// handle processes one event, reporting whether it produced a chunk for the caller.
func (s *stream) handle(event sseEvent) (gai.Chunk, bool, error) {
	switch event.name {
	case "content_block_start":
		var payload struct {
			Index        int         `json:"index"`
			ContentBlock wireContent `json:"content_block"`
		}
		if err := json.Unmarshal([]byte(event.data), &payload); err != nil {
			return gai.Chunk{}, false, fmt.Errorf("gai/anthropic: decode content_block_start: %w", err)
		}
		if payload.ContentBlock.Type == "tool_use" {
			call := &pendingCall{id: payload.ContentBlock.ID, name: payload.ContentBlock.Name}
			if len(payload.ContentBlock.Input) > 0 && string(payload.ContentBlock.Input) != "{}" {
				call.input.Write(payload.ContentBlock.Input)
			}
			s.pending[payload.Index] = call
		}
		return gai.Chunk{}, false, nil

	case "content_block_delta":
		var payload struct {
			Index int `json:"index"`
			Delta struct {
				Type        string `json:"type"`
				Text        string `json:"text"`
				PartialJSON string `json:"partial_json"`
			} `json:"delta"`
		}
		if err := json.Unmarshal([]byte(event.data), &payload); err != nil {
			return gai.Chunk{}, false, fmt.Errorf("gai/anthropic: decode content_block_delta: %w", err)
		}
		switch payload.Delta.Type {
		case "text_delta":
			if payload.Delta.Text == "" {
				return gai.Chunk{}, false, nil
			}
			return gai.Chunk{Delta: payload.Delta.Text}, true, nil
		case "input_json_delta":
			if call, ok := s.pending[payload.Index]; ok {
				call.input.WriteString(payload.Delta.PartialJSON)
			}
		}
		return gai.Chunk{}, false, nil

	case "content_block_stop":
		var payload struct {
			Index int `json:"index"`
		}
		if err := json.Unmarshal([]byte(event.data), &payload); err != nil {
			return gai.Chunk{}, false, fmt.Errorf("gai/anthropic: decode content_block_stop: %w", err)
		}
		call, ok := s.pending[payload.Index]
		if !ok {
			return gai.Chunk{}, false, nil
		}
		delete(s.pending, payload.Index)
		arguments := strings.TrimSpace(call.input.String())
		if arguments == "" {
			arguments = "{}"
		}
		if !json.Valid([]byte(arguments)) {
			return gai.Chunk{}, false, fmt.Errorf("gai/anthropic: tool %s arguments are not valid JSON: %s", call.name, arguments)
		}
		s.queued = append(s.queued, gai.ToolCall{ID: call.id, Name: call.name, Arguments: json.RawMessage(arguments)})
		chunk := gai.Chunk{ToolCall: &s.queued[0]}
		s.queued = s.queued[1:]
		return chunk, true, nil

	case "message_delta":
		var payload struct {
			Usage struct {
				OutputTokens int `json:"output_tokens"`
			} `json:"usage"`
		}
		if err := json.Unmarshal([]byte(event.data), &payload); err != nil {
			return gai.Chunk{}, false, nil
		}
		if payload.Usage.OutputTokens > 0 {
			usage := gai.Usage{CompletionTokens: payload.Usage.OutputTokens, TotalTokens: payload.Usage.OutputTokens}
			s.usage = &usage
		}
		return gai.Chunk{}, false, nil

	case "message_stop":
		s.done = true
		s.closed = true
		return s.finish(), true, nil

	case "error":
		return gai.Chunk{}, false, fmt.Errorf("gai/anthropic: stream error: %s", event.data)
	}
	return gai.Chunk{}, false, nil
}

// finish builds the final chunk from whatever the stream left behind.
func (s *stream) finish() gai.Chunk {
	chunk := gai.Chunk{Done: true, Usage: s.usage}
	// Unfinished tool calls are not handed out: their arguments may be half-written, and a
	// truncated call looking complete is worse than none.
	s.pending = map[int]*pendingCall{}
	return chunk
}

// Close implements gai.Stream.
func (s *stream) Close() error {
	s.closed = true
	return s.body.Close()
}
