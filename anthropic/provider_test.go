package anthropic_test

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/maxid/gai"
	"github.com/maxid/gai/anthropic"
)

// The Messages API puts the system prompt at the top level, wraps tool calls in content
// blocks, and turns a tool result into a user message. Getting this wrong is invisible until a
// real model refuses the request, so the test asserts the body.
func TestChatRendersTheMessagesWireFormat(t *testing.T) {
	var received map[string]any
	var headers http.Header
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		headers = r.Header.Clone()
		raw, _ := io.ReadAll(r.Body)
		if err := json.Unmarshal(raw, &received); err != nil {
			t.Errorf("decode body: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{
			"content": [
				{"type": "text", "text": "let me look"},
				{"type": "tool_use", "id": "toolu_1", "name": "orders.summary", "input": {"days": 7}}
			],
			"stop_reason": "tool_use",
			"usage": {"input_tokens": 12, "output_tokens": 34}
		}`)
	}))
	defer server.Close()

	provider, err := anthropic.New(anthropic.Config{BaseURL: server.URL, APIKey: "key-123", Model: "claude-test"})
	if err != nil {
		t.Fatal(err)
	}
	response, err := provider.Chat(context.Background(), gai.Request{
		Messages: []gai.Message{
			{Role: gai.RoleSystem, Content: "be brief"},
			{Role: gai.RoleSystem, Content: "and precise"},
			{Role: gai.RoleUser, Content: "how many orders?"},
			{Role: gai.RoleAssistant, Content: "", ToolCalls: []gai.ToolCall{{ID: "toolu_0", Name: "orders.summary", Arguments: json.RawMessage(`{"days":1}`)}}},
			{Role: gai.RoleTool, ToolCallID: "toolu_0", Content: "3"},
		},
		Tools: []gai.Tool{{Name: "orders.summary", Description: "summarise", Schema: json.RawMessage(`{"type":"object"}`)}},
	})
	if err != nil {
		t.Fatal(err)
	}

	if headers.Get("x-api-key") != "key-123" || headers.Get("anthropic-version") == "" {
		t.Fatalf("headers = %v", headers)
	}
	if received["model"] != "claude-test" || received["max_tokens"].(float64) != 1024 {
		t.Fatalf("body = %v", received)
	}
	if received["system"] != "be brief\n\nand precise" {
		t.Fatalf("system = %v", received["system"])
	}
	messages := received["messages"].([]any)
	if len(messages) != 3 {
		t.Fatalf("messages = %v", messages)
	}
	// The tool result is a user message, not a "tool" role.
	last := messages[2].(map[string]any)
	block := last["content"].([]any)[0].(map[string]any)
	if last["role"] != "user" || block["type"] != "tool_result" || block["tool_use_id"] != "toolu_0" {
		t.Fatalf("tool result = %v", last)
	}

	if response.Message.Content != "let me look" || len(response.Message.ToolCalls) != 1 {
		t.Fatalf("response = %+v", response.Message)
	}
	call := response.Message.ToolCalls[0]
	if call.Name != "orders.summary" || string(call.Arguments) != `{"days": 7}` {
		t.Fatalf("tool call = %+v", call)
	}
	if response.FinishReason != "tool_calls" || response.Usage.TotalTokens != 46 {
		t.Fatalf("finish/usage = %q %+v", response.FinishReason, response.Usage)
	}
}

// Streaming is a sequence of typed events: text arrives as text_delta, tool arguments as
// input_json_delta fragments that only mean something once the block closes.
func TestStreamAssemblesTextAndToolCalls(t *testing.T) {
	events := []string{
		`event: message_start
data: {"type":"message_start"}`,
		`event: content_block_start
data: {"index":0,"content_block":{"type":"text","text":""}}`,
		`event: content_block_delta
data: {"index":0,"delta":{"type":"text_delta","text":"checking"}}`,
		`event: content_block_stop
data: {"index":0}`,
		`event: content_block_start
data: {"index":1,"content_block":{"type":"tool_use","id":"toolu_2","name":"orders.summary","input":{}}}`,
		`event: content_block_delta
data: {"index":1,"delta":{"type":"input_json_delta","partial_json":"{\"days\":"}}`,
		`event: content_block_delta
data: {"index":1,"delta":{"type":"input_json_delta","partial_json":"7}"}}`,
		`event: content_block_stop
data: {"index":1}`,
		`event: message_delta
data: {"type":"message_delta","usage":{"output_tokens":9}}`,
		`event: message_stop
data: {"type":"message_stop"}`,
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		for _, event := range events {
			_, _ = io.WriteString(w, event+"\n\n")
		}
	}))
	defer server.Close()

	provider, err := anthropic.New(anthropic.Config{BaseURL: server.URL, Model: "claude-test"})
	if err != nil {
		t.Fatal(err)
	}
	stream, err := provider.Stream(context.Background(), gai.Request{
		Messages: []gai.Message{{Role: gai.RoleUser, Content: "hi"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()

	var text strings.Builder
	var calls []gai.ToolCall
	var usage *gai.Usage
	for {
		chunk, err := stream.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
		switch {
		case chunk.ToolCall != nil:
			calls = append(calls, *chunk.ToolCall)
		case chunk.Delta != "":
			text.WriteString(chunk.Delta)
		}
		if chunk.Usage != nil {
			usage = chunk.Usage
		}
		if chunk.Done {
			break
		}
	}
	if text.String() != "checking" {
		t.Fatalf("text = %q", text.String())
	}
	if len(calls) != 1 || calls[0].Name != "orders.summary" || string(calls[0].Arguments) != `{"days":7}` {
		t.Fatalf("calls = %+v", calls)
	}
	if usage == nil || usage.CompletionTokens != 9 {
		t.Fatalf("usage = %+v", usage)
	}
}

// An API error is reported, not turned into an empty answer.
func TestAPIErrorIsReported(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = io.WriteString(w, `{"error":{"message":"rate limited"}}`)
	}))
	defer server.Close()

	provider, err := anthropic.New(anthropic.Config{BaseURL: server.URL, Model: "claude-test"})
	if err != nil {
		t.Fatal(err)
	}
	_, err = provider.Chat(context.Background(), gai.Request{Messages: []gai.Message{{Role: gai.RoleUser, Content: "hi"}}})
	if err == nil || !strings.Contains(err.Error(), "429") || !strings.Contains(err.Error(), "rate limited") {
		t.Fatalf("err = %v", err)
	}
}

func TestMissingModelIsAnError(t *testing.T) {
	provider, err := anthropic.New(anthropic.Config{})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := provider.Chat(context.Background(), gai.Request{}); err == nil ||
		!strings.Contains(err.Error(), "model is required") {
		t.Fatalf("err = %v", err)
	}
}
