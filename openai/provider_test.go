package openai_test

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/maxid/gai"
	"github.com/maxid/gai/openai"
)

func provider(t *testing.T, handler http.HandlerFunc) (*openai.Provider, *httptest.Server) {
	t.Helper()
	server := httptest.NewServer(handler)
	t.Cleanup(server.Close)

	p, err := openai.New(openai.Config{
		BaseURL: server.URL,
		APIKey:  "test-key",
		Model:   "gpt-test",
	})
	if err != nil {
		t.Fatalf("provider: %v", err)
	}
	return p, server
}

func TestChatSendsTheOpenAIShape(t *testing.T) {
	var received map[string]any

	p, _ := provider(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/chat/completions" {
			t.Errorf("path = %q", r.URL.Path)
		}
		if got := r.Header.Get("Authorization"); got != "Bearer test-key" {
			t.Errorf("authorization = %q", got)
		}
		body, _ := io.ReadAll(r.Body)
		if err := json.Unmarshal(body, &received); err != nil {
			t.Fatalf("decode request: %v", err)
		}

		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []any{map[string]any{
				"message":       map[string]any{"role": "assistant", "content": "hello"},
				"finish_reason": "stop",
			}},
			"usage": map[string]any{"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
		})
	})

	response, err := p.Chat(context.Background(), gai.Request{
		Messages: []gai.Message{{Role: gai.RoleUser, Content: "hi"}},
		Tools: []gai.Tool{{
			Name:        "lookup",
			Description: "look something up",
			Schema:      json.RawMessage(`{"type":"object","properties":{"q":{"type":"string"}}}`),
		}},
	})
	if err != nil {
		t.Fatalf("chat: %v", err)
	}

	if response.Message.Content != "hello" || response.Usage.TotalTokens != 5 {
		t.Fatalf("response = %+v", response)
	}
	if received["model"] != "gpt-test" {
		t.Errorf("model = %v, want the configured default", received["model"])
	}
	tools, _ := received["tools"].([]any)
	if len(tools) != 1 {
		t.Fatalf("tools = %v", received["tools"])
	}
	function := tools[0].(map[string]any)["function"].(map[string]any)
	if function["name"] != "lookup" || function["parameters"] == nil {
		t.Errorf("tool definition = %v", function)
	}
}

// A tool call's arguments must travel as a JSON *string*; sending the object is a 400.
func TestChatEncodesToolCallArgumentsAsString(t *testing.T) {
	var received map[string]any

	p, _ := provider(t, func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &received)
		_, _ = w.Write([]byte(`{"choices":[{"message":{"role":"assistant","content":""}}]}`))
	})

	_, err := p.Chat(context.Background(), gai.Request{
		Messages: []gai.Message{{
			Role: gai.RoleAssistant,
			ToolCalls: []gai.ToolCall{{
				ID:        "call-1",
				Name:      "lookup",
				Arguments: json.RawMessage(`{"q":"x"}`),
			}},
		}},
	})
	if err != nil {
		t.Fatalf("chat: %v", err)
	}

	messages := received["messages"].([]any)
	first := messages[0].(map[string]any)
	calls := first["tool_calls"].([]any)
	call := calls[0].(map[string]any)
	arguments := call["function"].(map[string]any)["arguments"]

	if _, ok := arguments.(string); !ok {
		t.Fatalf("arguments = %T, want a JSON string", arguments)
	}
	if arguments != `{"q":"x"}` {
		t.Errorf("arguments = %v", arguments)
	}
	if call["type"] != "function" {
		t.Errorf("type = %v", call["type"])
	}
}

func TestChatDecodesToolCalls(t *testing.T) {
	p, _ := provider(t, func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{
			"choices": [{
				"message": {
					"role": "assistant",
					"tool_calls": [{"id":"call-9","type":"function","function":{"name":"lookup","arguments":"{\"q\":\"y\"}"}}]
				},
				"finish_reason": "tool_calls"
			}]
		}`))
	})

	response, err := p.Chat(context.Background(), gai.Request{})
	if err != nil {
		t.Fatalf("chat: %v", err)
	}
	if len(response.Message.ToolCalls) != 1 {
		t.Fatalf("tool calls = %+v", response.Message.ToolCalls)
	}
	call := response.Message.ToolCalls[0]
	if call.ID != "call-9" || call.Name != "lookup" || string(call.Arguments) != `{"q":"y"}` {
		t.Fatalf("call = %+v", call)
	}
	if response.FinishReason != "tool_calls" {
		t.Errorf("finish reason = %q", response.FinishReason)
	}
}

// A malformed argument string must not drop the call: the caller needs to see it.
func TestChatKeepsToolCallWithBadArguments(t *testing.T) {
	p, _ := provider(t, func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"choices":[{"message":{"tool_calls":[{"id":"c","function":{"name":"x","arguments":"not json"}}]}}]}`))
	})

	response, err := p.Chat(context.Background(), gai.Request{})
	if err != nil {
		t.Fatalf("chat: %v", err)
	}
	if len(response.Message.ToolCalls) != 1 || string(response.Message.ToolCalls[0].Arguments) != "{}" {
		t.Fatalf("call = %+v", response.Message.ToolCalls)
	}
}

func TestStreamDeliversTextThenToolCalls(t *testing.T) {
	p, _ := provider(t, func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher := w.(http.Flusher)
		for _, line := range []string{
			`data: {"choices":[{"delta":{"content":"hel"}}]}`,
			`data: {"choices":[{"delta":{"content":"lo"}}]}`,
			`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call-1","type":"function","function":{"name":"lookup","arguments":"{\"q\":"}}]}}]}`,
			`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"z\"}"}}]}}]}`,
			`data: [DONE]`,
		} {
			fmt.Fprintln(w, line)
			fmt.Fprintln(w)
			flusher.Flush()
		}
	})

	stream, err := p.Stream(context.Background(), gai.Request{})
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	defer stream.Close()

	var text strings.Builder
	var calls []gai.ToolCall
	for {
		chunk, err := stream.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatalf("recv: %v", err)
		}
		text.WriteString(chunk.Delta)
		if chunk.ToolCall != nil {
			calls = append(calls, *chunk.ToolCall)
		}
	}

	if text.String() != "hello" {
		t.Errorf("text = %q", text.String())
	}
	if len(calls) != 1 {
		t.Fatalf("tool calls = %+v", calls)
	}
	if calls[0].Name != "lookup" || string(calls[0].Arguments) != `{"q":"z"}` {
		t.Errorf("call = %+v", calls[0])
	}
}

func TestStreamReportsHTTPErrors(t *testing.T) {
	p, _ := provider(t, func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"error":{"message":"rate limited"}}`))
	})

	if _, err := p.Stream(context.Background(), gai.Request{}); err == nil {
		t.Fatal("a 429 must fail")
	} else if !strings.Contains(err.Error(), "429") || !strings.Contains(err.Error(), "rate limited") {
		t.Fatalf("err = %v, want the status and the body", err)
	}
}

func TestProviderRequiresAModel(t *testing.T) {
	p, err := openai.New(openai.Config{BaseURL: "http://127.0.0.1:1"})
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	if _, err := p.Chat(context.Background(), gai.Request{}); err == nil {
		t.Fatal("a request without a model must fail before the call")
	}
}

func TestNameDefaults(t *testing.T) {
	p, _ := openai.New(openai.Config{})
	if p.Name() != "openai" {
		t.Fatalf("name = %q", p.Name())
	}
}
