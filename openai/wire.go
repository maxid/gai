package openai

import (
	"encoding/json"
	"fmt"

	"github.com/maxid/gai"
)

// The wire types keep the API's shape out of gai: `arguments` is a JSON *string* on
// the wire and a RawMessage in the library, and tool calls appear in a different
// place in a request (assistant message) than in a response (choice).

type chatRequest struct {
	Model       string        `json:"model"`
	Messages    []wireMessage `json:"messages"`
	Tools       []any         `json:"tools,omitempty"`
	Temperature *float64      `json:"temperature,omitempty"`
	MaxTokens   int           `json:"max_tokens,omitempty"`
	Stream      bool          `json:"stream,omitempty"`
}

type wireMessage struct {
	Role       string         `json:"role"`
	Content    string         `json:"content,omitempty"`
	Name       string         `json:"name,omitempty"`
	ToolCallID string         `json:"tool_call_id,omitempty"`
	ToolCalls  []wireToolCall `json:"tool_calls,omitempty"`
}

type wireToolCall struct {
	// Index is only present in streaming deltas; it is what lets partial arguments be
	// reassembled.
	Index    *int             `json:"index,omitempty"`
	ID       string           `json:"id,omitempty"`
	Type     string           `json:"type,omitempty"`
	Function wireFunctionCall `json:"function"`
}

type wireFunctionCall struct {
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
}

type chatResponse struct {
	Choices []struct {
		Message      wireMessage `json:"message"`
		FinishReason string      `json:"finish_reason"`
	} `json:"choices"`
	Usage usage `json:"usage"`
}

type usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

func (u usage) toUsage() gai.Usage {
	return gai.Usage{PromptTokens: u.PromptTokens, CompletionTokens: u.CompletionTokens, TotalTokens: u.TotalTokens}
}

// toWireMessage renders one library message.
//
// A tool call's arguments are a JSON string here: sending the object is a common
// mistake that the API rejects with a 400, so the conversion is explicit.
func toWireMessage(message gai.Message) (wireMessage, error) {
	wire := wireMessage{
		Role:       string(message.Role),
		Content:    message.Content,
		Name:       message.Name,
		ToolCallID: message.ToolCallID,
	}
	for _, call := range message.ToolCalls {
		arguments := "{}"
		if len(call.Arguments) > 0 {
			if !json.Valid(call.Arguments) {
				return wireMessage{}, fmt.Errorf("gai/openai: tool call %s arguments are not valid JSON", call.Name)
			}
			arguments = string(call.Arguments)
		}
		wire.ToolCalls = append(wire.ToolCalls, wireToolCall{
			ID:       call.ID,
			Type:     "function",
			Function: wireFunctionCall{Name: call.Name, Arguments: arguments},
		})
	}
	return wire, nil
}

// toMessage renders a response message back into the library type.
func (m wireMessage) toMessage() gai.Message {
	message := gai.Message{Role: gai.Role(m.Role), Content: m.Content, Name: m.Name, ToolCallID: m.ToolCallID}
	if message.Role == "" {
		message.Role = gai.RoleAssistant
	}
	for _, call := range m.ToolCalls {
		arguments := json.RawMessage(call.Function.Arguments)
		if len(arguments) == 0 || !json.Valid(arguments) {
			// A malformed argument string becomes an empty object so the caller can
			// still see the call and report the problem, instead of losing it.
			arguments = json.RawMessage(`{}`)
		}
		message.ToolCalls = append(message.ToolCalls, gai.ToolCall{
			ID:        call.ID,
			Name:      call.Function.Name,
			Arguments: arguments,
		})
	}
	return message
}
