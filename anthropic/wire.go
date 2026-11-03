package anthropic

import (
	"encoding/json"
	"fmt"

	"github.com/maxid/gai"
)

// The wire types keep the Messages API's shape out of gai. Three differences drive them:
// the system prompt is top level, a tool call is a content block with a JSON *object* for its
// input (not a JSON string like OpenAI), and a tool result is a user message whose content
// block carries tool_use_id.

type messageRequest struct {
	Model       string        `json:"model"`
	System      string        `json:"system,omitempty"`
	Messages    []wireMessage `json:"messages"`
	Tools       []wireTool    `json:"tools,omitempty"`
	MaxTokens   int           `json:"max_tokens"`
	Temperature *float64      `json:"temperature,omitempty"`
	Stream      bool          `json:"stream,omitempty"`
}

type wireMessage struct {
	Role    string        `json:"role"`
	Content []wireContent `json:"content"`
}

// wireContent is one content block. The type selects which fields are meaningful.
type wireContent struct {
	Type string `json:"type"`

	// text
	Text string `json:"text,omitempty"`

	// tool_use
	ID    string          `json:"id,omitempty"`
	Name  string          `json:"name,omitempty"`
	Input json.RawMessage `json:"input,omitempty"`

	// tool_result
	ToolUseID string `json:"tool_use_id,omitempty"`
	Content   string `json:"content,omitempty"`
	IsError   bool   `json:"is_error,omitempty"`
}

type wireTool struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	InputSchema json.RawMessage `json:"input_schema"`
}

type messageResponse struct {
	Content    []wireContent `json:"content"`
	StopReason string        `json:"stop_reason"`
	Usage      struct {
		InputTokens  int `json:"input_tokens"`
		OutputTokens int `json:"output_tokens"`
	} `json:"usage"`
}

// marshal renders the request body.
func (p *Provider) marshal(request gai.Request, stream bool) ([]byte, error) {
	model := request.Model
	if model == "" {
		model = p.cfg.Model
	}
	if model == "" {
		return nil, fmt.Errorf("gai/anthropic: model is required")
	}
	maxTokens := request.MaxTokens
	if maxTokens <= 0 {
		maxTokens = p.cfg.MaxTokens
	}

	payload := messageRequest{
		Model:       model,
		MaxTokens:   maxTokens,
		Temperature: request.Temperature,
		Stream:      stream,
	}

	for _, tool := range request.Tools {
		if err := tool.Validate(); err != nil {
			return nil, err
		}
		schema := tool.Schema
		if len(schema) == 0 {
			schema = json.RawMessage(`{"type":"object","properties":{}}`)
		}
		payload.Tools = append(payload.Tools, wireTool{
			Name:        tool.Name,
			Description: tool.Description,
			InputSchema: schema,
		})
	}

	for _, message := range request.Messages {
		switch message.Role {
		case gai.RoleSystem:
			// The API has one system prompt; consecutive system messages are joined rather
			// than dropped.
			if payload.System != "" {
				payload.System += "\n\n"
			}
			payload.System += message.Content
		case gai.RoleTool:
			payload.Messages = append(payload.Messages, wireMessage{
				Role: "user",
				Content: []wireContent{{
					Type:      "tool_result",
					ToolUseID: message.ToolCallID,
					Content:   message.Content,
				}},
			})
		case gai.RoleAssistant:
			blocks := make([]wireContent, 0, 1+len(message.ToolCalls))
			if message.Content != "" {
				blocks = append(blocks, wireContent{Type: "text", Text: message.Content})
			}
			for _, call := range message.ToolCalls {
				input := call.Arguments
				if len(input) == 0 {
					input = json.RawMessage(`{}`)
				}
				if !json.Valid(input) {
					return nil, fmt.Errorf("gai/anthropic: tool call %s arguments are not valid JSON", call.Name)
				}
				blocks = append(blocks, wireContent{Type: "tool_use", ID: call.ID, Name: call.Name, Input: input})
			}
			if len(blocks) == 0 {
				// An empty assistant turn is not allowed; a single empty text block is the
				// closest honest rendering.
				blocks = append(blocks, wireContent{Type: "text", Text: ""})
			}
			payload.Messages = append(payload.Messages, wireMessage{Role: "assistant", Content: blocks})
		default:
			payload.Messages = append(payload.Messages, wireMessage{
				Role:    "user",
				Content: []wireContent{{Type: "text", Text: message.Content}},
			})
		}
	}
	return json.Marshal(payload)
}

// toGAI converts a response into the library's shape.
func (r messageResponse) toGAI() (gai.Response, error) {
	message := gai.Message{Role: gai.RoleAssistant}
	var text string
	for _, block := range r.Content {
		switch block.Type {
		case "text":
			text += block.Text
		case "tool_use":
			arguments := block.Input
			if len(arguments) == 0 {
				arguments = json.RawMessage(`{}`)
			}
			message.ToolCalls = append(message.ToolCalls, gai.ToolCall{
				ID:        block.ID,
				Name:      block.Name,
				Arguments: arguments,
			})
		}
	}
	message.Content = text

	finish := r.StopReason
	switch finish {
	case "max_tokens":
		finish = "length"
	case "tool_use":
		finish = "tool_calls"
	}
	return gai.Response{
		Message:      message,
		FinishReason: finish,
		Usage: gai.Usage{
			PromptTokens:     r.Usage.InputTokens,
			CompletionTokens: r.Usage.OutputTokens,
			TotalTokens:      r.Usage.InputTokens + r.Usage.OutputTokens,
		},
	}, nil
}
