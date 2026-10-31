package gai

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"sort"
	"strings"
	"sync"
)

// Role is a chat message role.
type Role string

const (
	RoleSystem    Role = "system"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleTool      Role = "tool"
)

// Message is one chat message.
type Message struct {
	Role       Role       `json:"role"`
	Content    string     `json:"content,omitempty"`
	Name       string     `json:"name,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
}

// ToolCall is a model request to invoke a tool.
type ToolCall struct {
	ID        string          `json:"id"`
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments"`
}

// Tool is one callable function exposed to a model.
type Tool struct {
	Name        string
	Description string
	Schema      json.RawMessage // JSON Schema for the arguments object
	Handler     func(context.Context, json.RawMessage) (any, error)
}

// Validate checks the fields a provider needs.
func (t Tool) Validate() error {
	if strings.TrimSpace(t.Name) == "" {
		return fmt.Errorf("gai: tool name is required")
	}
	if len(t.Schema) > 0 && !json.Valid(t.Schema) {
		return fmt.Errorf("gai: tool %s schema is not valid JSON", t.Name)
	}
	return nil
}

// Request is one chat completion request.
type Request struct {
	Model       string
	Messages    []Message
	Tools       []Tool
	Temperature *float64
	MaxTokens   int
	Metadata    map[string]string
}

// Usage reports token consumption.
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// Response is one chat completion response.
type Response struct {
	Message      Message `json:"message"`
	Usage        Usage   `json:"usage"`
	FinishReason string  `json:"finish_reason,omitempty"`
}

// Chunk is one streamed delta. Recv returns io.EOF after the final chunk.
type Chunk struct {
	Delta    string    `json:"delta,omitempty"`
	ToolCall *ToolCall `json:"tool_call,omitempty"`
	Usage    *Usage    `json:"usage,omitempty"`
	Done     bool      `json:"done,omitempty"`
}

// Stream is a streaming chat response.
type Stream interface {
	Recv() (Chunk, error)
	Close() error
}

// Provider is one model backend.
type Provider interface {
	Name() string
	Chat(context.Context, Request) (Response, error)
	Stream(context.Context, Request) (Stream, error)
}

// Registry holds named providers.
type Registry struct {
	mu        sync.RWMutex
	providers map[string]Provider
}

// NewRegistry returns an empty provider registry.
func NewRegistry() *Registry {
	return &Registry{providers: make(map[string]Provider)}
}

// Register adds a provider. The name comes from the provider itself.
func (r *Registry) Register(provider Provider) error {
	if provider == nil {
		return fmt.Errorf("gai: nil provider")
	}
	name := strings.TrimSpace(provider.Name())
	if name == "" {
		return fmt.Errorf("gai: provider name is required")
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.providers == nil {
		r.providers = make(map[string]Provider)
	}
	if _, exists := r.providers[name]; exists {
		return fmt.Errorf("gai: duplicate provider %q", name)
	}
	r.providers[name] = provider
	return nil
}

// Get returns a provider by name.
func (r *Registry) Get(name string) (Provider, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	provider, ok := r.providers[name]
	return provider, ok
}

// Names returns the registered provider names in sorted order.
func (r *Registry) Names() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()
	names := make([]string, 0, len(r.providers))
	for name := range r.providers {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

// Toolset holds named tools and dispatches calls.
type Toolset struct {
	mu    sync.RWMutex
	tools map[string]Tool
}

// NewToolset returns an empty toolset.
func NewToolset() *Toolset {
	return &Toolset{tools: make(map[string]Tool)}
}

// Register adds a tool.
func (t *Toolset) Register(tool Tool) error {
	if err := tool.Validate(); err != nil {
		return err
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.tools == nil {
		t.tools = make(map[string]Tool)
	}
	if _, exists := t.tools[tool.Name]; exists {
		return fmt.Errorf("gai: duplicate tool %q", tool.Name)
	}
	t.tools[tool.Name] = tool
	return nil
}

// Tools returns the registered tools in name order.
func (t *Toolset) Tools() []Tool {
	t.mu.RLock()
	defer t.mu.RUnlock()
	tools := make([]Tool, 0, len(t.tools))
	for _, tool := range t.tools {
		tools = append(tools, tool)
	}
	sort.Slice(tools, func(i, j int) bool { return tools[i].Name < tools[j].Name })
	return tools
}

// Call invokes a registered tool.
func (t *Toolset) Call(ctx context.Context, name string, arguments json.RawMessage) (any, error) {
	t.mu.RLock()
	tool, ok := t.tools[name]
	t.mu.RUnlock()
	if !ok {
		return nil, fmt.Errorf("gai: unknown tool %q", name)
	}
	if tool.Handler == nil {
		return nil, fmt.Errorf("gai: tool %q has no handler", name)
	}
	if len(arguments) == 0 {
		arguments = json.RawMessage(`{}`)
	}
	if !json.Valid(arguments) {
		return nil, fmt.Errorf("gai: tool %q arguments are not valid JSON", name)
	}
	return tool.Handler(ctx, arguments)
}

// Collect drains a stream into a response.
func Collect(stream Stream) (Response, error) {
	if stream == nil {
		return Response{}, fmt.Errorf("gai: nil stream")
	}
	defer stream.Close()

	var response Response
	var builder strings.Builder
	for {
		chunk, err := stream.Recv()
		if err != nil {
			if err == io.EOF {
				break
			}
			return Response{}, err
		}
		builder.WriteString(chunk.Delta)
		if chunk.ToolCall != nil {
			response.Message.ToolCalls = append(response.Message.ToolCalls, *chunk.ToolCall)
		}
		if chunk.Usage != nil {
			response.Usage = *chunk.Usage
		}
		if chunk.Done {
			break
		}
	}
	response.Message.Role = RoleAssistant
	response.Message.Content = builder.String()
	response.FinishReason = "stop"
	return response, nil
}
