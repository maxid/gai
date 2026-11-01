// Package openai implements gai.Provider against the OpenAI chat-completions API.
//
// "OpenAI-compatible" is the widest seam available: it covers the hosted API, Azure
// proxies, and the domestic and self-hosted servers (DeepSeek, Qwen, Moonshot,
// Ollama, vLLM) that copy the same wire format. A vendor that diverges gets its own
// adapter rather than a pile of flags here.
package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/maxid/gai"
)

// DefaultBaseURL is used when Config.BaseURL is empty.
const DefaultBaseURL = "https://api.openai.com/v1"

// Config configures the provider.
type Config struct {
	// BaseURL ends before "/chat/completions", e.g. https://api.openai.com/v1.
	BaseURL string
	// APIKey is sent as a bearer token. Some self-hosted servers ignore it.
	APIKey string
	// Model is the default when a request names none.
	Model string
	// Organization is the optional OpenAI-Organization header.
	Organization string
	// Headers are extra request headers (vendor routing, tracing).
	Headers map[string]string
	// HTTPClient overrides the default client (timeouts, proxies, tests).
	HTTPClient *http.Client
	// Name is the provider name reported to gai; defaults to "openai".
	Name string
	// RequestTimeout bounds one call when no deadline is set by the caller.
	RequestTimeout time.Duration
}

// Provider is an OpenAI-compatible chat provider.
type Provider struct {
	cfg    Config
	client *http.Client
}

// New validates the configuration.
func New(cfg Config) (*Provider, error) {
	if cfg.BaseURL == "" {
		cfg.BaseURL = DefaultBaseURL
	}
	cfg.BaseURL = strings.TrimRight(cfg.BaseURL, "/")
	if cfg.Name == "" {
		cfg.Name = "openai"
	}
	if cfg.RequestTimeout <= 0 {
		cfg.RequestTimeout = 2 * time.Minute
	}

	client := cfg.HTTPClient
	if client == nil {
		client = &http.Client{Timeout: cfg.RequestTimeout}
	}
	return &Provider{cfg: cfg, client: client}, nil
}

// Name implements gai.Provider.
func (p *Provider) Name() string { return p.cfg.Name }

// Chat implements gai.Provider.
func (p *Provider) Chat(ctx context.Context, request gai.Request) (gai.Response, error) {
	body, err := p.marshal(request, false)
	if err != nil {
		return gai.Response{}, err
	}

	raw, err := p.do(ctx, body)
	if err != nil {
		return gai.Response{}, err
	}

	var decoded chatResponse
	if err := json.Unmarshal(raw, &decoded); err != nil {
		return gai.Response{}, fmt.Errorf("gai/openai: decode response: %w", err)
	}
	if len(decoded.Choices) == 0 {
		return gai.Response{}, fmt.Errorf("gai/openai: response has no choices")
	}

	choice := decoded.Choices[0]
	return gai.Response{
		Message:      choice.Message.toMessage(),
		Usage:        decoded.Usage.toUsage(),
		FinishReason: choice.FinishReason,
	}, nil
}

// Stream implements gai.Provider.
func (p *Provider) Stream(ctx context.Context, request gai.Request) (gai.Stream, error) {
	body, err := p.marshal(request, true)
	if err != nil {
		return nil, err
	}

	response, err := p.send(ctx, body)
	if err != nil {
		return nil, err
	}
	return newStream(response.Body), nil
}

// marshal renders the request body.
func (p *Provider) marshal(request gai.Request, stream bool) ([]byte, error) {
	model := request.Model
	if model == "" {
		model = p.cfg.Model
	}
	if model == "" {
		return nil, fmt.Errorf("gai/openai: model is required")
	}

	payload := chatRequest{
		Model:       model,
		Messages:    make([]wireMessage, 0, len(request.Messages)),
		Temperature: request.Temperature,
		MaxTokens:   request.MaxTokens,
		Stream:      stream,
	}
	for _, tool := range request.Tools {
		if err := tool.Validate(); err != nil {
			return nil, err
		}
		definition := map[string]any{
			"type": "function",
			"function": map[string]any{
				"name":        tool.Name,
				"description": tool.Description,
			},
		}
		// The schema is the arguments object; without one, an empty object schema is
		// the only shape the API accepts.
		if len(tool.Schema) > 0 {
			definition["function"].(map[string]any)["parameters"] = json.RawMessage(tool.Schema)
		} else {
			definition["function"].(map[string]any)["parameters"] = json.RawMessage(`{"type":"object","properties":{}}`)
		}
		payload.Tools = append(payload.Tools, definition)
	}

	for _, message := range request.Messages {
		wire, err := toWireMessage(message)
		if err != nil {
			return nil, err
		}
		payload.Messages = append(payload.Messages, wire)
	}

	return json.Marshal(payload)
}

// do sends the request and returns the body of a 2xx response.
func (p *Provider) do(ctx context.Context, body []byte) ([]byte, error) {
	response, err := p.send(ctx, body)
	if err != nil {
		return nil, err
	}
	defer response.Body.Close()

	raw, err := io.ReadAll(response.Body)
	if err != nil {
		return nil, fmt.Errorf("gai/openai: read response: %w", err)
	}
	return raw, nil
}

// send performs the HTTP call, turning a non-2xx answer into an error that carries the
// status and a bounded body slice — a provider error is the first thing an operator
// needs, and the body is where the API explains it.
func (p *Provider) send(ctx context.Context, body []byte) (*http.Response, error) {
	endpoint := p.cfg.BaseURL + "/chat/completions"
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	request.Header.Set("Content-Type", "application/json")
	if p.cfg.APIKey != "" {
		request.Header.Set("Authorization", "Bearer "+p.cfg.APIKey)
	}
	if p.cfg.Organization != "" {
		request.Header.Set("OpenAI-Organization", p.cfg.Organization)
	}
	for key, value := range p.cfg.Headers {
		request.Header.Set(key, value)
	}

	response, err := p.client.Do(request)
	if err != nil {
		return nil, fmt.Errorf("gai/openai: request: %w", err)
	}
	if response.StatusCode >= 200 && response.StatusCode < 300 {
		return response, nil
	}

	defer response.Body.Close()
	snippet, _ := io.ReadAll(io.LimitReader(response.Body, 2<<10))
	return nil, fmt.Errorf("gai/openai: %d %s: %s", response.StatusCode, response.Status, strings.TrimSpace(string(snippet)))
}
