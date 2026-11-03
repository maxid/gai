// Package anthropic implements gai.Provider against the Anthropic Messages API.
//
// It is a separate adapter rather than a flag on the OpenAI one because the wire format
// diverges where it matters: the system prompt is a top-level field, tool calls are content
// blocks instead of a field on the message, tool results are user messages with a tool_use_id,
// and streaming delivers typed events rather than chat-completion deltas.
package anthropic

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
const DefaultBaseURL = "https://api.anthropic.com"

// DefaultVersion is the API version sent when Config.Version is empty.
//
// Anthropic requires the header on every request; pinning it here means an adapter upgrade is
// a deliberate act rather than a silent surprise.
const DefaultVersion = "2023-06-01"

// DefaultMaxTokens is used when neither the request nor the config sets one.
//
// The Messages API refuses a request without max_tokens, so the adapter has to choose; 1024 is
// small enough to be a safe default and large enough to be useful.
const DefaultMaxTokens = 1024

// Config configures the provider.
type Config struct {
	// BaseURL ends before "/v1/messages", e.g. https://api.anthropic.com.
	BaseURL string
	// APIKey is sent as x-api-key.
	APIKey string
	// Model is the default when a request names none.
	Model string
	// Version is the anthropic-version header.
	Version string
	// MaxTokens is the default cap; a request may override it.
	MaxTokens int
	// Headers are extra request headers (proxies, tracing).
	Headers map[string]string
	// HTTPClient overrides the default client.
	HTTPClient *http.Client
	// Name is the provider name reported to gai; defaults to "anthropic".
	Name string
	// RequestTimeout bounds one call when the caller sets no deadline.
	RequestTimeout time.Duration
}

// Provider is an Anthropic Messages provider.
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
		cfg.Name = "anthropic"
	}
	if cfg.Version == "" {
		cfg.Version = DefaultVersion
	}
	if cfg.MaxTokens <= 0 {
		cfg.MaxTokens = DefaultMaxTokens
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
	httpRequest, err := p.newRequest(ctx, body)
	if err != nil {
		return gai.Response{}, err
	}
	response, err := p.client.Do(httpRequest)
	if err != nil {
		return gai.Response{}, fmt.Errorf("gai/anthropic: request: %w", err)
	}
	defer response.Body.Close()
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		detail, _ := io.ReadAll(io.LimitReader(response.Body, 8<<10))
		return gai.Response{}, fmt.Errorf("gai/anthropic: %d %s: %s", response.StatusCode, response.Status,
			strings.TrimSpace(string(detail)))
	}

	var decoded messageResponse
	if err := json.NewDecoder(response.Body).Decode(&decoded); err != nil {
		return gai.Response{}, fmt.Errorf("gai/anthropic: decode: %w", err)
	}
	return decoded.toGAI()
}

// Stream implements gai.Provider.
func (p *Provider) Stream(ctx context.Context, request gai.Request) (gai.Stream, error) {
	body, err := p.marshal(request, true)
	if err != nil {
		return nil, err
	}
	httpRequest, err := p.newRequest(ctx, body)
	if err != nil {
		return nil, err
	}
	httpRequest.Header.Set("Accept", "text/event-stream")

	response, err := p.client.Do(httpRequest)
	if err != nil {
		return nil, fmt.Errorf("gai/anthropic: stream: %w", err)
	}
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		defer response.Body.Close()
		detail, _ := io.ReadAll(io.LimitReader(response.Body, 8<<10))
		return nil, fmt.Errorf("gai/anthropic: %d %s: %s", response.StatusCode, response.Status,
			strings.TrimSpace(string(detail)))
	}
	return newStream(response.Body), nil
}

func (p *Provider) newRequest(ctx context.Context, body []byte) (*http.Request, error) {
	httpRequest, err := http.NewRequestWithContext(ctx, http.MethodPost, p.cfg.BaseURL+"/v1/messages", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("gai/anthropic: build request: %w", err)
	}
	httpRequest.Header.Set("Content-Type", "application/json")
	httpRequest.Header.Set("anthropic-version", p.cfg.Version)
	if p.cfg.APIKey != "" {
		httpRequest.Header.Set("x-api-key", p.cfg.APIKey)
	}
	for name, value := range p.cfg.Headers {
		httpRequest.Header.Set(name, value)
	}
	return httpRequest, nil
}
