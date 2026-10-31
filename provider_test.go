package gai

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"testing"
)

type echoProvider struct{}

func (echoProvider) Name() string { return "echo" }

func (echoProvider) Chat(_ context.Context, req Request) (Response, error) {
	content := ""
	if len(req.Messages) > 0 {
		content = req.Messages[len(req.Messages)-1].Content
	}
	return Response{Message: Message{Role: RoleAssistant, Content: content}}, nil
}

func (echoProvider) Stream(context.Context, Request) (Stream, error) {
	return &echoStream{chunks: []Chunk{{Delta: "hel"}, {Delta: "lo"}, {Done: true}}}, nil
}

type echoStream struct {
	chunks []Chunk
	index  int
}

func (s *echoStream) Recv() (Chunk, error) {
	if s.index >= len(s.chunks) {
		return Chunk{}, io.EOF
	}
	chunk := s.chunks[s.index]
	s.index++
	return chunk, nil
}

func (*echoStream) Close() error { return nil }

func TestRegistryAndToolset(t *testing.T) {
	registry := NewRegistry()
	if err := registry.Register(echoProvider{}); err != nil {
		t.Fatal(err)
	}
	if err := registry.Register(echoProvider{}); err == nil {
		t.Fatal("expected duplicate provider error")
	}
	if names := registry.Names(); len(names) != 1 || names[0] != "echo" {
		t.Fatalf("names = %v", names)
	}

	tools := NewToolset()
	if err := tools.Register(Tool{
		Name:   "echo",
		Schema: json.RawMessage(`{"type":"object"}`),
		Handler: func(_ context.Context, args json.RawMessage) (any, error) {
			return string(args), nil
		},
	}); err != nil {
		t.Fatal(err)
	}
	result, err := tools.Call(context.Background(), "echo", json.RawMessage(`{"x":1}`))
	if err != nil {
		t.Fatal(err)
	}
	if result != `{"x":1}` {
		t.Fatalf("result = %v", result)
	}
	if _, err := tools.Call(context.Background(), "missing", nil); err == nil {
		t.Fatal("expected unknown tool error")
	}
	if err := tools.Register(Tool{Name: "bad", Schema: json.RawMessage(`{`)}); err == nil {
		t.Fatal("expected invalid schema error")
	}
}

func TestCollect(t *testing.T) {
	response, err := Collect(&echoStream{chunks: []Chunk{{Delta: "a"}, {Delta: "b"}, {Done: true}}})
	if err != nil {
		t.Fatal(err)
	}
	if response.Message.Content != "ab" || response.Message.Role != RoleAssistant {
		t.Fatalf("response = %+v", response)
	}
	if _, err := Collect(nil); err == nil {
		t.Fatal("expected nil stream error")
	}
}

type failingStream struct{}

func (failingStream) Recv() (Chunk, error) { return Chunk{}, errors.New("boom") }
func (failingStream) Close() error         { return nil }

func TestCollectPropagatesStreamError(t *testing.T) {
	if _, err := Collect(failingStream{}); err == nil {
		t.Fatal("expected stream error")
	}
}
