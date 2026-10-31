package main

import (
	"context"
	"fmt"
	"io"
	"log"

	"github.com/maxid/gai"
)

type echoProvider struct{}

func (echoProvider) Name() string { return "echo" }

func (echoProvider) Chat(_ context.Context, req gai.Request) (gai.Response, error) {
	content := ""
	if len(req.Messages) > 0 {
		content = req.Messages[len(req.Messages)-1].Content
	}
	return gai.Response{Message: gai.Message{Role: gai.RoleAssistant, Content: content}}, nil
}

func (echoProvider) Stream(context.Context, gai.Request) (gai.Stream, error) {
	return &stream{chunks: []gai.Chunk{{Delta: "hello "}, {Delta: "world"}, {Done: true}}}, nil
}

type stream struct {
	chunks []gai.Chunk
	index  int
}

func (s *stream) Recv() (gai.Chunk, error) {
	if s.index >= len(s.chunks) {
		return gai.Chunk{}, io.EOF
	}
	chunk := s.chunks[s.index]
	s.index++
	return chunk, nil
}

func (*stream) Close() error { return nil }

func main() {
	provider := echoProvider{}
	response, err := provider.Chat(context.Background(), gai.Request{
		Messages: []gai.Message{{Role: gai.RoleUser, Content: "ping"}},
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("chat:", response.Message.Content)

	stream, err := provider.Stream(context.Background(), gai.Request{})
	if err != nil {
		log.Fatal(err)
	}
	collected, err := gai.Collect(stream)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("stream:", collected.Message.Content)
}
