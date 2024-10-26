// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Command ragserver is an HTTP server that implements RAG (Retrieval
// Augmented Generation) using the Gemini model and Weaviate. See the
// accompanying README file for additional details.
package main

import (
	"bytes"
	"cmp"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strings"

	"github.com/weaviate/weaviate-go-client/v4/weaviate"
	"github.com/weaviate/weaviate-go-client/v4/weaviate/graphql"
	"github.com/weaviate/weaviate/entities/models"
)

const generativeModelName = "llama3"
const embeddingModelName = "llama3"

// This is a standard Go HTTP server. Server state is in the ragServer struct.
// The `main` function connects to the required services (Weaviate and Google
// AI), initializes the server state and registers HTTP handlers.
func main() {
	ctx := context.Background()
	wvClient, err := initWeaviate(ctx)
	if err != nil {
		log.Fatal(err)
	}

	m := &Ollama{
		genModel:   generativeModelName,
		embedModel: embeddingModelName,
	}
	server := &ragServer{
		ctx:      ctx,
		wvClient: wvClient,
		genModel: m,
		embModel: m,
	}

	mux := http.NewServeMux()
	mux.HandleFunc("POST /add/", server.addDocumentsHandler)
	mux.HandleFunc("POST /query/", server.queryHandler)

	port := cmp.Or(os.Getenv("SERVERPORT"), "9020")
	address := "localhost:" + port
	log.Println("listening on", address)
	log.Fatal(http.ListenAndServe(address, mux))
}

type Ollama struct {
	genModel   string
	embedModel string
}

func (m *Ollama) Generate(ctx context.Context, query string) (string, error) {
	data := map[string]any{"model": m.genModel, "prompt": query, "stream": false}
	js, err := json.Marshal(data)
	if err != nil {
		return "", fmt.Errorf("Marshal err: %v", err)
	}
	resp, err := http.Post("http://localhost:11434/api/generate", "application/json", bytes.NewReader(js))
	if err != nil {
		return "", fmt.Errorf("ollama generate err: %v", err)
	}
	d, err := io.ReadAll(resp.Body)
	if err != nil {
		panic(err)
	}
	type Resp struct {
		Model    string
		Response string
	}
	var t Resp
	if err = json.Unmarshal(d, &t); err != nil {
		panic(err)
	}
	return t.Response, nil
}

func (m *Ollama) Embedding(ctx context.Context, docs []string) ([][]float32, error) {
	data := map[string]any{"model": m.genModel, "input": docs}
	js, err := json.Marshal(data)
	if err != nil {
		return nil, fmt.Errorf("Marshal err: %v", err)
	}
	resp, err := http.Post("http://localhost:11434/api/embed", "application/json", bytes.NewReader(js))
	if err != nil {
		return nil, fmt.Errorf("ollama embedding err: %v", err)
	}
	d, err := io.ReadAll(resp.Body)
	if err != nil {
		panic(err)
	}
	type Resp struct {
		Model      string
		Embeddings [][]float32
	}
	var t Resp
	if err = json.Unmarshal(d, &t); err != nil {
		panic(err)
	}
	return t.Embeddings, nil
}

type LLM interface {
	Generate(context.Context, string) (string, error)
}

type Embed interface {
	Embedding(context.Context, []string) ([][]float32, error)
}

type ragServer struct {
	ctx      context.Context
	wvClient *weaviate.Client
	genModel LLM
	embModel Embed
}

func (rs *ragServer) addDocumentsHandler(w http.ResponseWriter, req *http.Request) {
	// Parse HTTP request from JSON.
	type document struct {
		Text string
	}
	type addRequest struct {
		Documents []document
	}
	ar := &addRequest{}

	err := readRequestJSON(req, ar)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	batch := []string{}
	for _, doc := range ar.Documents {
		batch = append(batch, doc.Text)
	}
	log.Printf("invoking embedding model with %v documents", len(ar.Documents))
	rsp, err := rs.embModel.Embedding(rs.ctx, batch)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	if len(rsp) != len(ar.Documents) {
		http.Error(w, "embedded batch size mismatch", http.StatusInternalServerError)
		return
	}

	// Convert our documents - along with their embedding vectors - into types
	// used by the Weaviate client library.
	objects := make([]*models.Object, len(ar.Documents))
	for i, doc := range ar.Documents {
		objects[i] = &models.Object{
			Class: "Document",
			Properties: map[string]any{
				"text": doc.Text,
			},
			Vector: rsp[i],
		}
	}

	// Store documents with embeddings in the Weaviate DB.
	log.Printf("storing %v objects in weaviate", len(objects))
	_, err = rs.wvClient.Batch().ObjectsBatcher().WithObjects(objects...).Do(rs.ctx)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
}

func (rs *ragServer) queryHandler(w http.ResponseWriter, req *http.Request) {
	// Parse HTTP request from JSON.
	type queryRequest struct {
		Content string
	}
	qr := &queryRequest{}
	err := readRequestJSON(req, qr)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	// Embed the query contents.
	rsp, err := rs.embModel.Embedding(rs.ctx, []string{qr.Content})
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// Search weaviate to find the most relevant (closest in vector space)
	// documents to the query.
	gql := rs.wvClient.GraphQL()
	result, err := gql.Get().
		WithNearVector(
			gql.NearVectorArgBuilder().WithVector(rsp[0])).
		WithClassName("Document").
		WithFields(graphql.Field{Name: "text"}).
		WithLimit(3).
		Do(rs.ctx)
	if werr := combinedWeaviateError(result, err); werr != nil {
		http.Error(w, werr.Error(), http.StatusInternalServerError)
		return
	}

	contents, err := decodeGetResults(result)
	if err != nil {
		http.Error(w, fmt.Errorf("reading weaviate response: %w", err).Error(), http.StatusInternalServerError)
		return
	}
	log.Printf("query: %q might matches these 3 docs: %+v", qr.Content, strings.Join(contents, "\n"))
	// Create a RAG query for the LLM with the most relevant documents as
	// context.
	ragQuery := fmt.Sprintf(ragTemplateStr, qr.Content, strings.Join(contents, "\n"))
	resp, err := rs.genModel.Generate(rs.ctx, ragQuery)
	if err != nil {
		log.Printf("calling generative model: %v", err.Error())
		http.Error(w, "generative model error", http.StatusInternalServerError)
		return
	}

	if resp == "" {
		http.Error(w, "generative model error", http.StatusInternalServerError)
		return
	}

	var respTexts []string = []string{resp}

	renderJSON(w, strings.Join(respTexts, "\n"))
}

const ragTemplateStr = `
I will ask you a question and will provide some additional context information.
Assume this context information is factual and correct, as part of internal
documentation.
If the question relates to the context, answer it using the context.
If the question does not relate to the context, answer it as normal.

For example, let's say the context has nothing in it about tropical flowers;
then if I ask you about tropical flowers, just answer what you know about them
without referring to the context.

For example, if the context does mention minerology and I ask you about that,
provide information from the context along with general knowledge.

Question:
%s

Context:
%s
`

// decodeGetResults decodes the result returned by Weaviate's GraphQL Get
// query; these are returned as a nested map[string]any (just like JSON
// unmarshaled into a map[string]any). We have to extract all document contents
// as a list of strings.
func decodeGetResults(result *models.GraphQLResponse) ([]string, error) {
	data, ok := result.Data["Get"]
	if !ok {
		return nil, fmt.Errorf("Get key not found in result")
	}
	doc, ok := data.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("Get key unexpected type")
	}
	slc, ok := doc["Document"].([]any)
	if !ok {
		return nil, fmt.Errorf("Document is not a list of results")
	}

	var out []string
	for _, s := range slc {
		smap, ok := s.(map[string]any)
		if !ok {
			return nil, fmt.Errorf("invalid element in list of documents")
		}
		s, ok := smap["text"].(string)
		if !ok {
			return nil, fmt.Errorf("expected string in list of documents")
		}
		out = append(out, s)
	}
	return out, nil
}
