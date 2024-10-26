
# LangChain Notes
## How to play with it
```bash 
python translater.py
``` 
Then visit follow urls for fun:
- http://localhost:8000/chain/playground/
- http://localhost:8000/qa/playground/
- http://localhost:8000/chainv1/playground/

Just follow [Official Doc](https://python.langchain.com/docs/get_started/quickstart/) and [YouTube](https://www.youtube.com/watch?v=_v_fgW2SkkQ&list=PLqZXAkvF1bPNQER9mLmDbntNfSpzdDIU5) to know about LangChain, not many interesting things.

Something such as APIs are deprecated, e.g. `Chain.run` -> `Chain.invoke`

Probably without OpenAI API usage, using open source model instead, with the guidance of [Official documentation](https://python.langchain.com/docs/get_started/quickstart/)

[Cookbook](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbVMzem5UR2ozNVJTenJZSnVNVkFQRkd2b3RGQXxBQ3Jtc0ttLURKWEVXNzBzMGhHenZhcW4taVZ3djhGdUVpQjJNZXlaQVVoQUF0a01NOW81cEoxLXdzeFo4QmtkV3M2cHMxcEhnSkNYVi1fZ2E5R093REFWRERMTVBHT0NtTTVIaU5RMlJWUC1WZUlrYi1kelBPaw&q=https%3A%2F%2Fgithub.com%2Fgkamradt%2Flangchain-tutorials%2Fblob%2Fmain%2FLangChain%2520Cookbook%2520Part%25201%2520-%2520Fundamentals.ipynb&v=2xxziIWmaSA)



# How to change your kernel in jupyter notebook to make it consistent with conda environment
1. Install a kernel for the environment by ipykernel
https://stackoverflow.com/questions/53004311/how-to-add-conda-environment-to-jupyter-lab
```bash
$ conda activate cenv           # . ./cenv/bin/activate in case of virtualenv
(cenv)$ conda install ipykernel
(cenv)$ ipython kernel install --user --name=<any_name_for_kernel>
(cenv)$ conda deactivate
```
2. change the kernel in the jupyterlab web page.


# RAG server in Go
reference: https://go.dev/blog/llmpowered
## How to run it
1. `cd` into the project directory.
2. Run the weaviate vector DB in docker.
```
docker-compose up
```
3. Run the RAG server
```
go run .
```
4. Test it!
```
./weaviate-delete-objects.sh # delete all existed docs.
./add-documents.sh # add some demo docs in the weaviate db by our server.
./weaviate-show-all.sh # check all your recorded docs.
./query.sh '<YOUR QUERY>'
```
## Others
1. Calculates the question’s vector embedding using the embedding model.
2. Uses the vector DB’s similarity search to find the most relevant documents to the question in the knowledge database.
3. Uses simple prompt engineering to reformulate the question with the most relevant documents found in step (2) as context, and sends it to the LLM, returning its answer to the user.


Embedding: text ->  vector. e.g. "I'm an engineer" -> [1.1, 2.3, 0.5, 0.8]

Pseudo Code:
```
for i, doc in docs:
  vs[i] = embed(doc)
  (vs[i], doc) -> Vector DB

input(q)

vq = embed(q)

Weaviate DB tries to find several(one or more) "nearest" and most "relevant" docs by "maybe" the distance bewteen vq and vs[0..5]  -> v[0] v[1] v[3]

`我有以下一些背景 v[0].doc v[1].doc v[3].doc，请回答我一个问题: q` 
    -> LLM 
```
