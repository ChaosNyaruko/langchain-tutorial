Just follow [Official Doc](https://python.langchain.com/docs/get_started/quickstart/) and [YouTube](https://www.youtube.com/watch?v=_v_fgW2SkkQ&list=PLqZXAkvF1bPNQER9mLmDbntNfSpzdDIU5) to know about LangChain, not many interesting things.

# Notes
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
