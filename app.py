import streamlit as st
from huggingface_hub import InferenceClient
from langchain import HuggingFaceHub
import requests
# Internal usage
import os
# Set HF API token  and HF repo
yourHFtoken = "hf_xxxxxx" #here your HF token

repo="HuggingFaceH4/starchat-beta"
### INITIALIZING STARCHAT FUNCTION MODEL
def starchat(model,myprompt, your_template):
    from langchain import PromptTemplate, LLMChain
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = yourHFtoken
    llm = HuggingFaceHub(repo_id=model , 
                         model_kwargs={"min_length":30,
                                       "max_new_tokens":256, "do_sample":True, 
                                       "temperature":0.2, "top_k":50, 
                                       "top_p":0.95, "eos_token_id":49155})
    template = your_template
    prompt = PromptTemplate(template=template, input_variables=["myprompt"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    llm_reply = llm_chain.run(myprompt)
    reply = llm_reply.partition('<|end|>')[0]
    return reply

myprompt = "List 3 benefits of vegan food"
my_template  = "<|system|>\n<|end|>\n<|user|>\n{myprompt}<|end|>\n<|assistant|>"

res  =  starchat(repo,myprompt, my_template)
print(myprompt)
print("---")
print(res)