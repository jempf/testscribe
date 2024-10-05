import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer,StoppingCriteria,StoppingCriteriaList,pipeline
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.llms import HuggingFacePipeline
from langchain import PromptTemplate
from typing import List
import torch
# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

generation_config = model.generation_config
generation_config.temperature = 0
generation_config.num_return_sequences = 1
generation_config.max_new_tokens = 256
generation_config.use_cache = False
generation_config.repetition_penalty = 1.7
generation_config.pad_token_id = tokenizer.eos_token_id
generation_config.eos_token_id = tokenizer.eos_token_id
generation_config
stop_tokens = [["Human", ":"], ["AI", ":"]]

class StopGenerationCriteria(StoppingCriteria):
    def __init__(
        self, tokens: List[List[str]], tokenizer: AutoTokenizer, device: torch.device
    ):
        stop_token_ids = [tokenizer.convert_tokens_to_ids(t) for t in tokens]
        self.stop_token_ids = [
            torch.tensor(x, dtype=torch.long, device=device) for x in stop_token_ids
        ]

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        for stop_ids in self.stop_token_ids:
            if torch.eq(input_ids[0][-len(stop_ids) :], stop_ids).all():
                return True
        return False


stopping_criteria = StoppingCriteriaList(
    [StopGenerationCriteria(stop_tokens, tokenizer, model.device)]
)

class StopGenerationCriteria(StoppingCriteria):
    def __init__(
        self, tokens: List[List[str]], tokenizer: AutoTokenizer, device: torch.device
    ):
        stop_token_ids = [tokenizer.convert_tokens_to_ids(t) for t in tokens]
        self.stop_token_ids = [
            torch.tensor(x, dtype=torch.long, device=device) for x in stop_token_ids
        ]

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        for stop_ids in self.stop_token_ids:
            if torch.eq(input_ids[0][-len(stop_ids) :], stop_ids).all():
                return True
        return False


generation_pipeline = pipeline(
    model=model,
    tokenizer=tokenizer,
    return_full_text=True,
    task="text-generation",
    stopping_criteria=stopping_criteria,
    generation_config=generation_config,
)

llm = HuggingFacePipeline(pipeline=generation_pipeline)
template = """
The following 
Current conversation:

{history}

Human: {input}
AI:""".strip()
prompt = PromptTemplate(input_variables=["history", "input"], template=template)

memory = ConversationBufferWindowMemory(
    memory_key="history", k=6, return_only_outputs=True
)

chain = ConversationChain(
    llm=llm,
    prompt=prompt,
    verbose=True,
)

def generate_response(input_text):
    res=chain.invoke(input_text)
    print('response:',res)
    print(4444444444444444444444444444444444444444444444)
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(inputs.input_ids, max_length=50)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return res
 
iface = gr.Interface(fn=generate_response, inputs="text", outputs="text")
iface.launch()