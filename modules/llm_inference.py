import re
from transformers import AutoTokenizer, AutoModelForCausalLM

LLM_MODEL_DIR = "/home/asr_tts/.virtualenvs/NEW_ASR_TTS/models/Qwen2.5-1.5B-Instruct"
model_llm = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL_DIR,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_DIR, trust_remote_code=True)

class ChatMemory:
    def __init__(self, max_length=1024):
        self.history = []
        self.max_length = max_length

    def add(self, user_input, model_resp):
        self.history.append((user_input, model_resp))
        if len(self.history) > self.max_length:
            self.history = self.history[-self.max_length:]

    def to_messages(self, system_prompt: str):
        messages = [{"role": "system", "content": system_prompt}]
        for user_input, assistant_output in self.history:
            messages.append({"role": "user", "content": user_input})
            if assistant_output:
                messages.append({"role": "assistant", "content": assistant_output})
        return messages

memory = ChatMemory()

def clean_response(text: str) -> str:
    text = re.split(r"(User:|用户:)", text)[0]
    text = re.sub(r"^Assistant[:：]?\s*", "", text.strip(), flags=re.IGNORECASE)
    return text.strip()

async def generate_reply_from_text(prompt: str) -> str:
    sys_prompt = (
        "你叫小善，你是一个语气甜美活泼开朗的智能语音助手，你输出内容简洁凝炼，"
        "你的回答会突出重点，同时也会体会用户的情绪"
        "回答时不要以“系统”两个字回复，并且不会超过80个中文字数。"
    )
    messages = memory.to_messages(sys_prompt)
    messages.append({"role": "user", "content": prompt})

    text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text_input], return_tensors="pt").to(model_llm.device)

    response = ""
    prev_len = 0
    for output in model_llm.generate(
            **model_inputs,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            streaming=True
    ):
        new_tokens = output[0][prev_len:]
        token_str = tokenizer.decode(new_tokens, skip_special_tokens=True)
        response += token_str
        prev_len = len(output[0])

    response = clean_response(response)
    memory.add(prompt, response)
    return response
