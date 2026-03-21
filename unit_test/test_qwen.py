from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()
client = OpenAI(
    # If the environment variable is not set, replace the following line with: api_key="sk-xxx"
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
)
completion = client.chat.completions.create(
model="qwen3-vl-flash-2026-01-22",  # Model list: https://www.alibabacloud.com/help/en/model-studio/models
messages=[{"role": "user","content": [
        {"type": "image_url",
        "image_url": {"url": "https://dashscope.oss-cn-beijing.aliyuncs.com/images/dog_and_girl.jpeg"}},
        {"type": "text", "text": "What is this?"},
        ]}]
)
print(completion.model_dump_json())