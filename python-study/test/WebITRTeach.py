import base64
from zhipuai import ZhipuAI

img_path = "/python-study/test/大题1.png"
with open(img_path, 'rb') as img_file:
    img_base = base64.b64encode(img_file.read()).decode('utf-8')

client = ZhipuAI(api_key="d1461c2bc37615f2d68294d27f3b5427.X7g1mT8WcQYtZBGk")
response = client.chat.completions.create(
    model="glm-4v-plus",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": img_base
                    }
                },
                {
                    "type": "text",
                    "text": "识别图中的文字"
                }
            ]
        }
    ]
)

# 打印识别到的文字
# 假设response.choices[0].message是一个字典，并且包含一个content字段，该字段下有一个text字段
if response and response.choices and len(response.choices) > 0:
    message = response.choices[0].message
    if isinstance(message, dict) and 'content' in message:
        content = message['content']
        if isinstance(content, list) and len(content) > 0 and 'text' in content[0]:
            print(content[0]['text'])
        elif isinstance(content, dict) and 'text' in content:
            print(content['text'])
else:
    print("No text found in the response.")
