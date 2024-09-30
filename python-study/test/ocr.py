import base64
from pylatexenc.latex2text import LatexNodes2Text
from zhipuai import ZhipuAI

img_path = "/python-study/test/大题1.png"  # 图片路径
with open(img_path, 'rb') as img_file:
    img_base = base64.b64encode(img_file.read()).decode('utf-8')

client = ZhipuAI(api_key="874860d429ffbdb79a9c04a5e330b91a.KJGv79Dt2MXAG7LH")  # API Key
response = client.chat.completions.create(
    model="glm-4v-plus",  # 模型名称
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
                    "text": "提取并识别图片中'解：'之后的文字"
                }
            ]
        }
    ]
)

# 获取并处理响应信息
latex_text = response.choices[0].message.content  # 使用 .content 属性获取文本
text = LatexNodes2Text().latex_to_text(latex_text)  # 将 LaTeX 文本转换为普通文本
print(text)
