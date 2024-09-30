import base64
from pylatexenc.latex2text import LatexNodes2Text
from zhipuai import ZhipuAI

def extract_text_from_image(client, img_path):
    with open(img_path, 'rb') as img_file:
        img_base = base64.b64encode(img_file.read()).decode('utf-8')

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
                        "text": "提取并识别图片中的文字"
                    }
                ]
            }
        ]
    )

    # 获取并处理响应信息
    latex_text = response.choices[0].message.content
    text = LatexNodes2Text().latex_to_text(latex_text)
    return text

# 图片路径
img_path1 = "D:/code/Python/python-study/test/大题1.png"  # 标准答案
img_path2 = "D:/code/Python/python-study/test/大题2.png"  # 学生答案

# 初始化ZhipuAI客户端
client = ZhipuAI(api_key="874860d429ffbdb79a9c04a5e330b91a.KJGv79Dt2MXAG7LH")

# 提取文本
std_text = extract_text_from_image(client, img_path1)
stu_text = extract_text_from_image(client, img_path2)

# 输出两张图的识别结果
print("标准答案的识别结果:")
print(std_text)
print("\n学生答案的识别结果:")
print(stu_text)

client = ZhipuAI(api_key="874860d429ffbdb79a9c04a5e330b91a.KJGv79Dt2MXAG7LH")  # 请填写您自己的APIKey
response = client.chat.completions.create(
    model="glm-4",  # 请填写您要调用的模型名称
    messages=[
        {"role": "user", "content": "第一段为标准答案："+std_text+",第二段为学生答案："+stu_text+",请根据图中每道题的给分信息结合你的理解写出该学生的得分"},
    ],
)
print(response.choices[0].message)