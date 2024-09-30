import base64
import requests
import re

def get_file_content_as_base64(file_path):
    with open(file_path, "rb") as file:
        return base64.b64encode(file.read()).decode()

def main():
    API_KEY = "fNk58Gl4gHlqXBWzLtrCj5pJ"
    SECRET_KEY = "WuKhQQw8J4O86K39DopzJgIJarTLRpsg"

    # 获取access_token
    token_url = "https://aip.baidubce.com/oauth/2.0/token"
    token_params = {
        "grant_type": "client_credentials",
        "client_id": API_KEY,
        "client_secret": SECRET_KEY
    }
    token_response = requests.post(token_url, data=token_params)
    access_token = token_response.json().get("access_token")

    if not access_token:
        print("获取access_token失败")
        return

    url = f"https://aip.baidubce.com/rest/2.0/ocr/v1/accurate_basic?access_token={access_token}"

    # 读取并编码图片
    image_base64 = get_file_content_as_base64("D:/code/Python/python-study/test/张三的答卷.png")

    # 构造请求数据
    payload = {
        "image": image_base64
    }
    headers = {
        "Content-Type": "application/x-www-form-urlencoded"
    }

    # 发出请求
    response = requests.post(url, data=payload, headers=headers)
    response_json = response.json()

    # 检查OCR结果
    if 'words_result' in response_json:
        ocr_text = ''.join([word['words'] for word in response_json['words_result']])
        # 使用正则表达式匹配形如 "_[A-D]_" 的模式
        pattern = r"_[A-D]_"
        matches = re.findall(pattern, ocr_text)
        # 将匹配到的答案放入数组中
        answers = [match[1] for match in matches]  # 只取中间的字母
        # 输出数组
        print(answers)
    else:
        print("未找到OCR结果")

if __name__ == "__main__":
    main()
