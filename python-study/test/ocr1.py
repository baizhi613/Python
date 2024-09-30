import requests
import base64
import re
from PIL import Image

# 替换成你的百度OCR的 API_KEY 和 SECRET_KEY
API_KEY = "fNk58Gl4gHlqXBWzLtrCj5pJ"
SECRET_KEY = "WuKhQQw8J4O86K39DopzJgIJarTLRpsg"


# 获取百度OCR的Access Token
def get_access_token():
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {
        "grant_type": "client_credentials",
        "client_id": API_KEY,
        "client_secret": SECRET_KEY
    }
    response = requests.get(url, params=params)

    # 打印出完整的响应内容，帮助排查问题
    print("Access Token Response:", response.json())

    if response.status_code != 200:
        raise Exception("获取 Access Token 失败，请检查 API_KEY 和 SECRET_KEY 是否正确")

    return response.json().get("access_token")

# 使用OCR提取图片中的文字和位置信息
def ocr_image_with_location(image_path):
    with open(image_path, "rb") as f:
        image_data = f.read()

    image_base64 = base64.b64encode(image_data).decode()

    access_token = get_access_token()
    ocr_url = f"https://aip.baidubce.com/rest/2.0/ocr/v1/accurate?access_token={access_token}"

    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    payload = {'image': image_base64}

    response = requests.post(ocr_url, data=payload, headers=headers)

    # 打印出完整的响应内容，帮助排查问题
    print("OCR Response:", response.json())

    if response.status_code != 200 or 'error_code' in response.json():
        raise Exception("OCR请求失败，请检查 API 调用和图片内容")

    return response.json()

# 查找每道题的题号位置
def find_question_positions(words_info):
    question_positions = []
    question_pattern = r"^\d+[、.]"  # 匹配题号，如 '1. ', '2. '

    for word_info in words_info:
        text = word_info['words']
        if re.match(question_pattern, text):
            # 记录题号及其位置信息
            question_positions.append({
                "question": text,
                "location": word_info['location']
            })

    return question_positions


# 根据题号位置裁剪图片
def crop_question_images(image_path, question_positions):
    image = Image.open(image_path)
    cropped_images = []

    for i in range(len(question_positions)):
        top = question_positions[i]['location']['top']
        if i == len(question_positions) - 1:
            bottom = image.height  # 最后一题直到图片底部
        else:
            bottom = question_positions[i + 1]['location']['top']  # 下一道题的顶部作为当前题的底部

        # 裁剪范围（左, 上, 右, 下）
        box = (0, top, image.width, bottom)
        cropped_image = image.crop(box)
        cropped_images.append(cropped_image)

    return cropped_images


# 保存分割出的题目图片
def save_cropped_images(cropped_images):
    for idx, cropped_image in enumerate(cropped_images, 1):
        filename = f'question_{idx}.png'
        cropped_image.save(filename)
        print(f"保存题目图片: {filename}")


# 主函数，执行OCR、识别题号并分割图片
def main(image_path):
    print("正在执行OCR...")
    ocr_result = ocr_image_with_location(image_path)

    if "words_result" not in ocr_result:
        print("OCR失败，请检查图片或API设置。")
        return

    words_info = ocr_result['words_result']
    question_positions = find_question_positions(words_info)

    if not question_positions:
        print("未识别到题号，请检查图片内容或正则表达式。")
        return

    print(f"识别到 {len(question_positions)} 道题目，开始分割图片...")
    cropped_images = crop_question_images(image_path, question_positions)

    save_cropped_images(cropped_images)
    print("图片分割完成！")


if __name__ == "__main__":
    # 指定答卷图片的路径
    image_path = "D:/code/Python/python-study/test/选择题.png"

    # 执行分割流程
    main(image_path)
