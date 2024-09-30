import re

# 示例OCR识别后的文本
ocr_text = """
2024 年上海第二工业大学计算机组成原理期末试卷（A 卷）
题目一：
解：这是题目一的解答。
题目二：
解：这是题目二的解答，继续往下写。
解：这是题目二的第二部分解答。
题目三：
解：这是题目三的解答。
"""

# 查找简答题的答案，以“解：”开头
def find_answers(text):
    # 使用正则表达式匹配以“解：”开头的部分
    answer_pattern = r"解：[\s\S]*?(?=\n\S|$)"  # 匹配“解：”后直到下一个非空行或文件结束
    answers = re.findall(answer_pattern, text)
    return answers

# 测试代码
def main():
    # OCR 识别后的文本内容
    text = ocr_text

    # 查找简答题答案
    answers = find_answers(text)

    if not answers:
        print("未识别到简答题答案。")
    else:
        print(f"识别到 {len(answers)} 个简答题答案：")
        for idx, answer in enumerate(answers, 1):
            print(f"答案 {idx}:")
            print(answer)

if __name__ == "__main__":
    main()
