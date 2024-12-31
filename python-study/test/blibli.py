import requests
import json


def bv_to_av(bvid):
    # 将bvid转换为aid（oid）
    table = 'fZodR9XQDSUm21yCkr6zBqiveYah8bt4xsWpHnJE7jL5VG3guMTKNPAwcF'
    tr = {char: i for i, char in enumerate(table)}
    s = [11, 10, 3, 8, 4, 6]
    xor = 177451812
    add = 8728348608
    r = sum(tr[bvid[s[i]]] * 58 ** i for i in range(6))
    return (r - add) ^ xor


def get_video_info(bvid):
    url = f"https://api.bilibili.com/x/web-interface/view?bvid={bvid}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        if data['code'] == 0:
            return data['data']
        else:
            print(f"Error: {data['message']}")
            return None
    else:
        print(f"Failed to fetch data, status code: {response.status_code}")
        return None


def get_bilibili_comments(oid, page=1):
    url = f"https://api.bilibili.com/x/v2/reply?type=1&oid={oid}&pn={page}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        if data['code'] == 0:
            return data['data']['replies']
        else:
            print(f"Error: {data['message']}")
            return None
    else:
        print(f"Failed to fetch data, status code: {response.status_code}")
        return None


def save_comments_to_file(comments, filename="comments.json"):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(comments, f, ensure_ascii=False, indent=4)
    print(f"Comments saved to {filename}")


if __name__ == "__main__":
    bvid = "BV1j36nYtELW"

    # 获取视频信息
    video_info = get_video_info(bvid)
    if video_info:
        print(f"视频标题: {video_info['title']}")
        oid = video_info['aid']
        print(f"视频oid: {oid}")

        # 获取评论
        comments = get_bilibili_comments(oid, page=1)
        if comments:
            save_comments_to_file(comments)
            for comment in comments:
                print(f"用户: {comment['member']['uname']}")
                print(f"评论: {comment['content']['message']}")
                print("-" * 40)
        else:
            print("该视频没有评论或评论获取失败")
    else:
        print("无法获取视频信息")