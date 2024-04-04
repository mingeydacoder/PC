#執行此python啟動server
#開啟另一終端輸入 ngrok http 5000
#修改line bot後台 webhook URL

import random
import os
from openai import OpenAI

client = OpenAI(
    # This is the default and can be omitted
    api_key="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"
)

from flask import Flask, request

# 載入 LINE Message API 相關函式庫
from linebot import LineBotApi, WebhookHandler
from linebot.models import TextSendMessage   # 載入 TextSendMessage 模組
import json

app = Flask(__name__)

@app.route("/", methods=['POST'])
def linebot():
    body = request.get_data(as_text=True)
    json_data = json.loads(body)
    print(json_data)
    try:
        line_bot_api = LineBotApi('08whgj6BQSDSp6iYKPnaKdmoy2sRURJA4BO/7DDJY/UK53YKMhTPndJKHlSIuFtt/Mxj8XErN6Q6frwBveCwdckKqv4JvwwdswgZOu+yju2Za8/wAll3aslkPakQFefByb6WVzRb4MZDH+ONa5OvVQdB04t89/1O/w1cDnyilFU=')
        handler = WebhookHandler('50525437e78371febb45bc3db4a229dd')
        signature = request.headers['X-Line-Signature']
        handler.handle(body, signature)
        tk = json_data['events'][0]['replyToken']
        msg = json_data['events'][0]['message']['text']
        # 取出文字的前五個字元，轉換成小寫
        ai_msg = msg[:6].lower()
        draw_msg = msg[:1]
        reply_msg = ''
        # 取出文字的前五個字元是 hi ai:
        if ai_msg == 'hi ai:':
            # 將第六個字元之後的訊息發送給 OpenAI
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": msg[6:],
                    }
                ],
                model="gpt-3.5-turbo",
            )
            # 接收到回覆訊息後，移除換行符號
            reply_msg = chat_completion.choices[0].message.content
            #reply_msg = "worked"
        
        if draw_msg == '!' or draw_msg == '！':
            # 定義兩個選項
            option1 = "YES"
            option2 = "NO"

            # 隨機生成一個 0 或 1 的數字
            choice = random.randint(0, 1)

            # 根據隨機數字選擇選項
            if choice == 0:
                reply_msg = option1
            else:
                reply_msg = option2

        text_message = TextSendMessage(text=reply_msg)
        line_bot_api.reply_message(tk,text_message)
    except:
        print('error')
    return 'OK'

if __name__ == "__main__":
    app.run()