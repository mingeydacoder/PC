import os
from openai import OpenAI

client = OpenAI(
    # This is the default and can be omitted
    api_key="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"
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "how are you",
        }
    ],
    model="gpt-3.5-turbo",
)

print(chat_completion.choices[0].message.content)