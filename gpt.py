import requests


api_url = 'https://www.gptapi.us/v1/chat/completions'
api_key = 'sk-cd7Tf71ItsyQK7RlC2E78aD6D5C3439b936545E71900DdEb'


def get_answer(user_content, model="gpt-4o-mini"):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    data = {'model': model,
        'messages': [{'role': 'user','content': user_content}]
        }
    response = requests.post(api_url, headers=headers, json=data)
    response =response.json()
    return response['choices'][0]['message']['content']


if __name__ == '__main__':
    text = 'who is Trump?'
    prompt = 'You are a healthcare assistant. Please try to answer questions related to healthcare as much as possible. '
    answer = get_answer(prompt + text)
    print(answer)