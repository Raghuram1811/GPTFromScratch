import requests

class OllamaClient:
    def __init__(self, local_url="http://localhost:11434"):
        self.local_url = local_url

    def get_models(self):
        response = requests.get(f"{self.local_url}/api/ps")
        if response.status_code == 200:
            return response.text
        else:
            raise Exception(f"Failed to get models: {response.status_code} - {response.text}")

    def chat(self, model, messages):
        payload = {
            "model": model,
            "messages": messages
        }
        response = requests.post(f"{self.local_url}/api/chat", json=payload)
        if response.status_code == 200:
            return response.text
        else:
            raise Exception(f"Failed to chat: {response.status_code} - {response.text}")    

def main():
    client = OllamaClient()

    # print(client.get_models())

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is moon?"}
    ]
    response = client.chat(model="gemma", messages=messages)
    print("Chat Response:", response)    


if __name__ == "__main__":
    main()

