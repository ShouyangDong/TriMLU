import openai

class OpenAIModel:
    def __init__(self, api_key, model_id="gpt-4"):
        self.api_key = api_key
        self.model_id = model_id
        openai.api_key = self.api_key

    def generate(self, prompt):
        try:
            response = openai.ChatCompletion.create(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}]
            )
            return response['choices'][0]['message']['content']
        except Exception as e:
            print(f"❌ Error during OpenAI API call: {e}")
            return None