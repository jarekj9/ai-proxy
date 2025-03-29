import os
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.inference.models import SystemMessage, UserMessage
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class AzureOpenAIClient:
    def __init__(self):
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.default_model = os.getenv("DEFAULT_MODEL", "gpt-4o-mini")
        
        if not self.api_key or not self.endpoint:
            raise ValueError("Azure OpenAI configuration is missing")
            
        self.client = ChatCompletionsClient(
            endpoint=self.endpoint,
            credential=AzureKeyCredential(self.api_key),
        )
    
    def convert_messages(self, messages):
        """Convert OpenAI format messages to Azure AI SDK format"""
        converted_messages = []
        for msg in messages:
            if msg["role"] == "system":
                converted_messages.append(SystemMessage(content=msg["content"]))
            elif msg["role"] == "user":
                converted_messages.append(UserMessage(content=msg["content"]))
        return converted_messages
    
    def format_response(self, response, model):
        """Convert Azure AI SDK response to OpenAI format"""
        return {
            "id": "chatcmpl-" + str(hash(str(response))),
            "object": "chat.completion",
            "created": 0,
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response.choices[0].message.content
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }
    
    def complete(self, request):
        """Process a chat completion request"""
        try:
            # Convert messages to the correct format
            messages = self.convert_messages(request.get("messages", []))
            
            # Get model from request or use default
            model = request.get("deployment_id", self.default_model)
            
            # Make the request
            response = self.client.complete(
                messages=messages,
                max_tokens=request.get("max_tokens", 4096),
                temperature=request.get("temperature", 1.0),
                top_p=request.get("top_p", 1.0),
                model=model
            )
            
            # Format and return the response
            return self.format_response(response, model)
            
        except Exception as e:
            raise Exception(f"Error in chat completion: {str(e)}")

# Example usage when running directly
if __name__ == "__main__":
    try:
        client = AzureOpenAIClient()
        
        # Example request
        test_request = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": "I am going to Paris, what should I see?"
                }
            ],
            "deployment_id": "gpt-4o-mini",
            "max_tokens": 4096,
            "temperature": 1.0,
            "top_p": 1.0
        }
        
        # Make the request
        response = client.complete(test_request)
        
        # Print the response
        print("Response:")
        print(response["choices"][0]["message"]["content"])
        
    except Exception as e:
        print(f"Error: {str(e)}") 