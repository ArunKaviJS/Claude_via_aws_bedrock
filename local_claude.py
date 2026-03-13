import boto3
import json
import os
from dotenv import load_dotenv

load_dotenv()

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = "ap-south-1"  # ✅ Changed to ap-south-1 (Mumbai)

def ask_claude(question):
    client = boto3.client(
        "bedrock-runtime",
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    )
    
    try:
        response = client.invoke_model(
            modelId='global.anthropic.claude-sonnet-4-6',  # ✅ Correct ID
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1024,
                "messages": [
                    {"role": "user", "content": question}
                ]
            })
        )
        
        result = json.loads(response['body'].read())
        return result['content'][0]['text']
        
    except Exception as e:
        return f"Error: {str(e)}"


def lambda_handler(event, context):
    # Get question from event or use default
    question = event.get('question', 'Hello! Who are you?')
    
    answer = ask_claude(question)
    
    return {
        'statusCode': 200,
        'body': answer
    }


# ✅ For running locally (not in Lambda)
if __name__ == "__main__":
    while True:
        question = input("\n🙋 Ask Claude: ")
        
        if question.lower() in ['exit', 'quit', 'q']:
            print("Goodbye! 👋")
            break
            
        print("\n🤖 Claude:", ask_claude(question))