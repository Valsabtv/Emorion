import discord 
from transformers import pipeline
import os
from dotenv import load_dotenv
import requests

load_dotenv()
discord_token = os.getenv("DISCORD_TOKEN")
openrouter_token = os.getenv("OPENROUTER_API_KEY")

classifier = pipeline('text-classification', model='j-hartmann/emotion-english-distilroberta-base')

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

api_url= 'https://openrouter.ai/api/v1/chat/completions'
headers = {
    'Authorization': f'Bearer {openrouter_token}',
    'Content-Type': 'application/json'
}

def chat_with_llama(emotion=str, message=str)->str:
    payload = {
        'model': 'meta-llama/llama-4-maverick:free',
        'messages': [
            
            {
                'role': 'system', 'content':(
                 'You are a emorion that can answer questions and help with tasks.')},
            {
                'role': 'user',
                'content': f"[Emotion: {emotion}]{message}"
            }
],
'temperature': 0.7,
'max_tokens': 1000,
    }

    resp = requests.post(api_url, headers=headers, json=payload)
    if resp.status_code == 200:
        return resp.json()['choices'][0]['message']['content'].strip()
    else:
        return f"Error: {resp.status_code} - {resp.text}"

@client.event
async def on_ready():
    print(f'We have logged in as {client}')

@client.event
async def on_message(message):
    if message.author == client.user:
        return
    
    result = classifier(message.content)
    emotion = result[0]['label']

    reply = chat_with_llama(emotion, message.content)
    await message.channel.send(reply)

client.run(discord_token)
    
