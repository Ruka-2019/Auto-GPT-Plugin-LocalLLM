import json
import os
from typing import List

from auto_gpt_plugin_template import Message
import requests

class LocalLLMClient():
    def build_prompt(self, messages: List[Message]) -> str:
        # llm_prompt_type = os.getenv("LOCAL_LLM_TYPE")
        # if llm_prompt_type == "":
        #     llm_prompt_type = "llama"
        prompt = ""
        ai_character_name = "assistant"
        for message in messages:
            prompt += f"\n$@${message['role']}:$@${message['content']}\n"
        prompt += f"$@${ai_character_name}:$@$"
        return prompt

    def build_payload(self, messages: List[Message], model: str, temperature: float, max_tokens: int) -> dict:
        prompt = self.build_prompt(messages)
        payload = {}
        payload["prompt"] = prompt
        if temperature == 0:
            temperature = 0.001
        payload["temperature"] = temperature
        payload["sampler_order"]=[6,0,1,2,3,4,5]
        return payload

    def post_request(self, payload: dict):
        api_url = os.getenv("LOCAL_LLM_API")
        generate_url = "/v1/generate"
        json_payload = json.dumps(payload)
        headers = {
            'Content-Type': 'application/json'
        }
        response = requests.post(api_url+generate_url, data=json_payload, headers=headers)
        return response

    def llm_client(self, messages: List[Message], model: str, temperature: float, max_tokens: int) -> str:
        payload = self.build_payload(messages, model, temperature, max_tokens)
        response = self.post_request(payload)
        return response.json()['results'][0]