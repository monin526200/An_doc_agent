from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openai import OpenAI

from config_loader import load_config


class llm_server:
    def __init__(self, use_api=False):
        self.use_api = use_api
        config = load_config()
        self._llm_config = config.llm

        if use_api:
            self._init_api_mode()
        else:
            self._init_local_mode()

    def _init_api_mode(self):
        errors = []
        if not self._llm_config.api.key:
            errors.append("  llm.api.key: 你的API密钥")
        if not self._llm_config.api.base_url:
            errors.append("  llm.api.base_url: 如 https://api.openai.com/v1")
        if not self._llm_config.api.model:
            errors.append("  llm.api.model: 如 gpt-3.5-turbo")

        if errors:
            raise ValueError("API模式需要配置，请在 config.yaml 中添加:\n" + "\n".join(errors))

        self.client = OpenAI(
            api_key=self._llm_config.api.key,
            base_url=self._llm_config.api.base_url,
        )
        self.api_model_name = self._llm_config.api.model

    def _init_local_mode(self):
        path = (self._llm_config.local.path or "").strip()
        if not path:
            raise ValueError("本地模式需要在 config.yaml 中设置 llm.local.path，并安装: pip install vllm")

        try:
            self.model = self.init_model(path)
        except ImportError as e:
            raise ValueError("vllm 未安装，运行: pip install vllm") from e

        try:
            from vllm import SamplingParams

            self.sampling_params = SamplingParams(
                temperature=0.8,
                top_p=0.95,
                max_tokens=128,
                stop=["<|endoftext|>", "<|user|>", "<|assistant|>", "<|system|>"],
            )
            self.tokenizer = self.model.get_tokenizer()
        except Exception as e:
            raise ValueError(f"vllm 初始化失败: {e}") from e

    def init_model(self, model_path: str):
        from vllm import LLM

        return LLM(
            model=model_path,
            trust_remote_code=True,
        )

    def reasoning(self, messages):

        if self.use_api:
            response = self.re_use_api(messages)
        else:
            response = self.re_wo_api(messages)
        return response

    def reasoning_wo_messages(self, user_prompt, system_prompt):
        messages = [
            {
                'role': 'system',
                'content': system_prompt
            },
            {
                'role': 'user',
                'content': user_prompt
            }
        ]
        if self.use_api:
            response = self.re_use_api(messages)
        else:
            response = self.re_wo_api(messages)
        return response

    def re_use_api(self, messages):

        completion = self.client.chat.completions.create(
            model=self.api_model_name,
            messages=messages

        )
        return completion.choices[0].message.content

    def re_wo_api(self, messages):
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        response = self.model.generate([prompt], self.sampling_params)

        req = response[0]
        parts = getattr(req, "outputs", None)
        if parts:
            return parts[0].text
        legacy = getattr(req, "output", None)
        if legacy:
            return legacy[0].text
        raise RuntimeError("unexpected vLLM response structure")
