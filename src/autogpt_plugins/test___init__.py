import unittest
from typing import List
from unittest import mock
from auto_gpt_plugin_template import Message
from numpy.testing._private.parameterized import parameterized

from src.autogpt_plugins import AutoGPTLocalLLMPlugin
from . import local_llm_client


class AutoGPTLocalLLMPluginTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.plugin = AutoGPTLocalLLMPlugin()

    def test_can_handle_chat_completion(self):
        assert self.plugin.can_handle_chat_completion(mock.MagicMock(), mock.MagicMock(), mock.MagicMock(),
                                                      mock.MagicMock())


    # def test_handle_chat_completion(self):
    #     print("heelloo")
    #     message =
    #     try:
    #         result = self.plugin.handle_chat_completion(mock.MagicMock(), mock.MagicMock(), mock.MagicMock(),
    #                                                   mock.MagicMock())
    #         print(result)
    #     except Exception as e:
    #         self.fail(f"An unexpected exception occurred: {e}")

    def test_handle_chat_completion(self):
        testcases = []
        testcases.append([[{'role': 'system', 'content': '\nYour task is to devise up to 5 highly effective goals and an appropriate role-based name (_GPT) for an autonomous agent, ensuring that the goals are optimally aligned with the successful completion of its assigned task.\n\nThe user will provide the task, you will provide only the output in the exact format specified below with no explanation or conversation.\n\nExample input:\nHelp me with marketing my business\n\nExample output:\nName: CMOGPT\nDescription: a professional digital marketer AI that assists Solopreneurs in growing their businesses by providing world-class expertise in solving marketing problems for SaaS, content products, agencies, and more.\nGoals:\n- Engage in effective problem-solving, prioritization, planning, and supporting execution to address your marketing needs as your virtual Chief Marketing Officer.\n\n- Provide specific, actionable, and concise advice to help you make informed decisions without the use of platitudes or overly wordy explanations.\n\n- Identify and prioritize quick wins and cost-effective campaigns that maximize results with minimal time and budget investment.\n\n- Proactively take the lead in guiding you and offering suggestions when faced with unclear information or uncertainty to ensure your marketing strategy remains on track.\n'}, {'role': 'user', 'content': "Task: 'tell me today's weather in Taipei'\nRespond only with the output in the exact format specified in the system prompt, with no explanation or conversation.\n"}], "", 0, None])
        with mock.patch.dict('os.environ', {'LOCAL_LLM_API': 'http://127.0.0.1:5000/api'}):
            for message, model, temperature, max_tokens in testcases:
                try:
                    # self.plugin.handle_chat_completion(mock.MagicMock(), mock.MagicMock(), mock.MagicMock(),
                    #                                        mock.MagicMock())
                    result = self.plugin.handle_chat_completion(message, model, temperature, max_tokens)
                except Exception as e:
                    assert False
        assert True
