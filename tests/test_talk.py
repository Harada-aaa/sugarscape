import unittest
from unittest.mock import MagicMock, patch
import json
import urllib.error

import vllm_client
import ollama_client
import gemini_client
import sugarscape
import agent
import cell
import environment


class TestLLMClientsTalk(unittest.TestCase):
    def test_vllm_client_talk_success(self):
        client = vllm_client.VLLMClient("http://127.0.0.1:8000/v1", "test-model", 10)
        mock_response = {
            "dialogue": [
                {"speaker": 0, "message": "Hello from agent 0!"},
                {"speaker": 1, "message": "Greetings, agent 0!"}
            ]
        }
        with patch.object(client, "_request", return_value=mock_response):
            result = client.talk({"id": 0}, {"id": 1})
            self.assertEqual(len(result), 2)
            self.assertEqual(result[0]["speaker"], 0)
            self.assertEqual(result[0]["message"], "Hello from agent 0!")
            self.assertEqual(result[1]["speaker"], 1)

    def test_vllm_client_talk_alt_format(self):
        client = vllm_client.VLLMClient("http://127.0.0.1:8000/v1", "test-model", 10)
        mock_response = {
            "speaker_message": "Do you have sugar?",
            "listener_message": "Yes, I have 10 sugar."
        }
        with patch.object(client, "_request", return_value=mock_response):
            result = client.talk({"id": 0}, {"id": 1})
            self.assertEqual(len(result), 2)
            self.assertEqual(result[0]["speaker"], 0)
            self.assertEqual(result[0]["message"], "Do you have sugar?")
            self.assertEqual(result[1]["speaker"], 1)

    def test_vllm_client_talk_failure(self):
        client = vllm_client.VLLMClient("http://127.0.0.1:8000/v1", "test-model", 10)
        with patch.object(client, "_request", side_effect=urllib.error.URLError("Connection refused")):
            result = client.talk({"id": 0}, {"id": 1})
            self.assertIsNone(result)

    def test_ollama_client_talk_success(self):
        client = ollama_client.OllamaClient("http://127.0.0.1:11434", "test-model", 10)
        mock_response = {
            "dialogue": [
                {"speaker": 5, "message": "Nice weather on this cell."},
                {"speaker": 6, "message": "Indeed it is."}
            ]
        }
        with patch.object(client, "_request", return_value=mock_response):
            result = client.talk({"id": 5}, {"id": 6})
            self.assertEqual(len(result), 2)
            self.assertEqual(result[0]["speaker"], 5)

    def test_ollama_client_talk_parallel(self):
        client = ollama_client.OllamaClient("http://127.0.0.1:11434", "test-model", 10)
        mock_response = {"dialogue": [{"speaker": 0, "message": "Hi"}]}
        with patch.object(client, "_request", return_value=mock_response):
            pairs = [({"id": 0}, {"id": 1}), ({"id": 2}, {"id": 3})]
            results = client.talk_parallel(pairs)
            self.assertEqual(len(results), 2)
            self.assertEqual(results[0][0]["message"], "Hi")
            self.assertEqual(results[1][0]["message"], "Hi")

    def test_ollama_client_talk_failure(self):
        client = ollama_client.OllamaClient("http://127.0.0.1:11434", "test-model", 10)
        with patch.object(client, "_request", side_effect=TimeoutError("Timed out")):
            result = client.talk({"id": 0}, {"id": 1})
            self.assertIsNone(result)

    def test_gemini_client_talk_success(self):
        client = gemini_client.GeminiClient("https://endpoint", "model", 10, api_key="dummy")
        mock_response = {
            "dialogue": [
                {"speaker": 10, "message": "Greetings."},
                {"speaker": 20, "message": "Peace be with you."}
            ]
        }
        with patch.object(client, "_request", return_value=mock_response):
            result = client.talk({"id": 10}, {"id": 20})
            self.assertEqual(len(result), 2)
            self.assertEqual(result[0]["speaker"], 10)

    def test_gemini_client_talk_failure(self):
        client = gemini_client.GeminiClient("https://endpoint", "model", 10, api_key="dummy")
        with patch.object(client, "_request", side_effect=ValueError("Invalid API key")):
            result = client.talk({"id": 10}, {"id": 20})
            self.assertIsNone(result)


class TestAgentTalkIntegration(unittest.TestCase):
    def _create_test_sugarscape(self, simulationMode="llm", agentTalk=True):
        with open("config/llm.json") as f:
            data = json.load(f)
        config = data.get("sugarscapeOptions", data).copy()
        config["startingAgents"] = 0
        config["headlessMode"] = True
        config["logfile"] = None
        config["agentLogfile"] = None
        config["simulationMode"] = simulationMode
        config["agentTalk"] = agentTalk
        config["agentTalkMaxNeighbors"] = 1
        verified_config = sugarscape.verifyConfiguration(config)
        s = sugarscape.Sugarscape(verified_config)
        s.llmClient = MagicMock()
        return s

    def _create_agent_at(self, s, agent_id, x, y):
        if len(s.agentEndowments) == 0:
            s.agentEndowments = s.randomizeAgentEndowments(10)
        agent_config = s.agentEndowments[agent_id % len(s.agentEndowments)].copy()
        target_cell = s.environment.grid[x][y]
        a = agent.Agent(agent_id, 0, target_cell, agent_config)
        target_cell.agent = a
        s.agents.append(a)
        return a

    def test_agent_get_talk_state(self):
        s = self._create_test_sugarscape()
        a = self._create_agent_at(s, 1, 2, 3)
        state = a.getTalkState()
        self.assertEqual(state["id"], 1)
        self.assertEqual(state["sugar"], a.sugar)
        self.assertEqual(state["x"], 2)
        self.assertEqual(state["y"], 3)

    def test_agent_talk_success(self):
        s = self._create_test_sugarscape()
        a1 = self._create_agent_at(s, 1, 2, 2)
        a2 = self._create_agent_at(s, 2, 2, 3) # adjacent neighbor

        mock_dialogue = [
            {"speaker": 1, "message": "Hello neighbor!"},
            {"speaker": 2, "message": "Good day to you."}
        ]
        s.llmClient.talk.return_value = mock_dialogue

        conv = a1.talk(a2)
        self.assertIsNotNone(conv)
        self.assertEqual(conv["speaker"], 1)
        self.assertEqual(conv["listener"], 2)
        self.assertEqual(conv["dialogue"], mock_dialogue)

        # Check recorded on both agents and sugarscape
        self.assertEqual(len(a1.conversations), 1)
        self.assertEqual(len(a2.conversations), 1)
        self.assertEqual(len(s.conversations), 1)
        self.assertEqual(a1.lastTalkPartners, 1)
        self.assertEqual(a2.lastTalkPartners, 1)

    def test_sugarscape_talk_helper(self):
        s = self._create_test_sugarscape()
        a1 = self._create_agent_at(s, 1, 2, 2)
        a2 = self._create_agent_at(s, 2, 2, 3)

        s.llmClient.talk.return_value = [{"speaker": 1, "message": "Hi"}]
        conv = s.talk(a1, a2)
        self.assertIsNotNone(conv)
        self.assertEqual(len(s.conversations), 1)

    def test_do_talk_timestep_execution(self):
        s = self._create_test_sugarscape(simulationMode="llm", agentTalk=True)
        a1 = self._create_agent_at(s, 1, 2, 2)
        a2 = self._create_agent_at(s, 2, 2, 3)
        a1.updateNeighbors()
        a2.updateNeighbors()

        mock_dialogue = [
            {"speaker": 1, "message": "Nice day!"},
            {"speaker": 2, "message": "Yes, plenty of sugar."}
        ]
        s.llmClient.talk.return_value = mock_dialogue

        # Run a1.doTalk()
        a1.doTalk()
        self.assertEqual(len(s.conversations), 1)
        self.assertIn((1, 2), s.timestepTalkPairs)

        # Running a2.doTalk() in the same timestep should NOT duplicate conversation with a1
        a2.doTalk()
        self.assertEqual(len(s.conversations), 1)

    def test_do_talk_disabled_in_normal_mode(self):
        s = self._create_test_sugarscape(simulationMode="normal", agentTalk=False)
        a1 = self._create_agent_at(s, 1, 2, 2)
        a2 = self._create_agent_at(s, 2, 2, 3)
        a1.updateNeighbors()
        a2.updateNeighbors()

        a1.doTalk()
        self.assertEqual(len(s.conversations), 0)

    def test_do_talk_disabled_by_config(self):
        s = self._create_test_sugarscape(simulationMode="llm", agentTalk=False)
        a1 = self._create_agent_at(s, 1, 2, 2)
        a2 = self._create_agent_at(s, 2, 2, 3)
        a1.updateNeighbors()
        a2.updateNeighbors()

    def test_debug_mode_talk_prints(self):
        import io
        s = self._create_test_sugarscape(simulationMode="llm", agentTalk=True)
        s.debug = ["talk"]
        a1 = self._create_agent_at(s, 1, 2, 2)
        a2 = self._create_agent_at(s, 2, 2, 3)
        a1.debug = ["talk"]
        a2.debug = ["talk"]

        mock_dialogue = [
            {"speaker": 1, "message": "Can I trade with you?"},
            {"speaker": 2, "message": "Sure, what do you need?"}
        ]
        s.llmClient.talk.return_value = mock_dialogue

        captured = io.StringIO()
        with patch('sys.stdout', captured):
            a1.talk(a2)
        output = captured.getvalue()
        self.assertIn("Agent 1 talked with Agent 2", output)
        self.assertIn("Can I trade with you?", output)
        self.assertIn("Sure, what do you need?", output)


if __name__ == "__main__":
    unittest.main()
