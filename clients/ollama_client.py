import json
import os
from concurrent.futures import ThreadPoolExecutor
import urllib.error
import urllib.request


class OllamaClient:
    def __init__(self, endpoint, model, timeout, options=None):
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.options = dict(options or {})
        try:
            self.numParallel = max(1, int(os.environ.get("OLLAMA_NUM_PARALLEL", "1")))
        except ValueError:
            self.numParallel = 1

    def _report_error(self, error):
        print(f"Ollama connection/request failed ({self.endpoint}, model={self.model}): {error}")

    def _request(self, prompt):
        requestBody = json.dumps({
            "model": self.model,
            "prompt": json.dumps(prompt),
            "stream": False,
            "format": "json",
            "options": self.options
        }).encode("utf-8")
        request = urllib.request.Request(
            f"{self.endpoint}/api/generate",
            data=requestBody,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            responseBody = json.loads(response.read().decode("utf-8"))
        return json.loads(responseBody.get("response", "{}"))

    def choose_cell(self, agent_state, candidates):
        prompt = {
            "agent": agent_state,
            "candidates": candidates,
            "instruction": "Choose one candidate cell. Return only JSON in the form {\"candidate\": <integer>}."
        }
        try:
            answer = self._request(prompt)
            candidate = int(answer["candidate"])
            if 0 <= candidate < len(candidates):
                return candidate
        except (KeyError, TypeError, ValueError, json.JSONDecodeError,
                urllib.error.URLError, TimeoutError) as error:
            self._report_error(error)
            return None
        return None

    def choose_cells(self, faction_state):
        prompt = {
            "faction": faction_state,
            "instruction": "Choose one candidate for every agent. Return only JSON in the form {\"decisions\": {\"<agent id>\": <candidate integer>}}."
        }
        try:
            answer = self._request(prompt)
            decisions = answer["decisions"]
            if not isinstance(decisions, dict):
                return {}
            return {str(agentID): int(candidate) for agentID, candidate in decisions.items()}
        except (KeyError, TypeError, ValueError, json.JSONDecodeError,
                urllib.error.URLError, TimeoutError) as error:
            self._report_error(error)
            return {}

    def choose_cells_parallel(self, faction_states):
        with ThreadPoolExecutor(max_workers=self.numParallel) as executor:
            decisions = executor.map(self.choose_cells, faction_states)
        return list(decisions)

    def talk(self, speaker_state, listener_state):
        prompt = {
            "speaker": speaker_state,
            "listener": listener_state,
            "instruction": "Generate a short natural language conversation (1-2 sentences each) between these two nearby agents in Sugarscape. Return only JSON in the form {\"dialogue\": [{\"speaker\": <id>, \"message\": \"<sentence>\"}]}."
        }
        try:
            answer = self._request(prompt)
            if not isinstance(answer, dict):
                return None
            dialogue = answer.get("dialogue") or answer.get("conversation") or answer.get("messages")
            if isinstance(dialogue, list):
                return dialogue
            if "speaker_message" in answer and "listener_message" in answer:
                return [
                    {"speaker": speaker_state.get("id"), "message": str(answer["speaker_message"])},
                    {"speaker": listener_state.get("id"), "message": str(answer["listener_message"])}
                ]
            return None
        except (KeyError, TypeError, ValueError, json.JSONDecodeError,
                urllib.error.URLError, TimeoutError) as error:
            self._report_error(error)
            return None

    def talk_parallel(self, pairs):
        with ThreadPoolExecutor(max_workers=self.numParallel) as executor:
            dialogues = executor.map(lambda pair: self.talk(pair[0], pair[1]), pairs)
        return list(dialogues)

