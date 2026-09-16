import json
import urllib.error
import urllib.request


class VLLMClient:
    def __init__(self, endpoint, model, timeout, options=None):
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.options = dict(options or {})
        if "num_predict" in self.options and "max_tokens" not in self.options:
            self.options["max_tokens"] = self.options.pop("num_predict")

    def _report_error(self, error):
        print(f"vLLM connection/request failed ({self.endpoint}, model={self.model}): {error}")

    def _request(self, prompt):
        messages = [
            {"role": "system", "content": "Return only valid JSON."},
            {"role": "user", "content": json.dumps(prompt)}
        ]
        requestBody = json.dumps({
            "model": self.model,
            "messages": messages,
            "stream": False,
            "response_format": {"type": "json_object"},
            **self.options
        }).encode("utf-8")
        request = urllib.request.Request(
            f"{self.endpoint}/chat/completions",
            data=requestBody,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            responseBody = json.loads(response.read().decode("utf-8"))
        content = responseBody["choices"][0]["message"]["content"]
        return json.loads(content)

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