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
                urllib.error.URLError, TimeoutError):
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
                urllib.error.URLError, TimeoutError):
            return {}