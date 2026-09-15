import json
import urllib.error
import urllib.request


class OllamaClient:
    def __init__(self, endpoint, model, timeout):
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.timeout = timeout

    def choose_cell(self, agent_state, candidates):
        prompt = {
            "agent": agent_state,
            "candidates": candidates,
            "instruction": "Choose one candidate cell. Return only JSON in the form {\"candidate\": <integer>}."
        }
        requestBody = json.dumps({
            "model": self.model,
            "prompt": json.dumps(prompt),
            "stream": False,
            "format": "json"
        }).encode("utf-8")
        request = urllib.request.Request(
            f"{self.endpoint}/api/generate",
            data=requestBody,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                responseBody = json.loads(response.read().decode("utf-8"))
            answer = json.loads(responseBody.get("response", "{}"))
            candidate = int(answer["candidate"])
            if 0 <= candidate < len(candidates):
                return candidate
        except (KeyError, TypeError, ValueError, json.JSONDecodeError,
                urllib.error.URLError, TimeoutError):
            return None
        return None