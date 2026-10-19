import json
import os
import urllib.error
import urllib.request


class GeminiClient:
    def __init__(self, endpoint, model, timeout, options=None, api_key=None):
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.options = dict(options or {})
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")

    def _report_error(self, error):
        print(f"Gemini connection/request failed ({self.endpoint}, model={self.model}): {error}")

    def _request(self, prompt):
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not set")
        requestBody = json.dumps({
            "contents": [{
                "role": "user",
                "parts": [{"text": json.dumps(prompt)}]
            }],
            "generationConfig": {
                "responseMimeType": "application/json",
                **self.options
            }
        }).encode("utf-8")
        request = urllib.request.Request(
            f"{self.endpoint}/models/{self.model}:generateContent",
            data=requestBody,
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self.api_key
            },
            method="POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                responseBody = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            details = error.read().decode("utf-8", errors="replace")
            raise urllib.error.HTTPError(
                error.url, error.code, f"{error.reason}: {details}", error.headers, None
            ) from error
        content = responseBody["candidates"][0]["content"]["parts"][0]["text"]
        return json.loads(content)

    def choose_cell(self, agent_state, candidates):
        prompt = {
            "agent": agent_state,
            "candidates": candidates,
            "instruction": "Choose one candidate cell. Return only JSON in the form {\"candidate\": <integer>}.",
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
            "instruction": "Choose one candidate for every agent. Return only JSON in the form {\"decisions\": {\"<agent id>\": <candidate integer>}}.",
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

