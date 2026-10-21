import json
import os
import urllib.error
import urllib.request

from .jev_client import JevClient


class TevClient(JevClient):
    def __init__(self, endpoint="http://127.0.0.1:11434/v1/systemone", model="tev1:4b", timeout=60, options=None):
        super().__init__(endpoint, model, timeout, options, api_key="ollama")
        try:
            self.numParallel = max(1, int(os.environ.get("TEV_NUM_PARALLEL", "1")))
        except ValueError:
            self.numParallel = 1

    def _report_error(self, error):
        print(f"Tev1 connection/request failed ({self.endpoint}, model={self.model}): {error}")

    def _request(self, state, questions):
        request_data = {
            "state": state,
            "model": self.model,
            "questions": questions,
        }
        if "keep_alive" in self.options:
            request_data["keep_alive"] = self.options["keep_alive"]
        headers = {"Content-Type": "application/json"}

        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(request_data).encode("utf-8"),
            headers=headers,
            method="POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            error_body = error.read().decode("utf-8", "replace").strip()
            if error_body:
                raise urllib.error.HTTPError(
                    error.url, error.code, f"{error.reason}: {error_body}", error.headers, error.fp
                ) from error
            raise