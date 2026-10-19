import json
import os
import urllib.error
import urllib.request

from jev_client import JevClient


class KevClient(JevClient):
    def __init__(self, endpoint="http://127.0.0.1:8009/v1/systemone", model="kev-latest", timeout=10, options=None, api_key=None):
        self.kev_api_key = api_key or os.environ.get("KEV_API_KEY")
        super().__init__(endpoint, model, timeout, options, self.kev_api_key)
        self.api_key = self.kev_api_key
        try:
            self.numParallel = max(1, int(os.environ.get("KEV_NUM_PARALLEL", "1")))
        except ValueError:
            self.numParallel = 1

    def _report_error(self, error):
        print(f"Kev connection/request failed ({self.endpoint}, model={self.model}): {error}")

    def _request(self, state, questions):
        request_data = {
            "state": state,
            "model": self.model,
            "questions": questions,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

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