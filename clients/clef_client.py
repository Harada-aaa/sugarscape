import json
import os
import urllib.error
import urllib.request

from .jev_client import JevClient


DEFAULT_ENDPOINT = "https://api.cloudflare.com/client/v4/accounts"
DEFAULT_MODEL = "clef"


class ClefClient(JevClient):
    """Client for Cloudflare's Clef System One decision models.

    Clef speaks the same request/response format as Jev. Cloudflare's Workers AI
    REST API additionally wraps the System One response in a ``result`` object,
    which this client unwraps before parsing the answer.

    The endpoint may be one of:
      * the Cloudflare accounts API base (the default), in which case the account
        id and model path are appended automatically;
      * a complete Workers AI URL such as
        ``https://api.cloudflare.com/client/v4/accounts/<id>/ai/run/@cf/cloudflare/clef``;
      * a local System One endpoint such as ``http://127.0.0.1:11434/v1/systemone``
        (for example when running Clef through Ollama).
    """

    def __init__(self, endpoint=DEFAULT_ENDPOINT, model=DEFAULT_MODEL, timeout=30,
                 options=None, api_key=None, account_id=None):
        self.account_id = account_id or os.environ.get("CLOUDFLARE_ACCOUNT_ID")
        clef_api_key = (
            api_key
            or os.environ.get("CLOUDFLARE_API_TOKEN")
            or os.environ.get("CLOUDFLARE_AUTH_TOKEN")
        )
        super().__init__(endpoint, model, timeout, options, clef_api_key)
        self.api_key = clef_api_key
        try:
            self.numParallel = max(1, int(os.environ.get("CLEF_NUM_PARALLEL", "1")))
        except ValueError:
            self.numParallel = 1

    def _report_error(self, error):
        print(f"Clef connection/request failed ({self.endpoint}, model={self.model}): {error}")

    def _request_url(self):
        endpoint = self.endpoint

        if "{account_id}" in endpoint:
            if not self.account_id:
                raise ValueError("CLOUDFLARE_ACCOUNT_ID is not set")
            endpoint = endpoint.replace("{account_id}", self.account_id)
        if "{model}" in endpoint:
            endpoint = endpoint.replace("{model}", self.model)

        if "/ai/run/" in endpoint:
            return endpoint

        # Local System One servers (for example Ollama's /v1/systemone) are used
        # verbatim and do not require a Cloudflare account id or API token.
        if endpoint.endswith("/v1/systemone") or endpoint.endswith("/systemone"):
            return endpoint

        if not self.account_id:
            raise ValueError("CLOUDFLARE_ACCOUNT_ID is not set")

        base = endpoint.rstrip("/")
        if base.endswith("/ai/run"):
            return f"{base}/@cf/cloudflare/{self.model}"
        if base.endswith("/accounts"):
            return f"{base}/{self.account_id}/ai/run/@cf/cloudflare/{self.model}"
        return f"{base}/{self.account_id}/ai/run/@cf/cloudflare/{self.model}"

    def _request(self, state, questions):
        url = self._request_url()

        request_data = {
            "state": state,
            "model": self.model,
            "questions": questions,
        }

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        request = urllib.request.Request(
            url,
            data=json.dumps(request_data).encode("utf-8"),
            headers=headers,
            method="POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                response_body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            error_body = error.read().decode("utf-8", "replace").strip()
            if error_body:
                raise urllib.error.HTTPError(
                    error.url, error.code, f"{error.reason}: {error_body}", error.headers, error.fp
                ) from error
            raise

        # Workers AI wraps responses as {"result": {...}, "success": true, ...}.
        if isinstance(response_body, dict) and "answers" not in response_body:
            result = response_body.get("result")
            if isinstance(result, dict):
                return result

        return response_body
