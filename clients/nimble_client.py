"""Client for Ollama decision models such as nimble-v3.

Decision models do not support ``/api/generate`` or ``/api/chat``.  They are
served by Ollama's System One endpoint (``/v1/systemone``) and answer one or
more named questions about a shared state.
"""

import json
import os
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from .jev_client import JevClient


def formatSystemOneEndpoint(endpoint):
    endpoint = endpoint.rstrip("/")
    if endpoint.endswith("/v1/systemone"):
        return endpoint
    if endpoint.endswith("/v1"):
        return f"{endpoint}/systemone"
    return f"{endpoint}/v1/systemone"


class NimbleClient(JevClient):
    def __init__(self, endpoint="http://127.0.0.1:11434", model="nimble-v3:9b",
                 timeout=120, options=None):
        super().__init__(formatSystemOneEndpoint(endpoint), model, timeout, options, api_key="ollama")
        try:
            self.numParallel = max(1, int(os.environ.get("NIMBLE_NUM_PARALLEL", "1")))
        except ValueError:
            self.numParallel = 1

    def _report_error(self, error):
        print(f"Nimble decision request failed ({self.endpoint}, model={self.model}): {error}")

    def _request(self, state, questions):
        requestData = {
            "state": state,
            "model": self.model,
            "questions": questions,
        }
        if "keep_alive" in self.options:
            requestData["keep_alive"] = self.options["keep_alive"]

        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(requestData).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            errorBody = error.read().decode("utf-8", "replace").strip()
            if errorBody:
                raise urllib.error.HTTPError(
                    error.url, error.code, f"{error.reason}: {errorBody}", error.headers, error.fp
                ) from error
            raise

    def _formatCandidates(self, candidates):
        criteria = {}
        for index, candidate in enumerate(candidates):
            criteria[str(index)] = (
                f"Candidate {candidate.get('candidate', index)}: "
                f"move to position (x={candidate.get('x', 0)}, y={candidate.get('y', 0)}) "
                f"with wealth {candidate.get('wealth', 0.0)} "
                f"at distance {candidate.get('distance', candidate.get('range', 1))}"
            )
        return criteria

    def choose_cells(self, factionState):
        if not factionState:
            return {}

        decisions = {}
        stateRecords = []
        questions = {}
        questionAgents = {}
        for record in factionState:
            agent = record.get("agent") or {}
            candidates = record.get("candidates") or []
            agentID = str(agent.get("id"))
            if len(candidates) == 1:
                decisions[agentID] = 0
                continue
            stateIndex = len(stateRecords)
            stateRecords.append({"agent": agent, "candidates": candidates})
            questionName = f"agent_{stateIndex + 1}"
            questions[questionName] = {
                "type": "choice",
                "instructions": (
                    f"For agent {agentID} in state.agents[{stateIndex}], select the best "
                    "candidate cell to move to for maximum resource collection and agent survival."
                ),
                "criteria": self._formatCandidates(candidates),
            }
            questionAgents[questionName] = (agentID, len(candidates))

        if not questions:
            return decisions

        try:
            response = self._request({"agents": stateRecords}, questions)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError,
                urllib.error.URLError, TimeoutError) as error:
            self._report_error(error)
            return decisions

        answers = response.get("answers") if isinstance(response, dict) else None
        if not isinstance(answers, dict):
            return decisions
        for questionName, (agentID, candidateCount) in questionAgents.items():
            answer = answers.get(questionName)
            choice = answer.get("choice") if isinstance(answer, dict) else None
            try:
                candidate = int(choice)
            except (TypeError, ValueError):
                continue
            if 0 <= candidate < candidateCount:
                decisions[agentID] = candidate
        return decisions

    def choose_cells_parallel(self, factionStates):
        with ThreadPoolExecutor(max_workers=self.numParallel) as executor:
            decisions = executor.map(self.choose_cells, factionStates)
        return list(decisions)
