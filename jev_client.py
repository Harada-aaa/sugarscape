import json
import os
from concurrent.futures import ThreadPoolExecutor
import urllib.error
import urllib.request


class JevClient:
    def __init__(self, endpoint="https://api.typesafe.ai/v1/systemone", model="jev-latest", timeout=10, options=None, api_key=None):
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.options = dict(options or {})
        self.api_key = api_key or os.environ.get("TYPESAFE_API_KEY")
        try:
            self.numParallel = max(1, int(os.environ.get("JEV_NUM_PARALLEL", "1")))
        except ValueError:
            self.numParallel = 1

    def _report_error(self, error):
        print(f"Jev connection/request failed ({self.endpoint}, model={self.model}): {error}")

    def _format_agent_state(self, agent_state, candidates):
        agent_id = str(agent_state.get("id", "0"))
        state_desc = (
            f"Agent {agent_id}: age {agent_state.get('age', 0)}, "
            f"sugar {agent_state.get('sugar', 0.0)}, spice {agent_state.get('spice', 0.0)}, "
            f"sugarMetabolism {agent_state.get('sugarMetabolism', 1)}, "
            f"spiceMetabolism {agent_state.get('spiceMetabolism', 0)}, "
            f"happiness {agent_state.get('happiness', 0.0)}."
        )
        criteria = {}
        for idx, cand in enumerate(candidates):
            key = str(idx)
            cand_idx = cand.get("candidate", idx)
            x = cand.get("x", 0)
            y = cand.get("y", 0)
            wealth = cand.get("wealth", 0.0)
            dist = cand.get("distance", cand.get("range", 1))
            criteria[key] = f"Candidate {cand_idx}: move to position (x={x}, y={y}) with wealth {wealth} at distance {dist}"

        return {
            "agent": state_desc,
            "candidates": candidates,
        }, {
            "type": "choice",
            "instructions": "Select the best candidate cell to move to for maximum resource collection and agent survival.",
            "criteria": criteria
        }

    def _extract_prediction(self, response, question_name="move"):
        if not response:
            return None
        answers = response.get("answers") if isinstance(response, dict) else None
        answer = answers.get(question_name) if isinstance(answers, dict) else None
        choice = answer.get("choice") if isinstance(answer, dict) else None
        try:
            return int(choice) if choice is not None else None
        except (TypeError, ValueError):
            return None

    def _request(self, state, questions):
        if not self.api_key:
            raise ValueError("TYPESAFE_API_KEY is not set")

        request_data = {
            "state": state,
            "model": self.model,
            "questions": questions,
        }

        request_body = json.dumps(request_data).encode("utf-8")

        request = urllib.request.Request(
            self.endpoint,
            data=request_body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                response_body = json.loads(response.read().decode("utf-8"))
            return response_body
        except urllib.error.HTTPError as error:
            error_body = error.read().decode("utf-8", "replace").strip()
            if error_body:
                raise urllib.error.HTTPError(
                    error.url, error.code, f"{error.reason}: {error_body}", error.headers, error.fp
                ) from error
            raise

    def choose_cell(self, agent_state, candidates):
        if not candidates:
            return None
        if len(candidates) == 1:
            return 0
        state, question = self._format_agent_state(agent_state, candidates)
        try:
            response = self._request(state, {"move": question})
            decision = self._extract_prediction(response, "move")
            if decision is not None and 0 <= decision < len(candidates):
                return decision
        except (KeyError, TypeError, ValueError, json.JSONDecodeError,
                urllib.error.URLError, TimeoutError) as error:
            self._report_error(error)
            return None
        return None

    def choose_cells(self, faction_state):
        if not faction_state:
            return {}
        decisions = {}
        for record in faction_state:
            agent = record["agent"]
            cands = record["candidates"]
            agent_id = str(agent["id"])
            if len(cands) == 1:
                decisions[agent_id] = 0
                continue
            decision = self.choose_cell(agent, cands)
            if decision is not None:
                decisions[agent_id] = decision
        return decisions

    def choose_cells_parallel(self, faction_states):
        with ThreadPoolExecutor(max_workers=self.numParallel) as executor:
            decisions = executor.map(self.choose_cells, faction_states)
        return list(decisions)

    def talk(self, speaker_state, listener_state):
        # Jev is a structured decision classifier/choice model
        # Provide dialogue choice selection or fallback structured message
        speaker_id = speaker_state.get("id", 0)
        listener_id = listener_state.get("id", 1)

        candidate_dialogues = [
            [
                {"speaker": speaker_id, "message": "Greetings, neighbor."},
                {"speaker": listener_id, "message": "Greetings. Let us share the landscape peacefully."}
            ],
            [
                {"speaker": speaker_id, "message": "Are resources plentiful around your cell?"},
                {"speaker": listener_id, "message": "There is enough sugar and spice for both of us."}
            ],
            [
                {"speaker": speaker_id, "message": "I am gathering resources nearby."},
                {"speaker": listener_id, "message": "Understood, I will gather from the neighboring patch."}
            ]
        ]

        state = {
            "speaker": speaker_state,
            "listener": listener_state,
        }

        criteria = {
            str(i): f"Dialogue option {i}: {d[0]['message']} / {d[1]['message']}"
            for i, d in enumerate(candidate_dialogues)
        }

        question = {
            "type": "choice",
            "instructions": "Select the most appropriate conversational dialogue between these two agents.",
            "criteria": criteria
        }

        try:
            response = self._request(state, {"dialogue": question})
            pred = self._extract_prediction(response, "dialogue")
            if pred is not None and 0 <= pred < len(candidate_dialogues):
                return candidate_dialogues[pred]
        except (KeyError, TypeError, ValueError, json.JSONDecodeError,
                urllib.error.URLError, TimeoutError) as error:
            self._report_error(error)
            return None
        return candidate_dialogues[0]

    def talk_parallel(self, pairs):
        with ThreadPoolExecutor(max_workers=self.numParallel) as executor:
            dialogues = executor.map(lambda pair: self.talk(pair[0], pair[1]), pairs)
        return list(dialogues)

