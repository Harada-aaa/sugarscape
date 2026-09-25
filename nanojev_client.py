import json
import os
from concurrent.futures import ThreadPoolExecutor
import urllib.error
import urllib.request


class NanoJevClient:
    def __init__(self, endpoint="http://127.0.0.1:8765", model="NanoJev", timeout=10, options=None):
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.options = dict(options or {})
        try:
            self.numParallel = max(1, int(os.environ.get("NANOJEV_NUM_PARALLEL", "1")))
        except ValueError:
            self.numParallel = 1

    def _report_error(self, error):
        print(f"NanoJev connection/request failed ({self.endpoint}, model={self.model}): {error}")

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
            "id": agent_id,
            "state": state_desc,
            "questions": {
                "move": {
                    "type": "choice",
                    "instructions": "Select the best candidate cell to move to for maximum resource collection and agent survival.",
                    "criteria": criteria
                }
            }
        }

    def _extract_prediction(self, response, agent_id, question_name="move"):
        if not response:
            return None

        # 1. Look for results array: [{"id": "0", "predictions": {"move": {"prediction": "0"}}}, ...]
        results = response.get("results") if isinstance(response, dict) else (response if isinstance(response, list) else None)
        if isinstance(results, list):
            for item in results:
                if isinstance(item, dict):
                    item_id = str(item.get("id", ""))
                    if item_id == "" or item_id == agent_id:
                        # Check item["predictions"][question_name]
                        preds = item.get("predictions")
                        if isinstance(preds, dict):
                            q_pred = preds.get(question_name)
                            if isinstance(q_pred, dict) and "prediction" in q_pred:
                                return int(q_pred["prediction"])
                            elif q_pred is not None and not isinstance(q_pred, dict):
                                return int(q_pred)

                        # Check item[question_name]
                        if question_name in item:
                            q_pred = item[question_name]
                            if isinstance(q_pred, dict) and "prediction" in q_pred:
                                return int(q_pred["prediction"])
                            elif q_pred is not None:
                                return int(q_pred)

                        # Check item["prediction"] or item["choice"]
                        if "prediction" in item:
                            return int(item["prediction"])
                        if "choice" in item:
                            return int(item["choice"])

        # 2. Look for top-level predictions: {"predictions": {"0": 1}} or {"decisions": {"0": 1}}
        if isinstance(response, dict):
            for key in ["predictions", "decisions", "choices", "results"]:
                container = response.get(key)
                if isinstance(container, dict) and agent_id in container:
                    val = container[agent_id]
                    if isinstance(val, dict) and "prediction" in val:
                        return int(val["prediction"])
                    return int(val)

            if "candidate" in response:
                return int(response["candidate"])
            if "prediction" in response:
                return int(response["prediction"])

        return None

    def _request(self, payload):
        request_data = {
            "states": payload.get("states", []),
            "model": self.model,
            **self.options
        }
        for k, v in payload.items():
            if k != "states":
                request_data[k] = v

        request_body = json.dumps(request_data).encode("utf-8")

        # Determine URL
        if self.endpoint.endswith("/api/evaluate") or self.endpoint.endswith("/evaluate"):
            url = self.endpoint
        else:
            url = f"{self.endpoint}/api/evaluate"

        request = urllib.request.Request(
            url,
            data=request_body,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                response_body = json.loads(response.read().decode("utf-8"))
            return response_body
        except urllib.error.HTTPError as error:
            if error.code == 404 and not self.endpoint.endswith("/evaluate"):
                # Fallback to /evaluate
                fallback_url = f"{self.endpoint}/evaluate"
                fb_req = urllib.request.Request(
                    fallback_url,
                    data=request_body,
                    headers={"Content-Type": "application/json"},
                    method="POST"
                )
                with urllib.request.urlopen(fb_req, timeout=self.timeout) as fb_resp:
                    return json.loads(fb_resp.read().decode("utf-8"))
            raise

    def choose_cell(self, agent_state, candidates):
        if not candidates:
            return None
        state = self._format_agent_state(agent_state, candidates)
        try:
            response = self._request({"states": [state]})
            decision = self._extract_prediction(response, str(agent_state.get("id", "0")), "move")
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
        states = []
        agent_candidates_len = {}
        for record in faction_state:
            agent = record["agent"]
            cands = record["candidates"]
            agent_id = str(agent["id"])
            agent_candidates_len[agent_id] = len(cands)
            states.append(self._format_agent_state(agent, cands))

        try:
            response = self._request({"states": states})
            decisions = {}
            for record in faction_state:
                agent_id = str(record["agent"]["id"])
                cand_len = agent_candidates_len.get(agent_id, 0)
                pred = self._extract_prediction(response, agent_id, "move")
                if pred is not None and 0 <= pred < cand_len:
                    decisions[agent_id] = pred
            return decisions
        except (KeyError, TypeError, ValueError, json.JSONDecodeError,
                urllib.error.URLError, TimeoutError) as error:
            self._report_error(error)
            return {}

    def choose_cells_parallel(self, faction_states):
        with ThreadPoolExecutor(max_workers=self.numParallel) as executor:
            decisions = executor.map(self.choose_cells, faction_states)
        return list(decisions)

    def talk(self, speaker_state, listener_state):
        # NanoJev is a structured decision classifier/choice model
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

        state_desc = (
            f"Speaker {speaker_id} (sugar={speaker_state.get('sugar', 0)}, spice={speaker_state.get('spice', 0)}) "
            f"meets Listener {listener_id} (sugar={listener_state.get('sugar', 0)}, spice={listener_state.get('spice', 0)})."
        )

        criteria = {
            str(i): f"Dialogue option {i}: {d[0]['message']} / {d[1]['message']}"
            for i, d in enumerate(candidate_dialogues)
        }

        req_state = {
            "id": f"talk_{speaker_id}_{listener_id}",
            "state": state_desc,
            "questions": {
                "dialogue": {
                    "type": "choice",
                    "instructions": "Select the most appropriate conversational dialogue between these two agents.",
                    "criteria": criteria
                }
            }
        }

        try:
            response = self._request({"states": [req_state]})
            pred = self._extract_prediction(response, f"talk_{speaker_id}_{listener_id}", "dialogue")
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

