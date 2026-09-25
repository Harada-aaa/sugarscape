import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from sugarscape import Sugarscape, detectLogTimestep, prepareLogFileForResume, verifyConfiguration


class TestDetectLogTimestep(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_detect_json_complete(self):
        path = os.path.join(self.test_dir, "test.json")
        data = [{"timestep": 0, "population": 10}, {"timestep": 1, "population": 10}, {"timestep": 2, "population": 9}]
        with open(path, "w") as f:
            json.dump(data, f)
        ts, seed = detectLogTimestep(path, "json")
        self.assertEqual(ts, 2)
        self.assertIsNone(seed)

    def test_detect_json_interrupted(self):
        path = os.path.join(self.test_dir, "test_interrupted.json")
        with open(path, "w") as f:
            f.write('[\n\t{"timestep": 0, "population": 10},\n\t{"timestep": 1, "population": 10},\n')
        ts, seed = detectLogTimestep(path, "json")
        self.assertEqual(ts, 1)
        self.assertIsNone(seed)

    def test_detect_csv_complete(self):
        path = os.path.join(self.test_dir, "test.csv")
        with open(path, "w") as f:
            f.write("deaths,timestep,population\n0,0,10\n0,1,10\n0,2,9\n")
        ts, seed = detectLogTimestep(path, "csv")
        self.assertEqual(ts, 2)
        self.assertIsNone(seed)

    def test_detect_nonexistent_or_empty(self):
        self.assertEqual(detectLogTimestep(os.path.join(self.test_dir, "nonexistent.json"), "json"), (-1, None))
        empty_path = os.path.join(self.test_dir, "empty.json")
        with open(empty_path, "w") as f:
            pass
        self.assertEqual(detectLogTimestep(empty_path, "json"), (-1, None))


class TestPrepareLogFileForResume(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_prepare_json_resume(self):
        path = os.path.join(self.test_dir, "test.json")
        data = [
            {"timestep": 0, "population": 10},
            {"timestep": 1, "population": 10},
            {"timestep": 2, "population": 9},
            {"timestep": 3, "population": 8},
        ]
        with open(path, "w") as f:
            json.dump(data, f)

        res = prepareLogFileForResume(path, 2, "json")
        self.assertTrue(res)

        # File should end with comma ready for append
        with open(path, "r") as f:
            content = f.read().strip()
            self.assertTrue(content.startswith("[\n"))
            self.assertTrue(content.endswith(","))

    def test_prepare_csv_resume(self):
        path = os.path.join(self.test_dir, "test.csv")
        with open(path, "w") as f:
            f.write("timestep,population\n0,10\n1,10\n2,9\n3,8\n")

        res = prepareLogFileForResume(path, 2, "csv")
        self.assertTrue(res)

        with open(path, "r") as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 4)  # header + 3 rows (0, 1, 2)
            self.assertEqual(lines[-1].strip(), "2,9")


class TestSimulationResume(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def _get_base_config(self, logfile, agentLogfile, fmt="json", mode="normal"):
        default_config = {
            "agentAggressionFactor": [0, 0],
            "agentDistributionMode": "uniform",
            "agentBaseInterestRate": [0.0, 0.0],
            "agentDecisionModels": ["none"],
            "agentDecisionModel": None,
            "agentDecisionModelAgeismFactor": [-1, -1],
            "agentDecisionModelFactor": [0, 0],
            "agentDecisionModelLookaheadDiscount": [0, 0],
            "agentDecisionModelLookaheadFactor": [0],
            "agentDecisionModelRacismFactor": [-1, -1],
            "agentDecisionModelSexismFactor": [-1, -1],
            "agentDecisionModelTribalFactor": [-1, -1],
            "agentDepressionPercentage": 0,
            "agentDiseaseProtectionChance": [0.0, 0.0],
            "agentDynamicDecisionModelFactor": [0.0, 0.0],
            "agentDynamicSelfishnessFactor": [0.0, 0.0],
            "agentDynamicSocialPressureFactor": [0, 1.0],
            "agentFemaleInfertilityAge": [0, 0],
            "agentFemaleFertilityAge": [0, 0],
            "agentFertilityFactor": [0, 0],
            "agentImmuneSystemLength": 0,
            "agentInheritancePolicy": "none",
            "agentLeader": False,
            "agentLendingFactor": [0, 0],
            "agentLoanDuration": [0, 0],
            "agentLogfile": agentLogfile,
            "agentLogfileFormat": fmt,
            "agentLookaheadFactor": [0, 0],
            "agentMaleInfertilityAge": [0, 0],
            "agentMaleFertilityAge": [0, 0],
            "agentMaleToFemaleRatio": 1.0,
            "agentMaxAge": [-1, -1],
            "agentMaxFriends": [0, 0],
            "agentMovement": [1, 1],
            "agentMovementMode": "cardinal",
            "agentRacialTagStringLength": 0,
            "agentReplacements": 0,
            "agentSelfishnessFactor": [-1, -1],
            "agentSpiceMetabolism": [0, 0],
            "agentStartingSpice": [10, 10],
            "agentStartingSugar": [10, 10],
            "agentSugarMetabolism": [1, 1],
            "agentTagging": False,
            "agentTagPreferences": False,
            "agentTagStringLength": 0,
            "agentTemperanceFactor": [0, 0],
            "agentTradeFactor": [0, 0],
            "agentUniversalSpice": [0, 0],
            "agentUniversalSugar": [0, 0],
            "agentVision": [1, 1],
            "agentVisionMode": "cardinal",
            "debugMode": ["none"],
            "diseaseAggressionPenalty": [0, 0],
            "diseaseFertilityPenalty": [0, 0],
            "diseaseFriendlinessPenalty": [0, 0],
            "diseaseHappinessPenalty": [0, 0],
            "diseaseIncubationPeriod": [0, 0],
            "diseaseList": [],
            "diseaseMovementPenalty": [0, 0],
            "diseaseSpiceMetabolismPenalty": [0, 0],
            "diseaseSugarMetabolismPenalty": [0, 0],
            "diseaseTagStringLength": [0, 0],
            "diseaseTimeframe": [0, 0],
            "diseaseTransmissionChance": [1.0, 1.0],
            "diseaseVisionPenalty": [0, 0],
            "environmentAgeistAbsoluteRanges": [],
            "environmentAgeistRelativeRange": -1,
            "environmentEquator": -1,
            "environmentFile": None,
            "environmentHeight": 10,
            "environmentInGroupRaces": [],
            "environmentMaxCombatLoot": 0,
            "environmentMaxRaces": 0,
            "environmentMaxSpice": 4,
            "environmentMaxSugar": 4,
            "environmentMaxTribes": 0,
            "environmentPollutionDiffusionDelay": 0,
            "environmentPollutionDiffusionTimeframe": [0, 0],
            "environmentPollutionTimeframe": [0, 0],
            "environmentQuadrantSizeFactor": 1,
            "environmentSeasonalGrowbackDelay": 0,
            "environmentSeasonInterval": 0,
            "environmentSexistGroups": [],
            "environmentSpiceConsumptionPollutionFactor": 0,
            "environmentSpicePeaks": [[7, 7, 4]],
            "environmentSpiceProductionPollutionFactor": 0,
            "environmentSpiceRegrowRate": 1,
            "environmentStartingQuadrants": [1, 2, 3, 4],
            "environmentSugarConsumptionPollutionFactor": 0,
            "environmentSugarPeaks": [[3, 3, 4]],
            "environmentSugarProductionPollutionFactor": 0,
            "environmentSugarRegrowRate": 1,
            "environmentTribePerQuadrant": False,
            "environmentUniversalSpiceIncomeInterval": 0,
            "environmentUniversalSugarIncomeInterval": 0,
            "environmentWidth": 10,
            "environmentWraparound": True,
            "experimentalGroup": None,
            "headlessMode": True,
            "interfaceHeight": 1000,
            "interfaceWidth": 900,
            "keepAlivePostExtinction": False,
            "keepAliveAtEnd": False,
            "logfile": logfile,
            "logfileFormat": fmt,
            "neighborhoodMode": "vonNeumann",
            "llmBackend": "vllm",
            "geminiEndpoint": "https://generativelanguage.googleapis.com/v1beta",
            "geminiModel": "gemini-3.6-flash",
            "geminiApiKey": None,
            "geminiTimeout": 30,
            "geminiOptions": {"temperature": 0.2},
            "ollamaEndpoint": "http://127.0.0.1:11434",
            "ollamaModel": "llama3.1:8b",
            "ollamaOptions": {},
            "ollamaTimeout": 10,
            "vllmEndpoint": "http://127.0.0.1:8000/v1",
            "vllmModel": "meta-llama/Llama-3.1-8B-Instruct",
            "vllmOptions": {},
            "vllmTimeout": 10,
            "profileMode": False,
            "screenshots": False,
            "simulationMode": mode,
            "agentTalk": False,
            "agentTalkMaxNeighbors": 1,
            "resumeFromLog": True,
            "seed": 12345,
            "startingAgents": 5,
            "startingDiseases": 0,
            "startingDiseasesPerAgent": [0, 0],
            "timesteps": 3,
        }
        return verifyConfiguration(default_config)

    @patch("builtins.exit")
    def test_resume_json_simulation(self, mock_exit):
        logfile = os.path.join(self.test_dir, "sim_log.json")
        agent_logfile = os.path.join(self.test_dir, "agent_log.json")

        # Step 1: Run 3 timesteps
        cfg1 = self._get_base_config(logfile, agent_logfile, "json")
        sim1 = Sugarscape(cfg1)
        sim1.runSimulation(3)

        # Verify initial logs
        self.assertTrue(os.path.exists(logfile))
        self.assertTrue(os.path.exists(agent_logfile))
        with open(logfile, "r") as f:
            data1 = json.load(f)
        self.assertEqual(len(data1), 4)  # timestep 0, 1, 2, 3
        self.assertEqual([d["timestep"] for d in data1], [0, 1, 2, 3])

        with open(agent_logfile, "r") as f:
            agent_data1 = json.load(f)
        self.assertTrue(len(agent_data1) > 0)
        self.assertEqual(max(d["timestep"] for d in agent_data1), 3)

        # Step 2: Resume up to 6 timesteps
        cfg2 = self._get_base_config(logfile, agent_logfile, "json")
        sim2 = Sugarscape(cfg2)
        self.assertEqual(sim2.resumeTimestep, 3)
        sim2.runSimulation(6)

        # Verify resumed logs
        with open(logfile, "r") as f:
            data2 = json.load(f)
        self.assertEqual(len(data2), 7)  # timesteps 0 through 6
        self.assertEqual([d["timestep"] for d in data2], [0, 1, 2, 3, 4, 5, 6])

        with open(agent_logfile, "r") as f:
            agent_data2 = json.load(f)
        self.assertEqual(max(d["timestep"] for d in agent_data2), 6)

        # Verify continuous single run matches resumed run exactly (deterministic)
        logfile_control = os.path.join(self.test_dir, "control_log.json")
        agent_logfile_control = os.path.join(self.test_dir, "control_agent_log.json")
        cfg_control = self._get_base_config(logfile_control, agent_logfile_control, "json")
        sim_control = Sugarscape(cfg_control)
        sim_control.runSimulation(6)

        with open(logfile_control, "r") as f:
            data_control = json.load(f)
        self.assertEqual(data2, data_control)

    @patch("builtins.exit")
    def test_resume_csv_simulation(self, mock_exit):
        logfile = os.path.join(self.test_dir, "sim_log.csv")
        agent_logfile = os.path.join(self.test_dir, "agent_log.csv")

        # Step 1: Run 2 timesteps
        cfg1 = self._get_base_config(logfile, agent_logfile, "csv")
        sim1 = Sugarscape(cfg1)
        sim1.runSimulation(2)

        # Step 2: Resume up to 4 timesteps
        cfg2 = self._get_base_config(logfile, agent_logfile, "csv")
        sim2 = Sugarscape(cfg2)
        self.assertEqual(sim2.resumeTimestep, 2)
        sim2.runSimulation(4)

        with open(logfile, "r") as f:
            lines = [line.strip() for line in f if line.strip()]
        # header + 5 timesteps (0, 1, 2, 3, 4)
        self.assertEqual(len(lines), 6)
        headers = lines[0].split(",")
        ts_idx = headers.index("timestep")
        timesteps = [int(line.split(",")[ts_idx]) for line in lines[1:]]
        self.assertEqual(timesteps, [0, 1, 2, 3, 4])

    @patch("builtins.exit")
    def test_resume_switch_to_llm_mode(self, mock_exit):
        logfile = os.path.join(self.test_dir, "llm_resume_log.json")
        agent_logfile = os.path.join(self.test_dir, "llm_resume_agent_log.json")

        # Step 1: Run normal mode for 2 timesteps
        cfg1 = self._get_base_config(logfile, agent_logfile, "json", mode="normal")
        sim1 = Sugarscape(cfg1)
        sim1.runSimulation(2)

        # Step 2: Resume with LLM mode up to 4 timesteps
        cfg2 = self._get_base_config(logfile, agent_logfile, "json", mode="llm")
        sim2 = Sugarscape(cfg2)
        self.assertEqual(sim2.resumeTimestep, 2)

        # Mock LLM client move
        sim2.llmClient.decide_move = MagicMock(return_value=None)
        sim2.runSimulation(4)

        with open(logfile, "r") as f:
            data = json.load(f)
        self.assertEqual(len(data), 5)  # 0, 1, 2, 3, 4
        self.assertEqual([d["timestep"] for d in data], [0, 1, 2, 3, 4])


if __name__ == "__main__":
    unittest.main()

