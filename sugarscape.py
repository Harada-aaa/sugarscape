#! /usr/bin/python

import agent
import cell
import condition
import environment
import ethics
from clients import gemini_client
from clients import kev_client

import copy
import hashlib
import json
import math
import os
import random
import re
import sys
import time

from clients import nanojev_client
from clients import jev_client
from clients import ollama_client
from clients import vllm_client
from sugarscape_agents import SugarscapeAgentMethods
from sugarscape_config import parseConfiguration, parseOptions, printHelp, sortConfigurationTimeframes
from sugarscape_log import detectLogTimestep, prepareLogFileForResume
from sugarscape_simulation import SugarscapeSimulationMethods
from sugarscape_statistics import SugarscapeStatisticsMethods

class Sugarscape(SugarscapeAgentMethods, SugarscapeSimulationMethods, SugarscapeStatisticsMethods):
    def __init__(self, configuration):
        self.startTime = time.perf_counter()
        self.agentConfigHashes = None
        self.diseaseConfigHashes = None
        self.configuration = configuration
        if configuration["llmBackend"] == "ollama":
            self.llmClient = ollama_client.OllamaClient(
                configuration["ollamaEndpoint"], configuration["ollamaModel"], configuration["ollamaTimeout"],
                configuration["ollamaOptions"])
        elif configuration["llmBackend"] == "gemini":
            self.llmClient = gemini_client.GeminiClient(
                configuration["geminiEndpoint"], configuration["geminiModel"], configuration["geminiTimeout"],
                configuration["geminiOptions"], configuration["geminiApiKey"])
        elif configuration["llmBackend"] == "nanojev":
            self.llmClient = nanojev_client.NanoJevClient(
                configuration["nanojevEndpoint"], configuration["nanojevModel"], configuration["nanojevTimeout"],
                configuration["nanojevOptions"])
        elif configuration["llmBackend"] == "jev":
            self.llmClient = jev_client.JevClient(
                configuration["jevEndpoint"], configuration["jevModel"], configuration["jevTimeout"],
                configuration["jevOptions"], configuration["jevApiKey"])
        elif configuration["llmBackend"] == "kev":
            self.llmClient = kev_client.KevClient(
                configuration["kevEndpoint"], configuration["kevModel"], configuration["kevTimeout"],
                configuration["kevOptions"], configuration["kevApiKey"])
        else:
            self.llmClient = vllm_client.VLLMClient(
                configuration["vllmEndpoint"], configuration["vllmModel"], configuration["vllmTimeout"],
                configuration["vllmOptions"])
        self.maxTimestep = configuration["timesteps"]
        self.timestep = 0
        self.nextAgentID = 0
        self.nextDiseaseID = 0
        environmentConfiguration = {"equator": configuration["environmentEquator"],
                                    "globalMaxSpice": configuration["environmentMaxSpice"],
                                    "globalMaxSugar": configuration["environmentMaxSugar"],
                                    "ageistAbsoluteRanges": configuration["environmentAgeistAbsoluteRanges"],
                                    "ageistRelativeRange": configuration["environmentAgeistRelativeRange"],
                                    "inGroupRaces": configuration["environmentInGroupRaces"],
                                    "maxCombatLoot": configuration["environmentMaxCombatLoot"],
                                    "neighborhoodMode": configuration["neighborhoodMode"],
                                    "pollutionDiffusionDelay": configuration["environmentPollutionDiffusionDelay"],
                                    "pollutionDiffusionTimeframe": configuration["environmentPollutionDiffusionTimeframe"],
                                    "pollutionTimeframe": configuration["environmentPollutionTimeframe"],
                                    "seasonalGrowbackDelay": configuration["environmentSeasonalGrowbackDelay"],
                                    "seasonInterval": configuration["environmentSeasonInterval"],
                                    "sexistGroups": configuration["environmentSexistGroups"],
                                    "spiceConsumptionPollutionFactor": configuration["environmentSpiceConsumptionPollutionFactor"],
                                    "spiceProductionPollutionFactor": configuration["environmentSpiceProductionPollutionFactor"],
                                    "spiceRegrowRate": configuration["environmentSpiceRegrowRate"],
                                    "sugarConsumptionPollutionFactor": configuration["environmentSugarConsumptionPollutionFactor"],
                                    "sugarProductionPollutionFactor": configuration["environmentSugarProductionPollutionFactor"],
                                    "sugarRegrowRate": configuration["environmentSugarRegrowRate"],
                                    "sugarscapeSeed": configuration["seed"],
                                    "universalSpiceIncomeInterval": configuration["environmentUniversalSpiceIncomeInterval"],
                                    "universalSugarIncomeInterval": configuration["environmentUniversalSugarIncomeInterval"],
                                    "wraparound": configuration["environmentWraparound"]}
        self.resumeTimestep = 0
        self.replaying = False
        self.logStarted = False
        self.agentLogStarted = False
        logFile = configuration["logfile"]
        agentLogFile = configuration["agentLogfile"]
        logFormat = configuration["logfileFormat"]
        ts_log = -1
        ts_agent = -1
        if configuration.get("resumeFromLog", True) == True:
            ts_log, seed_log = detectLogTimestep(logFile, logFormat) if logFile != None else (-1, None)
            ts_agent, _ = detectLogTimestep(agentLogFile, logFormat) if agentLogFile != None else (-1, None)

            if logFile != None and agentLogFile != None:
                if ts_log >= 0 and ts_agent >= 0:
                    self.resumeTimestep = min(ts_log, ts_agent)
                elif ts_log >= 0 and ts_agent < 0:
                    self.resumeTimestep = ts_log
                elif ts_agent >= 0 and ts_log < 0:
                    self.resumeTimestep = ts_agent
            elif logFile != None:
                if ts_log >= 0:
                    self.resumeTimestep = ts_log
            elif agentLogFile != None:
                if ts_agent >= 0:
                    self.resumeTimestep = ts_agent

            if seed_log != None:
                if configuration["seed"] == -1 or configuration["seed"] == seed_log:
                    configuration["seed"] = seed_log

            if ts_log >= 0:
                self.logStarted = True
            if ts_agent >= 1:
                self.agentLogStarted = True

            if self.resumeTimestep > 0 or ts_log >= 0:
                if logFile != None and ts_log >= 0:
                    prepareLogFileForResume(logFile, self.resumeTimestep, logFormat)
                if agentLogFile != None and ts_agent >= 0:
                    prepareLogFileForResume(agentLogFile, self.resumeTimestep, logFormat)

        self.seed = configuration["seed"]
        if self.seed != -1 and self.seed is not None:
            random.seed(self.seed)
        self.environment = environment.Environment(configuration["environmentHeight"], configuration["environmentWidth"], self, environmentConfiguration)
        self.environmentHeight = configuration["environmentHeight"]
        self.environmentWidth = configuration["environmentWidth"]
        self.configureEnvironment(configuration["environmentMaxSugar"], configuration["environmentMaxSpice"], configuration["environmentSugarPeaks"], configuration["environmentSpicePeaks"], configuration["environmentFile"])
        self.debug = configuration["debugMode"]
        self.keepAlive = configuration["keepAlivePostExtinction"]
        self.agentEndowmentIndex = 0
        self.agentEndowments = []
        self.agentLeader = None
        self.agents = []
        self.llmFactionDecisions = {}
        self.conversations = []
        self.timestepTalkPairs = set()
        self.bornAgents = []
        self.deadAgents = []
        self.depression = True if configuration["agentDepressionPercentage"] > 0 else False
        self.diseaseEndowmentIndex = 0
        self.diseaseEndowments = []
        self.diseases = []
        self.remainingDiseases = []
        self.replacedAgents = []

        self.activeQuadrants = self.findActiveQuadrants()
        self.agentRuntimeStats = []
        self.configureDepression()
        self.configureAgents(configuration["startingAgents"])
        self.configureDiseases(configuration["startingDiseases"], configuration["diseaseList"])
        if configuration["headlessMode"] == False:
            import gui
            self.gui = gui.GUI(self, self.configuration["interfaceHeight"], self.configuration["interfaceWidth"])
        else:
            self.gui = None
        self.run = False # Simulation start flag
        self.end = False # Simulation end flag
        # TODO: Remove redundant metrics
        # TODO: Streamline naming
        self.runtimeStats = {"timestep": 0, "population": 0, "meanMetabolism": 0, "meanMovement": 0, "meanVision": 0, "meanWealth": 0, "meanAge": 0, "giniCoefficient": 0,
                             "meanTradePrice": 0, "tradeVolume": 0, "maxWealth": 0, "minWealth": 0, "meanHappiness": 0, "meanWealthHappiness": 0, "meanHealthHappiness": 0,
                             "meanSocialHappiness": 0, "meanFamilyHappiness": 0, "meanConflictHappiness": 0, "meanAgeAtDeath": 0, "seed": self.seed, "agentsReplaced": 0,
                             "agentsBorn": 0, "agentStarvationDeaths": 0, "agentDiseaseDeaths": 0, "environmentWealthCreated": 0, "agentWealthTotal": 0,
                             "environmentWealthTotal": 0, "agentWealthCollected": 0, "agentWealthBurnRate": 0, "agentMeanTimeToLive": 0, "agentTotalMetabolism": 0,
                             "agentCombatDeaths": 0, "agentAgingDeaths": 0, "agentDeaths": 0, "largestRace": 0, "largestTribe": 0, "largestRaceSize": 0, "largestTribeSize": 0,
                             "remainingRaces": self.configuration["environmentMaxRaces"], "remainingTribes": self.configuration["environmentMaxTribes"],
                             "sickAgents": 0, "carryingCapacity": 0, "meanDeathsPercentage": 0, "sickAgentsPercentage": 0, "meanSelfishness": 0,
                             "diseaseEffectiveReproductionRate": 0, "diseaseIncidence": 0, "diseasePrevalence": 0, "agentLastMoveOptimalityPercentage": 0, "meanNeighbors": 0,
                             "meanMoveRank": 0, "meanMoveDifferenceFromOptimal": 0, "meanValidMoves": 0, "totalHappiness": 0, "loanVolume": 0,
                             "meanAgeismFactor": 0, "meanRacismFactor": 0, "meanSexismFactor": 0, "moveSpace": 0
                             }
        self.graphStats = {"ageBins": [], "sugarBins": [], "spiceBins": [], "lorenzCurvePoints": [], "meanTribeTags": [],
                           "maxSugar": 0, "maxSpice": 0, "maxWealth": 0}

        self.log = open(configuration["logfile"], 'a') if configuration["logfile"] != None else None
        self.agentLog = open(configuration["agentLogfile"], 'a') if configuration["agentLogfile"] != None else None
        self.logFormat = configuration["logfileFormat"]
        self.experimentalGroup = configuration["experimentalGroup"]
        if self.experimentalGroup != None:
            self.groupMovementStats = {"meanControlNeighbors": 0, "meanExperimentalNeighbors": 0}
            self.runtimeStats.update(self.groupMovementStats)
            # Convert keys to Pythonic case scheme and initialize values
            groupRuntimeStats = {}
            for key in self.runtimeStats.keys():
                controlGroupKey = "control" + key[0].upper() + key[1:]
                experimentalGroupKey = self.experimentalGroup + key[0].upper() + key[1:]
                groupRuntimeStats[controlGroupKey] = 0
                groupRuntimeStats[experimentalGroupKey] = 0
            self.runtimeStats.update(groupRuntimeStats)
            self.groupInteractionRuntimeStats = {"combatControlGroupToControlGroup": 0, "combatControlGroupToExperimentalGroup": 0,
                                                 "combatExperimentalGroupToControlGroup": 0, "combatExperimentalGroupToExperimentalGroup": 0,
                                                 "diseaseControlGroupToControlGroup": 0, "diseaseControlGroupToExperimentalGroup": 0,
                                                 "diseaseExperimentalGroupToControlGroup": 0, "diseaseExperimentalGroupToExperimentalGroup": 0,
                                                 "lendingControlGroupToControlGroup": 0, "lendingControlGroupToExperimentalGroup": 0,
                                                 "lendingExperimentalGroupToControlGroup": 0, "lendingExperimentalGroupToExperimentalGroup": 0,
                                                 "reproductionControlGroupToControlGroup": 0, "reproductionControlGroupToExperimentalGroup": 0,
                                                 "reproductionExperimentalGroupToControlGroup": 0, "reproductionExperimentalGroupToExperimentalGroup": 0,
                                                 "tradeControlGroupToControlGroup": 0, "tradeControlGroupToExperimentalGroup": 0,
                                                 "tradeExperimentalGroupToControlGroup": 0, "tradeExperimentalGroupToExperimentalGroup": 0
                                                 }
            self.runtimeStats.update(self.groupInteractionRuntimeStats)


def verifyConfiguration(configuration):
    configuration.setdefault("simulationMode", "normal")
    configuration.setdefault("agentDistributionMode", "uniform")
    if "llmBackend" not in configuration:
        configuration["llmBackend"] = "jev" if configuration["simulationMode"] == "jev" else "vllm"
    configuration.setdefault("jevEndpoint", "https://api.typesafe.ai/v1/systemone")
    configuration.setdefault("jevModel", "jev-latest")
    configuration.setdefault("jevTimeout", 10)
    configuration.setdefault("jevOptions", {})
    configuration.setdefault("jevApiKey", None)
    configuration.setdefault("kevEndpoint", "http://127.0.0.1:8009/v1/systemone")
    configuration.setdefault("kevModel", "kev-latest")
    configuration.setdefault("kevTimeout", 10)
    configuration.setdefault("kevOptions", {})
    configuration.setdefault("kevApiKey", None)
    configuration.setdefault("geminiEndpoint", "https://generativelanguage.googleapis.com/v1beta")
    configuration.setdefault("geminiModel", "gemini-3.6-flash")
    configuration.setdefault("geminiApiKey", None)
    configuration.setdefault("geminiTimeout", 30)
    configuration.setdefault("geminiOptions", {"temperature": 0.2})
    configuration.setdefault("nanojevEndpoint", "http://127.0.0.1:8765")
    configuration.setdefault("nanojevModel", "NanoJev")
    configuration.setdefault("nanojevTimeout", 10)
    configuration.setdefault("nanojevOptions", {})
    configuration.setdefault("ollamaEndpoint", "http://127.0.0.1:11434")
    configuration.setdefault("ollamaModel", "llama3.1:8b")
    configuration.setdefault("ollamaTimeout", 10)
    configuration.setdefault("ollamaOptions", {})
    configuration.setdefault("vllmEndpoint", "http://127.0.0.1:8000/v1")
    configuration.setdefault("vllmModel", "meta-llama/Llama-3.1-8B-Instruct")
    configuration.setdefault("vllmTimeout", 10)
    configuration.setdefault("vllmOptions", {})
    if configuration["simulationMode"] not in ["normal", "llm", "jev"]:
        configuration["simulationMode"] = "normal"
    if configuration["llmBackend"] not in ["vllm", "ollama", "gemini", "jev", "kev", "nanojev"]:
        configuration["llmBackend"] = "vllm"
    if configuration["agentDistributionMode"] not in ["uniform", "distributed"]:
        configuration["agentDistributionMode"] = "uniform"
    if configuration["vllmTimeout"] <= 0:
        configuration["vllmTimeout"] = 10
    if configuration["ollamaTimeout"] <= 0:
        configuration["ollamaTimeout"] = 10
    if configuration["geminiTimeout"] <= 0:
        configuration["geminiTimeout"] = 30
    if configuration["nanojevTimeout"] <= 0:
        configuration["nanojevTimeout"] = 10
    if configuration["jevTimeout"] <= 0:
        configuration["jevTimeout"] = 10
    if configuration["kevTimeout"] <= 0:
        configuration["kevTimeout"] = 10
    if not isinstance(configuration["ollamaOptions"], dict):
        configuration["ollamaOptions"] = {}
    if not isinstance(configuration["vllmOptions"], dict):
        configuration["vllmOptions"] = {}
    if not isinstance(configuration["geminiOptions"], dict):
        configuration["geminiOptions"] = {"temperature": 0.2}
    if not isinstance(configuration["nanojevOptions"], dict):
        configuration["nanojevOptions"] = {}
    if not isinstance(configuration["kevOptions"], dict):
        configuration["kevOptions"] = {}
    if not isinstance(configuration["jevOptions"], dict):
        configuration["jevOptions"] = {}
    if "agentTalk" not in configuration:
        configuration["agentTalk"] = True if configuration["simulationMode"] in ["llm", "jev"] else False
    if not isinstance(configuration["agentTalk"], bool):
        configuration["agentTalk"] = True if configuration["simulationMode"] in ["llm", "jev"] else False
    configuration.setdefault("agentTalkMaxNeighbors", 1)
    if configuration["agentTalkMaxNeighbors"] <= 0:
        configuration["agentTalkMaxNeighbors"] = 1
    configuration.setdefault("resumeFromLog", True)
    if not isinstance(configuration["resumeFromLog"], bool):
        configuration["resumeFromLog"] = True



    negativesAllowed = ["agentDecisionModelAgeismFactor", "agentDecisionModelRacismFactor", "agentDecisionModelSexismFactor", "agentDecisionModelTribalFactor", "agentMaxAge", "agentSelfishnessFactor"]
    negativesAllowed += ["diseaseAggressionPenalty", "diseaseFertilityPenalty", "diseaseFriendlinessPenalty", "diseaseHappinessPenalty", "diseaseMovementPenalty"]
    negativesAllowed += ["diseaseSpiceMetabolismPenalty", "diseaseSugarMetabolismPenalty", "diseaseTimeframe", "diseaseVisionPenalty"]
    negativesAllowed += ["environmentAgeistAbsoluteRanges", "environmentAgeistRelativeRange", "environmentEquator", "environmentPollutionDiffusionTimeframe", "environmentPollutionTimeframe", "environmentMaxSpice", "environmentMaxSugar"]
    negativesAllowed += ["interfaceHeight", "interfaceWidth", "seed", "timesteps"]
    timeframes = ["diseaseTimeframe", "environmentPollutionDiffusionTimeframe", "environmentPollutionTimeframe"]
    negativeFlag = 0
    for configName, configValue in configuration.items():
        if isinstance(configValue, list):
            if len(configValue) == 0:
                continue
            configType = type(configValue[0])
            if configName in timeframes:
                configuration[configName] = sortConfigurationTimeframes(configuration, configName)
            else:
                configValue.sort()
            if configName not in negativesAllowed and (configType == int or configType == float):
                for i in range(len(configValue)):
                    if configValue[i] < 0:
                        configValue[i] = 0
                        negativeFlag += 1
        else:
            configType = type(configValue)
            if configName not in negativesAllowed and (configType == int or configType == float) and configValue < 0:
                configValue = 0
                negativeFlag += 1
    if negativeFlag > 0:
        if "all" in configuration["debugMode"] or "sugarscape" in configuration["debugMode"]:
            print(f"Detected negative values provided for {negativeFlag} option(s). Setting these values to zero.")

    # If no specific age range is tracked, revert to generic in-group age ranges experimental group
    if configuration["experimentalGroup"] != None and "ageRange" in configuration["experimentalGroup"]:
        experimentalAgeRangeID = re.search(r"ageRange(?P<ID>\d+)", configuration["experimentalGroup"])
        if experimentalAgeRangeID == None:
            configuration["experimentalGroup"] = "ageInGroup"

    # If no specific disease is tracked, revert to generic sick experimental group
    if configuration["experimentalGroup"] != None and "disease" in configuration["experimentalGroup"]:
        experimentalDiseaseID = re.search(r"disease(?P<ID>\d+)", configuration["experimentalGroup"])
        if experimentalDiseaseID == None:
            configuration["experimentalGroup"] = "sick"

    # If no specific race is tracked, revert to generic in-group races experimental group
    if configuration["experimentalGroup"] != None and "race" in configuration["experimentalGroup"]:
        experimentalRaceID = re.search(r"race(?P<ID>\d+)", configuration["experimentalGroup"])
        if experimentalRaceID == None:
            configuration["experimentalGroup"] = "raceInGroup"

    if configuration["environmentMaxSpice"] < 0:
        configuration["environmentMaxSpice"] = random.randint(1, 10)
    if configuration["environmentMaxSugar"] < 0:
        configuration["environmentMaxSugar"] = random.randint(1, 10)
    for peak in configuration["environmentSpicePeaks"]:
        if len(peak) < 3:
            peak.append(configuration["environmentMaxSpice"])
        if peak[0] < 0 or peak[0] > configuration["environmentWidth"]:
            peak[0] = random.randint(0, configuration["environmentWidth"] - 1)
        if peak[1] < 0 or peak[1] > configuration["environmentHeight"]:
            peak[1] = random.randint(0, configuration["environmentHeight"] - 1)
        if len(peak) < 3 or peak[2] < 0:
            peak[2] = random.randint(1, configuration["environmentMaxSpice"])
        elif peak[2] > configuration["environmentMaxSpice"]:
            peak[2] = configuration["environmentMaxSpice"]
    for peak in configuration["environmentSugarPeaks"]:
        if len(peak) < 3:
            peak.append(configuration["environmentMaxSugar"])
        if peak[0] < 0 or peak[0] > configuration["environmentWidth"]:
            peak[0] = random.randint(0, configuration["environmentWidth"] - 1)
        if peak[1] < 0 or peak[1] > configuration["environmentHeight"]:
            peak[1] = random.randint(0, configuration["environmentHeight"] - 1)
        if peak[2] < 0:
            peak[2] = random.randint(1, configuration["environmentMaxSugar"])
        elif peak[2] > configuration["environmentMaxSugar"]:
            peak[2] = configuration["environmentMaxSugar"]

    if configuration["environmentQuadrantSizeFactor"] > 1 or configuration["environmentQuadrantSizeFactor"] < 0:
        if "all" in configuration["debugMode"] or "environment" in configuration["debugMode"]:
            print(f"Cannot have a quadrant size factor of {configuration['environmentQuadrantSizeFactor']}. Setting quadrant size factor to 1.")
        configuration["environmentQuadrantSizeFactor"] = 1

    if len(configuration["environmentStartingQuadrants"]) == 0:
        configuration["environmentStartingQuadrants"] = [1, 2, 3, 4]

    if configuration["environmentTribePerQuadrant"] == True:
        configuration["environmentMaxTribes"] = len(configuration["environmentStartingQuadrants"])

    # Ensure starting agents are not larger than available cells
    totalCells = configuration["environmentHeight"] * configuration["environmentWidth"]
    totalCells = totalCells * (configuration["environmentQuadrantSizeFactor"] ** 2) * len(configuration["environmentStartingQuadrants"]) / 4
    if configuration["startingAgents"] > totalCells:
        if "all" in configuration["debugMode"] or "sugarscape" in configuration["debugMode"]:
            print(f"Could not allocate {configuration['startingAgents']} agents. Allocating maximum of {totalCells} agents.")
        configuration["startingAgents"] = totalCells

    # Set timesteps to (seemingly) unlimited runtime
    if configuration["timesteps"] < 0:
        if "all" in configuration["debugMode"] or "agent" in configuration["debugMode"]:
            print("Cannot have a negative amount of timesteps. Setting timesetup to unlimited runtime.")
        configuration["timesteps"] = sys.maxsize

    # Ensure infinitely-lived agents are properly initialized
    if configuration["agentDecisionModelAgeismFactor"][0] < 0:
        if configuration["agentDecisionModelAgeismFactor"][1] != -1:
            if "all" in configuration["debugMode"] or "agent" in configuration["debugMode"]:
                print(f"Cannot have agent ageism factor range of {configuration['agentDecisionModelAgeismFactor']}. Disabling agent ageism factor.")
        configuration["agentDecisionModelAgeismFactor"] = [-1, -1]
    elif configuration["agentDecisionModelAgeismFactor"][1] > 1:
        if "all" in configuration["debugMode"] or "agent" in configuration["debugMode"]:
            print(f"Cannot have agent maximum ageism factor of {configuration['agentDecisionModelAgeismFactor'][1]}. Setting agent maximum ageism factor to 1.0.")
        configuration["agentDecisionModelAgeismFactor"][1] = 1

    if configuration["agentDecisionModelRacismFactor"][0] < 0:
        if configuration["agentDecisionModelRacismFactor"][1] != -1:
            if "all" in configuration["debugMode"] or "agent" in configuration["debugMode"]:
                print(f"Cannot have agent racism factor range of {configuration['agentDecisionModelRacismFactor']}. Disabling agent racism factor.")
        configuration["agentDecisionModelRacismFactor"] = [-1, -1]
    elif configuration["agentDecisionModelRacismFactor"][1] > 1:
        if "all" in configuration["debugMode"] or "agent" in configuration["debugMode"]:
            print(f"Cannot have agent maximum racism factor of {configuration['agentDecisionModelRacismFactor'][1]}. Setting agent maximum racism factor to 1.0.")
        configuration["agentDecisionModelRacismFactor"][1] = 1

    if configuration["agentDecisionModelSexismFactor"][0] < 0:
        if configuration["agentDecisionModelSexismFactor"][1] != -1:
            if "all" in configuration["debugMode"] or "agent" in configuration["debugMode"]:
                print(f"Cannot have agent sexism factor range of {configuration['agentDecisionModelSexismFactor']}. Disabling agent sexism factor.")
        configuration["agentDecisionModelSexismFactor"] = [-1, -1]
    elif configuration["agentDecisionModelSexismFactor"][1] > 1:
        if "all" in configuration["debugMode"] or "agent" in configuration["debugMode"]:
            print(f"Cannot have agent maximum sexism factor of {configuration['agentDecisionModelSexismFactor'][1]}. Setting agent maximum sexism factor to 1.0.")
        configuration["agentDecisionModelSexismFactor"][1] = 1

    if configuration["agentDecisionModelTribalFactor"][0] < 0:
        if configuration["agentDecisionModelTribalFactor"][1] != -1:
            if "all" in configuration["debugMode"] or "agent" in configuration["debugMode"]:
                print(f"Cannot have age tribal factor range of {configuration['agentDecisionModelTribalFactor']}. Disabling agent tribal factor.")
        configuration["agentDecisionModelTribalFactor"] = [-1, -1]
    elif configuration["agentDecisionModelTribalFactor"][1] > 1:
        if "all" in configuration["debugMode"] or "agent" in configuration["debugMode"]:
            print(f"Cannot have agent maximum tribal factor of {configuration['agentDecisionModelTribalFactor'][1]}. Setting agent maximum tribal factor to 1.0.")
        configuration["agentDecisionModelTribalFactor"][1] = 1

    if configuration["agentMaxAge"][0] < 0:
        if configuration["agentMaxAge"][1] != -1:
            if "all" in configuration["debugMode"] or "agent" in configuration["debugMode"]:
                print(f"Agent cannot have agent maximum age range of {configuration['agentMaxAge']}. Setting agents to live infinitely.")
        configuration["agentMaxAge"] = [-1, -1]

    if configuration["agentSelfishnessFactor"][0] < 0:
        if configuration["agentSelfishnessFactor"][1] != -1:
            if "all" in configuration["debugMode"] or "agent" in configuration["debugMode"]:
                print(f"Cannot have agent selfishness factor range of {configuration['agentSelfishnessFactor']}. Disabling agent selfishness.")
        configuration["agentSelfishnessFactor"] = [-1, -1]
    elif configuration["agentSelfishnessFactor"][1] > 1:
        if "all" in configuration["debugMode"] or "agent" in configuration["debugMode"]:
            print(f"Cannot have agent maximum selfishness factor of {configuration['agentSelfishnessFactor'][1]}. Setting agent maximum selfishness factor to 1.0.")
        configuration["agentSelfishnessFactor"][1] = 1

    if configuration["agentDynamicDecisionModelFactor"][0] < 0:
        if configuration["agentTemperanceFactor"][1] != -1:
            if "all" in configuration["debugMode"] or "agent" in configuration["debugMode"]:
                print(f"Cannot have agent dynamic decision model factor of {configuration['agentDynamicDecisionModelFactor']}. Disabling agent temperance.")
        configuration["agentDynamicDecisionModelFactor"] = [-1,-1]
    elif configuration["agentDynamicDecisionModelFactor"][1] > 1:
        if "all" in configuration["debugMode"] or "agent" in configuration["debugMode"]:
            print(f"Cannot have agent maximum dynamic decision model factor of {configuration['agentDynamicDecisionModelFactor'][1]}. Setting agent maximum dynamic decision model change to 1.0.")
        configuration["agentDynamicDecisionModelFactor"][1] = 1.0
    
    if configuration["agentDynamicSocialPressureFactor"][0] < 0:
        if configuration["agentTemperanceFactor"][1] != -1:
            if "all" in configuration["debugMode"] or "agent" in configuration["debugMode"]:
                print(f"Cannot have agent dynamic social pressure factor of {configuration['agentDynamicSocialPressureFactor']}. Disabling agent social pressure.")
        configuration["agentDynamicSocialPressureFactor"] = [-1,-1]
    elif configuration["agentDynamicSocialPressureFactor"][1] > 1:
        if "all" in configuration["debugMode"] or "agent" in configuration["debugMode"]:
            print(f"Cannot have agent maximum dynamic social pressure factor of {configuration['agentDynamicSocialPressureFactor'][1]}. Setting agent maximum dynamic social change to 1.0.")
        configuration["agentDynamicSocialPressureFactor"][1] = 1.0

    if configuration["agentRacialTagStringLength"] < 0:
        if "all" in configuration["debugMode"] or "agent" in configuration["debugMode"]:
            print(f"Cannot have a negative agent racial tag string length. Setting agent racial tag string length to 0.")
        configuration["agentRacialTagStringLength"] = 0

    if configuration["agentTagStringLength"] < 0:
        if "all" in configuration["debugMode"] or "agent" in configuration["debugMode"]:
            print(f"Cannot have a negative agent tag string length. Setting agent tag string length to 0.")
        configuration["agentTagStringLength"] = 0

    if configuration["environmentMaxRaces"] < 0:
        if "all" in configuration["debugMode"] or "environment" in configuration["debugMode"]:
            print(f"Cannot have a negative number of races. Setting number of races to 0.")
        configuration["environmentMaxRaces"] = 0

    if configuration["environmentMaxTribes"] < 0:
        if "all" in configuration["debugMode"] or "environment" in configuration["debugMode"]:
            print(f"Cannot have a negative number of tribes. Setting number of tribes to 0.")
        configuration["environmentMaxTribes"] = 0

    # Ensure at most number of tribes is equal to agent tag string length
    if configuration["agentTagStringLength"] > 0 and configuration["environmentMaxTribes"] > configuration["agentTagStringLength"]:
        if "all" in configuration["debugMode"] or "environment" in configuration["debugMode"] or "agent" in configuration["debugMode"]:
            print(f"Cannot have a longer agent tag string length than maximum number of tribes. Setting the number of tribes to {configuration['agentTagStringLength']}.")
        configuration["environmentMaxTribes"] = configuration["agentTagStringLength"]

    # Ensure at most number of races, tribes, and decision models are equal to the number of colors in the GUI
    maxColors = 25
    uniqueAgentDecisionModels = set(configuration["agentDecisionModels"])
    numUniqueAgentDecisionModels = len(uniqueAgentDecisionModels)
    if numUniqueAgentDecisionModels > maxColors:
        if "all" in configuration["debugMode"] or "agent" in configuration["debugMode"]:
            print(
                f"Cannot provide {len(configuration['agentDecisionModels'])} decision models. Allocating maximum of {maxColors}.")
        removeDecisionModels = uniqueAgentDecisionModels[maxColors:]
        configuration["agentDecisionModels"] = [i for i in configuration["agentDecisionModels"] if
                                                i not in removeDecisionModels]
    if configuration["environmentMaxRaces"] > maxColors:
        if "all" in configuration["debugMode"] or "sugarscape" in configuration["debugMode"] or "environment" in configuration["debugMode"]:
            print(f"Cannot provide {configuration['environmentMaxRaces']} races. Allocating maximum of {maxColors}.")
        configuration["environmentMaxRaces"] = maxColors
    if configuration["environmentMaxTribes"] > maxColors:
        if "all" in configuration["debugMode"] or "sugarscape" in configuration["debugMode"] or "environment" in configuration["debugMode"]:
            print(f"Cannot provide {configuration['environmentMaxTribes']} tribes. Allocating maximum of {maxColors}.")
        configuration["environmentMaxTribes"] = maxColors

    # Ensure no negative values for ageism absolute ranges
    for ageRange in configuration["environmentAgeistAbsoluteRanges"][:]:
        minAge, maxAge = ageRange
        if minAge < -1 or maxAge < -1 or (minAge > maxAge and maxAge != -1):
            if "all" in configuration["debugMode"] or "environment" in configuration["debugMode"]:
                print(f"Cannot have ageism absolute range of [{minAge}, {maxAge}]. Removing range from list.")
            configuration["environmentAgeistAbsoluteRanges"].remove(ageRange)

    # Ensure no negative value for ageism relative range
    if configuration["environmentAgeistRelativeRange"] < 0 and configuration["environmentAgeistRelativeRange"] != -1:
        if "all" in configuration["debugMode"] or "environment" in configuration["debugMode"]:
            print(f"Cannot have negative ageism relative range. Disabling environmentAgeistRelativeRange.")
        configuration["environmentAgeistRelativeRange"] = -1
    
    # If both environmentAgeistAbsoluteRanges and environmentAgeistRelativeRange are disabled, ageism must be disabled since there is no mechanism for determining in-grouping
    if configuration["environmentAgeistAbsoluteRanges"] == [] and configuration["environmentAgeistRelativeRange"] == -1 and configuration["agentDecisionModelAgeismFactor"] != [-1, -1]:
        if "all" in configuration["debugMode"] or "agent" in configuration["debugMode"]:
            print(f"Cannot have ageism without in-grouping mechanism. Disabling agentDecisionModelAgeismFactor.")
        configuration["agentDecisionModelAgeismFactor"] = [-1, -1]

    if any(race >= configuration["environmentMaxRaces"] for race in configuration["environmentInGroupRaces"]):
        if "all" in configuration["debugMode"] or "environment" in configuration["debugMode"]:
            print(f"Cannot have in-group races greater than total races. Removing in-group races greater than or equal to {configuration['environmentMaxRaces']}")
        configuration["environmentInGroupRaces"] = [race for race in configuration["environmentInGroupRaces"] if race < configuration["environmentMaxRaces"]]

    # Ensure the most number of starting diseases per agent is equal to total starting diseases in the environment
    if configuration["startingDiseasesPerAgent"] != [0, 0]:
        startingDiseasesPerAgent = sorted(configuration["startingDiseasesPerAgent"])
        startingDiseasesPerAgent = [max(0, numDiseases) for numDiseases in startingDiseasesPerAgent]
        startingDiseases = configuration["startingDiseases"]
        startingDiseasesPerAgent = [min(startingDiseases, numDiseases) for numDiseases in startingDiseasesPerAgent]
        if "all" in configuration["debugMode"] or "disease" in configuration["debugMode"] and startingDiseasesPerAgent != configuration["startingDiseasesPerAgent"]:
            print(f"Range of starting diseases per agent exceeds {startingDiseases} starting diseases. Setting range to {startingDiseasesPerAgent}.")
        configuration["startingDiseasesPerAgent"] = startingDiseasesPerAgent

    if configuration["logfile"] == "":
        configuration["logfile"] = None

    if configuration["agentLogfile"] == "":
        configuration["agentLogfile"] = None

    recognizedDebugModes = ["agent", "all", "cell", "disease", "environment", "ethics", "none", "sugarscape", "talk"]
    validModes = True
    for mode in configuration["debugMode"]:
        if mode not in recognizedDebugModes:
            print(f"Debug mode {mode} not recognized")
            validModes = False
    if validModes == False:
        printHelp()

    if "all" in configuration["debugMode"] and "none" in configuration["debugMode"]:
        print("Cannot have \"all\" and \"none\" debug modes enabled at the same time")
        printHelp()
    elif "all" in configuration["debugMode"] and len(configuration["debugMode"]) > 1:
        configuration["debugMode"] = "all"
    elif "none" in configuration["debugMode"] and len(configuration["debugMode"]) > 1:
        configuration["debugMode"] = "none"

    # Keep compatibility with outdated configuration files
    if configuration["agentDecisionModel"] != None and type(configuration["agentDecisionModel"]) == str:
            configuration["agentDecisionModels"] = [configuration["agentDecisionModel"]]
    elif configuration["agentDecisionModel"] != None and type(configuration["agentDecisionModel"]) == list:
            configuration["agentDecisionModels"] = configuration["agentDecisionModel"]
    if type(configuration["agentDecisionModels"]) == str:
            configuration["agentDecisionModels"] = [configuration["agentDecisionModels"]]

    # Ensure experimental group is properly defined or otherwise ignored
    if configuration["experimentalGroup"] == "":
        configuration["experimentalGroup"] = None
    groupList = ["ageInGroup", "depressed", "female", "male", "raceInGroup", "sick"]
    if type(configuration["agentDecisionModels"]) == str:
        groupList.append(configuration["agentDecisionModels"])
    else:
        groupList += configuration["agentDecisionModels"]
    if configuration["experimentalGroup"] != None and configuration["experimentalGroup"] not in groupList and "ageRange" not in configuration["experimentalGroup"] and "disease" not in configuration["experimentalGroup"] and "race" not in configuration["experimentalGroup"]:
        if "all" in configuration["debugMode"] or "agent" in configuration["debugMode"]:
            print(f"Cannot provide separate log stats for experimental group {configuration['experimentalGroup']}. Disabling separate log stats.")
        configuration["experimentalGroup"] = None
    return configuration

def verifyRandomSeed(configuration):
    if type(configuration["seed"]) != int or configuration["seed"] == -1:
        configuration["seed"] = random.randrange(sys.maxsize)
    random.seed(configuration["seed"])


if __name__ == "__main__":
    from sugarscape_cli import main

    main()
