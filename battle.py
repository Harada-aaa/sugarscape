"""Battle mode: run a single Sugarscape environment with two factions.

Each faction is configured by its own Sugarscape config file and acts through
the agent backend described by that config (e.g. normal rule-based agents vs.
LLM agents). Both factions share one environment, whose settings are taken from
the first faction's config.

Usage:
	python battle.py --conf1 config/normal.json --conf2 config/llm.json
"""

import copy
import getopt
import json
import math
import os
import sys

from clients import gemini_client
from clients import clef_client
from clients import kev_client
from clients import hybrid_client
from clients import nanojev_client
from clients import jev_client
from clients import ollama_client
from clients import tev_client
from clients import vllm_client
from sugarscape import Sugarscape, verifyRandomSeed, verifyConfiguration

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config", "default.json")

# Configuration keys that define how agents behave as opposed to how the
# shared environment behaves. These are applied per faction from each config.
AGENT_CONFIG_PREFIX = "agent"
AGENT_CONFIG_KEYS = {"debugMode", "neighborhoodMode"}
AGENT_COUNT_KEYS = {"startingAgents", "startingDiseases", "startingDiseasesPerAgent"}
FACTION_CONFIG_PREFIXES = ["jev", "kev", "tev", "nanojev", "gemini", "ollama", "hybrid", "vllm", "clef", "llm"]
FACTION_CONFIG_KEYS = {"simulationMode"}

defaultFactionQuadrants = {1: [1, 3], 2: [2, 4]}


class GreedyClient:
	"""Fallback client for rule-based ("normal") factions.

	When the shared simulation runs in an LLM mode, movement decisions are
	routed through the client interface. Non-LLM factions still move greedily,
	so this client simply always selects the greedy best candidate (rank 0).
	"""

	def choose_cells(self, faction_state):
		return {str(entry["agent"]["id"]): 0 for entry in faction_state}

	def choose_cells_parallel(self, faction_states):
		return [self.choose_cells(faction_state) for faction_state in faction_states]

	def talk(self, speaker_state, listener_state):
		return None


class FactionClientRouter:
	"""Routes LLM calls to the client belonging to the acting faction.

	Factions are distinguished by their tribe, which is included in the agent
	state of every movement and talk request.
	"""

	def __init__(self, clients):
		# clients maps the stringified faction tribe to an LLM client
		self.clients = clients
		self.defaultClient = list(clients.values())[0] if len(clients) > 0 else GreedyClient()

	def findClient(self, tribe):
		return self.clients.get(str(tribe), self.defaultClient)

	def choose_cells(self, faction_state):
		tribe = "None"
		if len(faction_state) > 0:
			tribe = faction_state[0]["agent"].get("tribe", "None")
		return self.findClient(tribe).choose_cells(faction_state)

	def choose_cells_parallel(self, faction_states):
		decisions = [None for i in range(len(faction_states))]
		statesByClient = {}
		for index, faction_state in enumerate(faction_states):
			tribe = "None"
			if len(faction_state) > 0:
				tribe = faction_state[0]["agent"].get("tribe", "None")
			client = self.findClient(tribe)
			key = id(client)
			if key not in statesByClient:
				statesByClient[key] = {"client": client, "indices": [], "states": []}
			statesByClient[key]["indices"].append(index)
			statesByClient[key]["states"].append(faction_state)
		# Each backend receives one parallel call containing only its own factions
		for entry in statesByClient.values():
			clientDecisions = entry["client"].choose_cells_parallel(entry["states"])
			for index, decision in zip(entry["indices"], clientDecisions):
				decisions[index] = decision
		return decisions

	def talk(self, speaker_state, listener_state):
		client = self.findClient(speaker_state.get("tribe", "None"))
		if not hasattr(client, "talk"):
			return None
		return client.talk(speaker_state, listener_state)


class BattleSugarscape(Sugarscape):
	def __init__(self, configuration):
		super().__init__(configuration)
		self.factionConfigs = {}
		self.factionClients = {}
		self.llmClient = FactionClientRouter({})

	# ------------------------------------------------------------------
	# Faction configuration
	# ------------------------------------------------------------------
	def registerFaction(self, factionID, factionConfig):
		"""Prepare one faction: its own agent settings, starting quadrants,
		tribe tags, and agent client."""
		quadrants = factionConfig["environmentStartingQuadrants"]
		quadrantCells = self.findFactionQuadrants(quadrants)
		tribe = (factionID - 1) % max(1, self.configuration["environmentMaxTribes"])
		numPlaced = self.configureFactionAgents(copy.deepcopy(self.configuration), factionConfig, quadrantCells, tribe, factionID)
		self.factionClients[str(tribe)] = buildFactionClient(factionConfig)
		self.factionConfigs[factionID] = factionConfig
		print(f"Battle faction {factionID}: tribe {tribe}, quadrants {quadrants}, "
			  f"simulationMode {factionConfig['simulationMode']}, agents placed {numPlaced}")

	def findFactionQuadrants(self, quadrants):
		"""Return the list of quadrant cell lists matching given quadrant IDs."""
		savedQuadrants = self.configuration["environmentStartingQuadrants"]
		self.configuration["environmentStartingQuadrants"] = quadrants
		try:
			quadrantCells = self.findActiveQuadrants()
		finally:
			self.configuration["environmentStartingQuadrants"] = savedQuadrants
		return quadrantCells

	def configureFactionAgents(self, baseConfig, factionConfig, quadrantCells, tribe, factionID):
		"""Place this faction's agents by temporarily overriding agent settings
		with the faction's own configuration. Returns the number of agents.
		The configured agent count is kept as the battle total and split evenly
		between the two factions (e.g. 50 -> 25 vs 25)."""
		totalAgents = factionConfig["startingAgents"]
		if factionID == 1:
			numAgents = math.ceil(totalAgents / 2)
		else:
			numAgents = totalAgents // 2
		if numAgents <= 0:
			return 0
		joinedConfig = copy.deepcopy(baseConfig)
		for key, value in factionConfig.items():
			if key.startswith(AGENT_CONFIG_PREFIX) or key in AGENT_CONFIG_KEYS:
				joinedConfig[key] = value
		joinedConfig["agentLeader"] = False

		savedConfig = self.configuration
		savedQuadrants = self.activeQuadrants
		savedEndowments = self.agentEndowments
		savedEndowmentIndex = self.agentEndowmentIndex
		savedAgentCount = len(self.agents)
		try:
			self.configuration = joinedConfig
			self.agentEndowments = []
			self.agentEndowmentIndex = 0
			self.activeQuadrants = quadrantCells
			self.configureAgents(numAgents)
		finally:
			factionEndowments = self.agentEndowments
			self.configuration = savedConfig
			self.activeQuadrants = savedQuadrants
			# Keep faction endowments around so replacement agents stay
			# faithful to the faction they belong to
			self.agentEndowments = savedEndowments + factionEndowments
			self.agentEndowmentIndex = savedEndowmentIndex + len(factionEndowments)

		newAgents = self.agents[savedAgentCount:]
		# Assign battle tribe tags so that agents of this faction share one
		# tribe (and therefore one LLM faction decision group)
		for a in newAgents:
			a.tags = self.generateTribeTags(tribe)
			a.tribe = a.findTribe()
			a.tagZeroes = a.tags.count(0)
		return len(newAgents)

	def printBattleBanner(self):
		print("=== Battle! ===")
		for factionID, config in self.factionConfigs.items():
			print(f"Faction {factionID}: {config['simulationMode']} agents "
				  f"({config['startingAgents'] // 2 if factionID != 1 else -(-config['startingAgents'] // 2)})"
				  f" in quadrants {config['environmentStartingQuadrants']}")

	# ------------------------------------------------------------------
	# Reporting
	# ------------------------------------------------------------------
	def printFactionSummary(self):
		stats = {}
		for a in self.agents:
			if not a.isAlive():
				continue
			stats.setdefault(a.tribe, {"count": 0, "wealth": 0})
			stats[a.tribe]["count"] += 1
			stats[a.tribe]["wealth"] += a.sugar + a.spice
		print("\n=== Battle summary ===")
		for factionID, config in self.factionConfigs.items():
			tribe = (factionID - 1) % max(1, self.configuration["environmentMaxTribes"])
			factionStats = stats.get(tribe, {"count": 0, "wealth": 0})
			print(f"Faction {factionID} ({config['simulationMode']}): survivors "
				  f"{factionStats['count']}, wealth {factionStats['wealth']:.2f}")


def buildFactionClient(configuration):
	"""Instantiate the client described by a faction's configuration."""
	if configuration["simulationMode"] == "hybrid":
		return hybrid_client.HybridClient(
			configuration["ollamaEndpoint"], configuration["hybridNimbleModel"], configuration["hybridQwenModel"],
			configuration["hybridTimeout"], configuration["hybridOptions"],
			configuration["hybridDifficultyCandidateThreshold"], configuration["hybridDifficultyWealthMargin"],
			configuration["hybridDifficultyScarcityFactor"])
	elif configuration["llmBackend"] == "ollama":
		return ollama_client.OllamaClient(
			configuration["ollamaEndpoint"], configuration["ollamaModel"], configuration["ollamaTimeout"],
			configuration["ollamaOptions"], configuration.get("ollamaThink"))
	elif configuration["llmBackend"] == "gemini":
		return gemini_client.GeminiClient(
			configuration["geminiEndpoint"], configuration["geminiModel"], configuration["geminiTimeout"],
			configuration["geminiOptions"], configuration["geminiApiKey"])
	elif configuration["llmBackend"] == "nanojev":
		return nanojev_client.NanoJevClient(
			configuration["nanojevEndpoint"], configuration["nanojevModel"], configuration["nanojevTimeout"],
			configuration["nanojevOptions"])
	elif configuration["llmBackend"] == "jev":
		return jev_client.JevClient(
			configuration["jevEndpoint"], configuration["jevModel"], configuration["jevTimeout"],
			configuration["jevOptions"], configuration["jevApiKey"])
	elif configuration["llmBackend"] == "kev":
		return kev_client.KevClient(
			configuration["kevEndpoint"], configuration["kevModel"], configuration["kevTimeout"],
			configuration["kevOptions"], configuration["kevApiKey"])
	elif configuration["llmBackend"] == "tev1":
		return tev_client.TevClient(
			configuration["tevEndpoint"], configuration["tevModel"], configuration["tevTimeout"],
			configuration["tevOptions"])
	elif configuration["llmBackend"] == "clef":
		return clef_client.ClefClient(
			configuration["clefEndpoint"], configuration["clefModel"], configuration["clefTimeout"],
			configuration["clefOptions"], configuration["clefApiKey"], configuration["clefAccountId"])
	elif configuration["simulationMode"] in ["llm", "jev", "hybrid"]:
		return vllm_client.VLLMClient(
			configuration["vllmEndpoint"], configuration["vllmModel"], configuration["vllmTimeout"],
			configuration["vllmOptions"])
	# Rule-based faction: use the greedy fallback client
	return GreedyClient()


def loadConfiguration(configFile):
	from sugarscape_config import parseConfiguration
	configuration = json.load(open(DEFAULT_CONFIG_PATH, encoding="utf-8"))["sugarscapeOptions"]
	configuration = parseConfiguration(configFile, configuration)
	configuration["logfile"] = None
	configuration["agentLogfile"] = None
	return configuration


def parseOptions():
	commandLineArgs = sys.argv[1:]
	try:
		parsedOptionPairs = getopt.getopt(commandLineArgs, "", ["conf1=", "conf2=", "help"])
	except getopt.GetoptError as err:
		print(err)
		printHelp()
	factionFiles = [None, None]
	for currArg, currVal in parsedOptionPairs[0]:
		if currArg in ("-c1", "--conf1"):
			factionFiles[0] = currVal
		elif currArg in ("-c2", "--conf2"):
			factionFiles[1] = currVal
		elif currArg in ("-h", "--help"):
			printHelp()
	if factionFiles[0] == None or factionFiles[1] == None:
		print("Battle mode requires two config files: --conf1 config/normal.json --conf2 config/llm.json")
		printHelp()
	return factionFiles


def printHelp():
	print("Usage:\n\tpython battle.py --conf1 config/normal.json --conf2 config/llm.json\n\nOptions:\n\t--conf1\tConfig file for faction 1.\n\t--conf2\tConfig file for faction 2.\n\t--help\tDisplay this message.")
	exit(0)


def buildFactionConfig(merged, ownConfig, ownQuadrants, factionID):
	"""Create the agent-behavior configuration for a faction. The shared
	environment stays untouched; only agent-related options come from the
	faction's own config file."""
	factionConfig = copy.deepcopy(merged)
	# Agent-behavior options and this faction's agent backend come from the
	# faction's own config file
	for key, value in ownConfig.items():
		if key.startswith(AGENT_CONFIG_PREFIX) or key in AGENT_CONFIG_KEYS or key in AGENT_COUNT_KEYS:
			factionConfig[key] = value
		elif any(key.startswith(prefix) for prefix in FACTION_CONFIG_PREFIXES) or key in FACTION_CONFIG_KEYS:
			factionConfig[key] = value
	factionConfig["environmentStartingQuadrants"] = ownQuadrants
	factionConfig["startingAgents"] = ownConfig["startingAgents"]
	return factionConfig


def makeQuadrantsDisjoint(conf1, conf2):
	"""Battle factions should start in separate quadrants. If the two configs
	share quadrants, give faction 2 the unoccupied quadrants; if the whole
	environment is occupied, split it diagonally."""
	remaining = [q for q in [1, 2, 3, 4] if q not in conf1["environmentStartingQuadrants"]]
	if set(conf1["environmentStartingQuadrants"]) & set(conf2["environmentStartingQuadrants"]) == set():
		return
	if len(remaining) >= len(conf2["environmentStartingQuadrants"]):
		conf2["environmentStartingQuadrants"] = remaining[:len(conf2["environmentStartingQuadrants"])]
	elif len(remaining) > 0:
		conf2["environmentStartingQuadrants"] = remaining
	if conf1["environmentStartingQuadrants"] == [1, 2, 3, 4]:
		conf1["environmentStartingQuadrants"] = [1, 3]
		conf2["environmentStartingQuadrants"] = [2, 4]
		print("Environment fully occupied. Splitting quadrants between factions: "
			  f"faction 1 -> {conf1['environmentStartingQuadrants']}, faction 2 -> {conf2['environmentStartingQuadrants']}")
	else:
		print(f"Faction quadrants overlapping. Faction 2 starts in quadrants {conf2['environmentStartingQuadrants']}.")


def mergeBattleConfiguration(conf1, conf2):
	merged = copy.deepcopy(conf1)
	# The environment belongs to both factions: combine starting quadrants and
	# make sure two distinct tribes exist so factions can be told apart
	quadrants = sorted(set(conf1["environmentStartingQuadrants"]) | set(conf2["environmentStartingQuadrants"]))
	merged["environmentStartingQuadrants"] = quadrants
	merged["environmentTribePerQuadrant"] = True
	merged["environmentMaxTribes"] = max(2, conf1["environmentMaxTribes"])
	merged["startingAgents"] = 0  # Agents are placed per faction afterwards
	if merged["agentTagStringLength"] == 0:
		merged["agentTagStringLength"] = 11  # Tribe tags are required to separate factions
	if conf1["simulationMode"] in ["llm", "jev", "hybrid"] or conf2["simulationMode"] in ["llm", "jev", "hybrid"]:
		merged["simulationMode"] = "llm"
		merged["agentTalk"] = conf1.get("agentTalk", False) or conf2.get("agentTalk", False)
	if conf1["agentLeader"] == True or conf2["agentLeader"] == True:
		merged["agentLeader"] = False
		print("Battle mode does not support agent leaders. Disabling agentLeader.")
	return merged


def main():
	factionFiles = parseOptions()
	conf1 = loadConfiguration(factionFiles[0])
	conf2 = loadConfiguration(factionFiles[1])
	if "environmentStartingQuadrants" not in conf1:
		conf1["environmentStartingQuadrants"] = defaultFactionQuadrants[1]
	if "environmentStartingQuadrants" not in conf2:
		conf2["environmentStartingQuadrants"] = defaultFactionQuadrants[2]
	makeQuadrantsDisjoint(conf1, conf2)

	merged = mergeBattleConfiguration(conf1, conf2)
	verifyRandomSeed(merged)
	merged = verifyConfiguration(merged)

	factionConfigs = []
	ownQuadrants = [conf1["environmentStartingQuadrants"], conf2["environmentStartingQuadrants"]]
	for factionID, ownConfig in [(1, conf1), (2, conf2)]:
		factionConfig = buildFactionConfig(merged, ownConfig, ownQuadrants[factionID - 1], factionID)
		factionConfig = verifyConfiguration(factionConfig)
		factionConfigs.append(factionConfig)

	print("=== Battle configuration ===")
	print(f"faction 1: {factionFiles[0]} (quadrants {conf1['environmentStartingQuadrants']}, "
		  f"mode {conf1['simulationMode']}, backend {conf1['llmBackend']})")
	print(f"faction 2: {factionFiles[1]} (quadrants {conf2['environmentStartingQuadrants']}, "
		  f"mode {conf2['simulationMode']}, backend {conf2['llmBackend']})")

	if merged["headlessMode"] == False:
		import gui

	simulation = BattleSugarscape(merged)
	for factionID, factionConfig in enumerate(factionConfigs, start=1):
		simulation.registerFaction(factionID, factionConfig)
	simulation.llmClient = FactionClientRouter(simulation.factionClients)

	if merged["profileMode"] == True:
		import cProfile
		import tracemalloc
		tracemalloc.start()
		cProfile.run("simulation.runSimulation(merged[\"timesteps\"])")
		snapshot = tracemalloc.take_snapshot()
		memoryStats = snapshot.statistics("lineno", True)
		for stat in memoryStats[:100]:
			print(stat)
	# endSimulation() exits the process; catch it so the battle summary prints
	try:
		simulation.runSimulation(merged["timesteps"])
	except SystemExit:
		pass
	simulation.printFactionSummary()
	exit(0)


if __name__ == "__main__":
	main()
