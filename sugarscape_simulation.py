"""Simulation lifecycle and endowment generation for Sugarscape."""

import hashlib
import json
import math
import random
import time

class SugarscapeSimulationMethods:
    def doTimestep(self):
        if self.timestep >= self.maxTimestep:
            self.toggleEnd()
            return
        if "all" in self.debug or "sugarscape" in self.debug:
            print(f"Timestep: {self.timestep}\nLiving Agents: {len(self.agents)}")
        self.timestep += 1
        if self.end == True or (len(self.agents) == 0 and self.keepAlive == False):
            self.toggleEnd()
        else:
            self.timestepTalkPairs.clear()
            self.environment.doTimestep(self.timestep)
            random.shuffle(self.agents)
            self.addRemainingDiseases()
            if self.agentLeader != None:
                self.agentLeader.doTimestep(self.timestep)
            for agent in self.agents:
                if self.agentLeader != None and agent == self.agentLeader:
                    continue
                agent.doTimestep(self.timestep)
            self.removeDeadAgents()
            self.replaceDeadAgents()
            self.updateRuntimeStats()
            if self.gui != None:
                self.updateGraphStats()
                self.gui.doTimestep()
            # If final timestep, do not write to log to cleanly close JSON array log structure
            if not self.replaying and self.timestep != self.maxTimestep and len(self.agents) > 0:
                self.writeToLog(self.log)
                # Start recording agent actions only after agents have started acting
                if self.timestep == 1 and not self.agentLogStarted:
                    self.startLog(self.agentLog)
                    self.agentLogStarted = True
                self.writeToLog(self.agentLog)
                self.agentRuntimeStats = []
            elif self.replaying:
                self.agentRuntimeStats = []

    def endLog(self, log):
        if log == None:
            return
        # Update total wealth accumulation to include still living agents at simulation end
        stats = self.runtimeStats
        logString = ""
        if log == self.agentLog:
            stats = self.agentRuntimeStats
            for agentStats in stats:
                if agentStats == stats[-1]:
                    logString += f"\t{json.dumps(agentStats)}\n]"
                else:
                    logString += f"\t{json.dumps(agentStats)},\n"
        else:
            environmentWealthCreated = 0
            environmentWealthTotal = 0
            for i in range(self.environment.width):
                for j in range(self.environment.height):
                    environmentWealthCreated += self.environment.grid[i][j].sugarLastProduced + self.environment.grid[i][j].spiceLastProduced
                    environmentWealthTotal += self.environment.grid[i][j].sugar + self.environment.grid[i][j].spice
            self.runtimeStats["environmentWealthCreated"] = environmentWealthCreated
            self.runtimeStats["environmentWealthTotal"] = environmentWealthTotal
            logString = f"\t{json.dumps(stats)}\n]"
        if self.logFormat == "csv":
            logString = ""
            # Ensure consistent ordering for CSV format
            if log == self.agentLog:
                for agentStats in stats:
                    for stat in agentStats:
                        if logString == "":
                            logString += f"{agentStats[stat]}"
                        else:
                            logString += f",{agentStats[stat]}"
                    logString += "\n"
            else:
                for stat in sorted(stats):
                    if logString == "":
                        logString += f"{stats[stat]}"
                    else:
                        logString += f",{stats[stat]}"
                logString += "\n"
        log.write(logString)
        log.flush()
        log.close()

    def endSimulation(self):
        self.removeDeadAgents()
        self.endLog(self.log)
        self.endLog(self.agentLog)
        print("=== Simulation results ===")
        print("totalHappiness: " + str(self.runtimeStats["totalHappiness"]))
        print(f"totalExecutionTime: {time.perf_counter() - self.startTime:.2f} seconds")
        if "all" in self.debug or "sugarscape" in self.debug:
            print(str(self))
        exit(0)

    def findActiveQuadrants(self):
        quadrants = self.configuration["environmentStartingQuadrants"]
        cellRange = []
        quadrantWidth = math.floor(self.environmentWidth / 2 * self.configuration["environmentQuadrantSizeFactor"])
        quadrantHeight = math.floor(self.environmentHeight / 2 * self.configuration["environmentQuadrantSizeFactor"])
        quadrantIndex = 0
        # Quadrant I at origin in top left corner, other quadrants in clockwise order
        if 1 in quadrants:
            quadrantOne = [self.environment.grid[i][j] for j in range(quadrantHeight) for i in range(quadrantWidth)]
            cellRange.append(quadrantOne)
            quadrantIndex += 1
        if 2 in quadrants:
            quadrantTwo = [self.environment.grid[i][j] for j in range(quadrantHeight) for i in range(self.environmentWidth - quadrantWidth, self.environmentWidth)]
            cellRange.append(quadrantTwo)
            quadrantIndex += 1
        if 3 in quadrants:
            quadrantThree = [self.environment.grid[i][j] for j in range(self.environmentHeight - quadrantHeight, self.environmentHeight) for i in range(self.environmentWidth - quadrantWidth, self.environmentWidth)]
            cellRange.append(quadrantThree)
            quadrantIndex += 1
        if 4 in quadrants:
            quadrantFour = [self.environment.grid[i][j] for j in range(self.environmentHeight - quadrantHeight, self.environmentHeight) for i in range(quadrantWidth)]
            cellRange.append(quadrantFour)
        return cellRange

    def generateAgentID(self):
        agentID = self.nextAgentID
        self.nextAgentID += 1
        return agentID

    def generateAgentRacialTags(self, numAgents):
        configs = self.configuration
        if configs["agentRacialTagStringLength"] == 0 or configs["environmentMaxRaces"] == 0:
            return [None for i in range(numAgents)]
        numRaces = configs["environmentMaxRaces"]
        racialTagsEndowments = []
        for i in range(numAgents):
            currRace = i % numRaces
            tags = self.generateRacialTags(currRace)
            racialTagsEndowments.append(tags)
        random.shuffle(racialTagsEndowments)
        return racialTagsEndowments

    def generateAgentTags(self, numAgents):
        configs = self.configuration
        if configs["agentTagStringLength"] == 0 or configs["environmentMaxTribes"] == 0 or self.configuration["environmentTribePerQuadrant"] == True:
            return [None for i in range(numAgents)]
        numTribes = configs["environmentMaxTribes"]
        tagsEndowments = []
        for i in range(numAgents):
            currTribe = i % numTribes
            tags = self.generateTribeTags(currTribe)
            tagsEndowments.append(tags)
        random.shuffle(tagsEndowments)
        return tagsEndowments

    def generateDiseaseID(self):
        diseaseID = self.nextDiseaseID
        self.nextDiseaseID += 1
        return diseaseID

    def generateRacialTags(self, race):
        tagStringLength = self.configuration["agentRacialTagStringLength"]
        numRaces = self.configuration["environmentMaxRaces"]
        if numRaces == 1:
            return [race for i in range(tagStringLength)]
        majorityTags = math.floor(tagStringLength / 2) + 1
        assignedRaceTags = random.randint(majorityTags, tagStringLength)
        otherRaceTags = tagStringLength - assignedRaceTags
        otherRaceOptions = [r for r in range(numRaces) if r != race]
        tags = [race for i in range(assignedRaceTags)] + [random.choice(otherRaceOptions) for i in range(otherRaceTags)]
        random.shuffle(tags)
        return tags

    def generateTribeTags(self, tribe):
        tagStringLength = self.configuration["agentTagStringLength"]
        numTribes = self.configuration["environmentMaxTribes"]
        tribeSize = (tagStringLength + 1) / numTribes
        minZeroes = math.floor(tribe * tribeSize)
        maxZeroes = math.floor((tribe + 1) * tribeSize) - 1
        maxZeroes = min(maxZeroes, tagStringLength)
        zeroes = random.randint(minZeroes, maxZeroes)
        ones = tagStringLength - zeroes
        tags = [0 for i in range(zeroes)] + [1 for i in range(ones)]
        random.shuffle(tags)
        return tags

    def isDiseaseExperimentalGroup(self, diseaseID):
        if self.experimentalGroup == None:
            return False
        if "disease" in self.experimentalGroup:
            experimentalDiseaseID = re.search(r"disease(?P<ID>\d+)", self.experimentalGroup).group("ID")
            if int(diseaseID) == int(experimentalDiseaseID):
                return True
        return False

    def pauseSimulation(self):
        while self.run == False:
            if self.gui != None and self.end == False:
                self.gui.window.update()
            if self.end == True:
                self.endSimulation()

    def randomizeAgentEndowments(self, numAgents):
        configs = self.configuration
        aggressionFactor = configs["agentAggressionFactor"]
        baseInterestRate = configs["agentBaseInterestRate"]
        decisionModelAgeismFactor = configs["agentDecisionModelAgeismFactor"]
        decisionModelFactor = configs["agentDecisionModelFactor"]
        decisionModelLookaheadDiscount = configs["agentDecisionModelLookaheadDiscount"]
        decisionModelLookaheadFactor = configs["agentDecisionModelLookaheadFactor"]
        decisionModelRacismFactor = configs["agentDecisionModelRacismFactor"]
        decisionModelSexismFactor = configs["agentDecisionModelSexismFactor"]
        decisionModelTribalFactor = configs["agentDecisionModelTribalFactor"]
        diseaseProtectionChance = configs["agentDiseaseProtectionChance"]
        dynamicSelfishnessFactor = configs["agentDynamicSelfishnessFactor"]
        dynamicSocialPressureFactor = configs["agentDynamicSocialPressureFactor"]
        dynamicDecisionModelFactor = configs["agentDynamicDecisionModelFactor"]
        femaleFertilityAge = configs["agentFemaleFertilityAge"]
        femaleInfertilityAge = configs["agentFemaleInfertilityAge"]
        fertilityFactor = configs["agentFertilityFactor"]
        immuneSystemLength = configs["agentImmuneSystemLength"]
        inheritancePolicy = configs["agentInheritancePolicy"]
        lendingFactor = configs["agentLendingFactor"]
        loanDuration = configs["agentLoanDuration"]
        lookaheadFactor = configs["agentLookaheadFactor"]
        maleFertilityAge = configs["agentMaleFertilityAge"]
        maleInfertilityAge = configs["agentMaleInfertilityAge"]
        maleToFemaleRatio = configs["agentMaleToFemaleRatio"]
        maxAge = configs["agentMaxAge"]
        maxFriends = configs["agentMaxFriends"]
        movement = configs["agentMovement"]
        movementMode = configs["agentMovementMode"]
        neighborhoodMode = configs["neighborhoodMode"]
        selfishnessFactor = configs["agentSelfishnessFactor"]
        spiceMetabolism = configs["agentSpiceMetabolism"]
        startingSpice = configs["agentStartingSpice"]
        startingSugar = configs["agentStartingSugar"]
        sugarMetabolism = configs["agentSugarMetabolism"]
        tagPreferences = configs["agentTagPreferences"]
        tagging = configs["agentTagging"]
        tradeFactor = configs["agentTradeFactor"]
        universalSpice = configs["agentUniversalSpice"]
        universalSugar = configs["agentUniversalSugar"]
        vision = configs["agentVision"]
        visionMode = configs["agentVisionMode"]

        numDepressedAgents = int(math.ceil(numAgents * configs["agentDepressionPercentage"]))
        depressionFactors = [1 for i in range(numDepressedAgents)] + [0 for i in range(numAgents - numDepressedAgents)]
        random.shuffle(depressionFactors)

        configurations = {"aggressionFactor": {"endowments": [], "curr": aggressionFactor[0], "min": aggressionFactor[0], "max": aggressionFactor[1]},
                          "baseInterestRate": {"endowments": [], "curr": baseInterestRate[0], "min": baseInterestRate[0], "max": baseInterestRate[1]},
                          "decisionModelAgeismFactor": {"endowments": [], "curr": decisionModelAgeismFactor[0], "min": decisionModelAgeismFactor[0], "max": decisionModelAgeismFactor[1]},
                          "decisionModelFactor": {"endowments": [], "curr": decisionModelFactor[0], "min": decisionModelFactor[0], "max": decisionModelFactor[1]},
                          "decisionModelLookaheadDiscount": {"endowments": [], "curr": decisionModelLookaheadDiscount[0], "min": decisionModelLookaheadDiscount[0], "max": decisionModelLookaheadDiscount[1]},
                          "decisionModelRacismFactor": {"endowments": [], "curr": decisionModelRacismFactor[0], "min": decisionModelRacismFactor[0], "max": decisionModelRacismFactor[1]},
                          "decisionModelSexismFactor": {"endowments": [], "curr": decisionModelSexismFactor[0], "min": decisionModelSexismFactor[0], "max": decisionModelSexismFactor[1]},
                          "decisionModelTribalFactor": {"endowments": [], "curr": decisionModelTribalFactor[0], "min": decisionModelTribalFactor[0], "max": decisionModelTribalFactor[1]},
                          "diseaseProtectionChance": {"endowments": [], "curr": diseaseProtectionChance[0], "min": diseaseProtectionChance[0], "max": diseaseProtectionChance[1]},
                          "dynamicDecisionModelFactor": {"endowments": [], "curr": dynamicDecisionModelFactor[0], "min": dynamicDecisionModelFactor[0], "max": dynamicDecisionModelFactor[1]},
                          "dynamicSelfishnessFactor": {"endowments": [], "curr": dynamicSelfishnessFactor[0], "min": dynamicSelfishnessFactor[0], "max": dynamicSelfishnessFactor[1]},
                          "dynamicSocialPressureFactor": {"endowments": [], "curr": dynamicSocialPressureFactor[0], "min": dynamicSocialPressureFactor[0], "max": dynamicSocialPressureFactor[1]},
                          "femaleFertilityAge": {"endowments": [], "curr": femaleFertilityAge[0], "min": femaleFertilityAge[0], "max": femaleFertilityAge[1]},
                          "femaleInfertilityAge": {"endowments": [], "curr": femaleInfertilityAge[0], "min": femaleInfertilityAge[0], "max": femaleInfertilityAge[1]},
                          "fertilityFactor": {"endowments": [], "curr": fertilityFactor[0], "min": fertilityFactor[0], "max": fertilityFactor[1]},
                          "lendingFactor": {"endowments": [], "curr": lendingFactor[0], "min": lendingFactor[0], "max": lendingFactor[1]},
                          "loanDuration": {"endowments": [], "curr": loanDuration[0], "min": loanDuration[0], "max": loanDuration[1]},
                          "lookaheadFactor": {"endowments": [], "curr": lookaheadFactor[0], "min": lookaheadFactor[0], "max": lookaheadFactor[1]},
                          "maleFertilityAge": {"endowments": [], "curr": maleFertilityAge[0], "min": maleFertilityAge[0], "max": maleFertilityAge[1]},
                          "maleInfertilityAge": {"endowments": [], "curr": maleInfertilityAge[0], "min": maleInfertilityAge[0], "max": maleInfertilityAge[1]},
                          "maxAge": {"endowments": [], "curr": maxAge[0], "min": maxAge[0], "max": maxAge[1]},
                          "maxFriends": {"endowments": [], "curr": maxFriends[0], "min": maxFriends[0], "max": maxFriends[1]},
                          "movement": {"endowments": [], "curr": movement[0], "min": movement[0], "max": movement[1]},
                          "selfishnessFactor": {"endowments": [], "curr": selfishnessFactor[0], "min": selfishnessFactor[0], "max": selfishnessFactor[1]},
                          "spice": {"endowments": [], "curr": startingSpice[0], "min": startingSpice[0], "max": startingSpice[1]},
                          "spiceMetabolism": {"endowments": [], "curr": spiceMetabolism[0], "min": spiceMetabolism[0], "max": spiceMetabolism[1]},
                          "sugar": {"endowments": [], "curr": startingSugar[0], "min": startingSugar[0], "max": startingSugar[1]},
                          "sugarMetabolism": {"endowments": [], "curr": sugarMetabolism[0], "min": sugarMetabolism[0], "max": sugarMetabolism[1]},
                          "tradeFactor": {"endowments": [], "curr": tradeFactor[0], "min": tradeFactor[0], "max": tradeFactor[1]},
                          "universalSpice": {"endowments": [], "curr": universalSpice[0], "min": universalSpice[0], "max": universalSugar[1]},
                          "universalSugar": {"endowments": [], "curr": universalSugar[0], "min": universalSugar[0], "max": universalSugar[1]},
                          "vision": {"endowments": [], "curr": vision[0], "min": vision[0], "max": vision[1]}
                          }

        if self.agentConfigHashes == None:
            self.agentConfigHashes = {}
            # Map configuration to a random number via hash to make random number generation independent of iteration order
            for config in configurations.keys():
                hashed = hashlib.md5(config.encode())
                hashNum = int(hashed.hexdigest(), 16)
                self.agentConfigHashes[config] = hashNum

        for config in configurations:
            configMin = configurations[config]["min"]
            configMax = configurations[config]["max"]
            configMinDecimals = str(configMin).split('.')
            configMaxDecimals = str(configMax).split('.')
            decimalRange = []
            if len(configMinDecimals) == 2:
                configMinDecimals = len(configMinDecimals[1])
                decimalRange.append(configMinDecimals)
            if len(configMaxDecimals) == 2:
                configMaxDecimals = len(configMaxDecimals[1])
                decimalRange.append(configMaxDecimals)
            # If no fractional component to configuration item, assume increment of 1
            decimals = max(decimalRange) if len(decimalRange) > 0 else 0
            increment = 10 ** (-1 * decimals)
            configurations[config]["inc"] = increment
            configurations[config]["decimals"] = decimals

        if configs["agentDistributionMode"] == "uniform":
            for config in configurations.values():
                config["max"] = config["min"]

        decisionModels = []
        endowments = []
        immuneSystems = []
        racialTags = self.generateAgentRacialTags(numAgents)
        sexes = []
        tags = self.generateAgentTags(numAgents)

        sexDistributionCountdown = numAgents
        # Determine count of male agents and set as switch for agent generation
        if maleToFemaleRatio != None and maleToFemaleRatio != 0:
            sexDistributionCountdown = math.floor(sexDistributionCountdown / (maleToFemaleRatio + 1)) * maleToFemaleRatio

        for i in range(numAgents):
            for config in configurations.values():
                config["endowments"].append(config["curr"])
                config["curr"] += config["inc"]
                config["curr"] = round(config["curr"], config["decimals"])
                if config["curr"] > config["max"]:
                    config["curr"] = config["min"]

            if immuneSystemLength > 0:
                immuneSystems.append([random.randrange(2) for i in range(immuneSystemLength)])
            else:
                immuneSystems.append(None)

            if maleToFemaleRatio != None and maleToFemaleRatio != 0:
                if sexDistributionCountdown == 0:
                    sexes.append("female")
                else:
                    sexes.append("male")
                    sexDistributionCountdown -= 1
            else:
                sexes.append(None)
            decisionModel = configs["agentDecisionModels"][i % len(configs["agentDecisionModels"])]
            # Convert clever name for default behavior
            if decisionModel == "rawSugarscape":
                decisionModel = "none"
            decisionModels.append(decisionModel)

        # Keep state of random numbers to allow extending agent endowments without altering original random object state
        randomNumberReset = random.getstate()
        for config in configurations:
            random.seed(self.agentConfigHashes[config] + self.timestep)
            random.shuffle(configurations[config]["endowments"])
        random.setstate(randomNumberReset)
        random.shuffle(sexes)
        random.shuffle(decisionModels)
        for i in range(numAgents):
            agentEndowment = {"seed": self.seed, "sex": sexes[i], "racialTags": racialTags.pop(), "tags": tags.pop(), "tagPreferences": tagPreferences, "tagging": tagging,
                              "immuneSystem": immuneSystems.pop(), "inheritancePolicy": inheritancePolicy,
                              "decisionModel": decisionModels.pop(), "decisionModelLookaheadFactor": decisionModelLookaheadFactor,
                              "movementMode": movementMode, "neighborhoodMode": neighborhoodMode, "visionMode": visionMode,
                              "depressionFactor": depressionFactors[i], "follower": True}
            for config in configurations:
                # If sexes are enabled, ensure proper fertility and infertility ages are set
                if sexes[i] == "female" and config == "femaleFertilityAge":
                    agentEndowment["fertilityAge"] = configurations["femaleFertilityAge"]["endowments"].pop()
                elif sexes[i] == "female" and config == "femaleInfertilityAge":
                    agentEndowment["infertilityAge"] = configurations["femaleInfertilityAge"]["endowments"].pop()
                elif sexes[i] == "male" and config == "maleFertilityAge":
                    agentEndowment["fertilityAge"] = configurations["maleFertilityAge"]["endowments"].pop()
                elif sexes[i] == "male" and config == "maleInfertilityAge":
                    agentEndowment["infertilityAge"] = configurations["maleInfertilityAge"]["endowments"].pop()
                elif sexes[i] == None and (config == "femaleInfertilityAge" or config == "femaleFertilityAge" or
                                           config == "maleInfertilityAge" or config == "maleFertilityAge"):
                    continue
                else:
                    agentEndowment[config] = configurations[config]["endowments"].pop()

            if sexes[i] == None:
                agentEndowment["fertilityAge"] = 0
                agentEndowment["infertilityAge"] = 0
            endowments.append(agentEndowment)
        return endowments

    def randomizeDiseaseEndowments(self, numDiseases):
        configs = self.configuration
        aggressionPenalty = configs["diseaseAggressionPenalty"]
        fertilityPenalty = configs["diseaseFertilityPenalty"]
        friendlinessPenalty = configs["diseaseFriendlinessPenalty"]
        happinessPenalty = configs["diseaseHappinessPenalty"]
        incubationPeriod = configs["diseaseIncubationPeriod"]
        movementPenalty = configs["diseaseMovementPenalty"]
        spiceMetabolismPenalty = configs["diseaseSpiceMetabolismPenalty"]
        startTimestep = configs["diseaseTimeframe"]
        sugarMetabolismPenalty = configs["diseaseSugarMetabolismPenalty"]
        tagLength = configs["diseaseTagStringLength"]
        transmissionChance = configs["diseaseTransmissionChance"]
        visionPenalty = configs["diseaseVisionPenalty"]

        configurations = {"aggressionPenalty": {"endowments": [], "curr": aggressionPenalty[0], "min": aggressionPenalty[0], "max": aggressionPenalty[1]},
                          "fertilityPenalty": {"endowments": [], "curr": fertilityPenalty[0], "min": fertilityPenalty[0], "max": fertilityPenalty[1]},
                          "friendlinessPenalty": {"endowments": [], "curr": friendlinessPenalty[0], "min": friendlinessPenalty[0], "max": friendlinessPenalty[1]},
                          "happinessPenalty": {"endowments": [], "curr": happinessPenalty[0], "min": happinessPenalty[0], "max": happinessPenalty[1]},
                          "incubationPeriod": {"endowments": [], "curr": incubationPeriod[0], "min": incubationPeriod[0], "max": incubationPeriod[1]},
                          "movementPenalty": {"endowments": [], "curr": movementPenalty[0], "min": movementPenalty[0], "max": movementPenalty[1]},
                          "spiceMetabolismPenalty": {"endowments": [], "curr": spiceMetabolismPenalty[0], "min": spiceMetabolismPenalty[0], "max": spiceMetabolismPenalty[1]},
                          "startTimestep": {"endowments": [], "curr": startTimestep[0], "min": startTimestep[0], "max": startTimestep[1]},
                          "sugarMetabolismPenalty": {"endowments": [], "curr": sugarMetabolismPenalty[0], "min": sugarMetabolismPenalty[0], "max": sugarMetabolismPenalty[1]},
                          "tagLength": {"endowments": [], "curr": tagLength[0], "min": tagLength[0], "max": tagLength[1]},
                          "transmissionChance": {"endowments": [], "curr": transmissionChance[0], "min": transmissionChance[0], "max": transmissionChance[1]},
                          "visionPenalty": {"endowments": [], "curr": visionPenalty[0], "min": visionPenalty[0], "max": visionPenalty[1]}
                          }

        # Map configuration to a random number via hash to make random number generation independent of iteration order
        if (self.diseaseConfigHashes == None):
            self.diseaseConfigHashes = {}
            for penalty in configurations.keys():
                hashed = hashlib.md5(penalty.encode())
                self.diseaseConfigHashes[penalty] = int(hashed.hexdigest(), 16)

        for config in configurations:
            configMin = configurations[config]["min"]
            configMax = configurations[config]["max"]
            configMinDecimals = str(configMin).split('.')
            configMaxDecimals = str(configMax).split('.')
            decimalRange = []
            if len(configMinDecimals) == 2:
                configMinDecimals = len(configMinDecimals[1])
                decimalRange.append(configMinDecimals)
            if len(configMaxDecimals) == 2:
                configMaxDecimals = len(configMaxDecimals[1])
                decimalRange.append(configMaxDecimals)
            # If no fractional component to configuration item, assume increment of 1
            decimals = max(decimalRange) if len(decimalRange) > 0 else 0
            increment = 10 ** (-1 * decimals)
            configurations[config]["inc"] = increment
            configurations[config]["decimals"] = decimals

        endowments = []
        tags = []
        for i in range(numDiseases):
            for config in configurations.keys():
                if config == "tagLength":
                    tagLength = configurations[config]["curr"]
                    tags.append([random.randrange(2) for i in range(tagLength)])
                config = configurations[config]
                config["endowments"].append(config["curr"])
                config["curr"] += config["inc"]
                config["curr"] = round(config["curr"], config["decimals"])
                if config["curr"] > config["max"]:
                    config["curr"] = config["min"]

        # Keep state of random numbers to allow extending agent endowments without altering original random object state
        randomNumberReset = random.getstate()
        for config in configurations:
            random.seed(self.diseaseConfigHashes[config] + self.timestep)
            random.shuffle(configurations[config]["endowments"])
        random.setstate(randomNumberReset)
        random.shuffle(tags)

        for i in range(numDiseases):
            diseaseEndowment = {"tags": tags.pop()}
            for config in configurations:
                    diseaseEndowment[config] = configurations[config]["endowments"].pop()
            endowments.append(diseaseEndowment)
        return endowments

