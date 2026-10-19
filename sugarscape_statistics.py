"""Statistics and log output methods for Sugarscape."""

import json
import math
import sys

class SugarscapeStatisticsMethods:
    def removeDeadAgents(self):
        deadAgents = []
        for agent in self.agents:
            if agent.isAlive() == False:
                deadAgents.append(agent)
            elif agent.cell == None:
                deadAgents.append(agent)
        self.deadAgents += deadAgents
        for agent in deadAgents:
            agent.doDeath()
            self.agents.remove(agent)

    def replaceDeadAgents(self):
        numAgents = len(self.agents)
        if numAgents < self.configuration["agentReplacements"]:
            numReplacements = self.configuration["agentReplacements"] - numAgents
            self.configureAgents(numReplacements)

    def runSimulation(self, timesteps=5):
        self.maxTimestep = timesteps
        self.updateRuntimeStats()
        if self.resumeTimestep > 0:
            if "all" in self.debug or "sugarscape" in self.debug:
                print(f"Resuming simulation from log at timestep {self.resumeTimestep} (target: {timesteps})...")
            # Deterministic replay phase
            targetSimulationMode = self.configuration.get("simulationMode", "normal")
            self.configuration["simulationMode"] = "normal"
            self.replaying = True
            for _ in range(self.resumeTimestep):
                self.doTimestep()
            self.replaying = False
            self.configuration["simulationMode"] = targetSimulationMode
            if "all" in self.debug or "sugarscape" in self.debug:
                print(f"Replay complete to timestep {self.resumeTimestep}. Continuing in {targetSimulationMode} mode...")
        else:
            if not self.logStarted:
                self.startLog(self.log)
        if self.gui != None:
            # Simulation begins paused until start button in GUI pressed
            self.gui.updateLabels()
            self.pauseSimulation()
        t = 1
        timesteps = timesteps - self.timestep
        screenshots = 0
        while t <= timesteps:
            if len(self.agents) == 0 and self.keepAlive == False:
                break
            if self.configuration["screenshots"] == True and self.configuration["headlessMode"] == False:
                self.gui.canvas.postscript(file=f"screenshot{screenshots}.ps", colormode="color")
                screenshots += 1
            self.doTimestep()
            t += 1
            if self.gui != None and self.run == False:
                self.pauseSimulation()
        if self.configuration["keepAliveAtEnd"] == True and self.gui != None:
            self.run = False
            self.pauseSimulation()
        else:
            self.endSimulation()

    def startLog(self, log):
        if log == None:
            return
        if log == self.log and self.logStarted:
            return
        if log == self.agentLog and self.agentLogStarted:
            return
        stats = sorted(self.runtimeStats)
        if log == self.agentLog:
            stats = self.agentRuntimeStats[0]
            self.agentLogStarted = True
        else:
            self.logStarted = True
        if self.logFormat == "csv":
            header = ""
            # Ensure consistent ordering for CSV format
            for stat in stats:
                if header == "":
                    header += f"{stat}"
                else:
                    header += f",{stat}"
            header += "\n"
            log.write(header)
        else:
            log.write("[\n")
        self.writeToLog(log)

    def toggleEnd(self):
        self.end = True

    def toggleRun(self):
        self.run = not self.run

    def talk(self, agent1, agent2):
        if agent1 == None or agent2 == None:
            return None
        return agent1.talk(agent2)


    def updateGiniCoefficient(self):
        if len(self.agents) == 0:
            return 0
        agentWealths = sorted([agent.sugar + agent.spice for agent in self.agents])
        # Calculate normalized area of Lorenz curve of agent wealths
        numAgents = len(agentWealths)
        totalWealth = sum(agentWealths)
        if totalWealth == 0:
            return 1
        cumulativeWealth = 0
        lorenzCurveArea = 0
        for i in range(numAgents - 1):
            cumulativeWealth += agentWealths[i]
            lorenzCurveArea += cumulativeWealth / totalWealth
        # Use trapezoidal area to maintain accuracy with smaller populations
        cumulativeWealth += agentWealths[-1]
        lorenzCurveArea += (cumulativeWealth / 2) / totalWealth
        lorenzCurveArea /= numAgents
        # The total area under the equality line will be 0.5
        equalityLineArea = 0.5
        giniCoefficient = round((equalityLineArea - lorenzCurveArea) / equalityLineArea, 3)
        return giniCoefficient

    def updateGraphStats(self):
        histogramBins = self.gui.xTicks

        maxAge = self.configuration["agentMaxAge"][1]
        ageBins = [0] * histogramBins
        if maxAge != -1:
            for agent in self.agents:
                if agent == self.agentLeader:
                    continue
                ageBins[math.floor(agent.age / (maxAge + 1) * histogramBins)] += 1

        maxSpice = 0
        maxSugar = 0
        maxWealth = 0
        for agent in self.agents:
            if agent.spice > maxSpice:
                maxSpice = agent.spice
            if agent.sugar > maxSugar:
                maxSugar = agent.sugar
            if agent.sugar + agent.spice > maxWealth:
                maxWealth = agent.sugar + agent.spice
        self.graphStats["maxSpice"] = maxSpice
        self.graphStats["maxSugar"] = maxSugar
        self.graphStats["maxWealth"] = maxWealth

        sugarBins = [0] * histogramBins
        spiceBins = [0] * histogramBins
        agentWealths = []
        for agent in self.agents:
            if agent == self.agentLeader:
                continue
            spiceBins[math.floor(agent.spice / (maxSpice + 1) * histogramBins)] += 1
            sugarBins[math.floor(agent.sugar / (maxSugar + 1) * histogramBins)] += 1
            agentWealths.append(agent.sugar + agent.spice)

        meanTribeTags = [0] * self.configuration["agentTagStringLength"]
        totalPopulation = len(self.agents)
        if self.configuration["agentTagStringLength"] > 0 and totalPopulation > 0:
            for agent in self.agents:
                meanTribeTags = [i + j for i, j in zip(meanTribeTags, agent.tags)]
            meanTribeTags = [round(tag / totalPopulation, 2) * 100 for tag in meanTribeTags]

        agentWealths.sort()
        totalWealth = sum(agentWealths)
        cumulativeWealth = 0
        if totalWealth > 0 and len(agentWealths) > 0:
            lorenzCurvePoints = [(0, 0)]
            for i, wealth in enumerate(agentWealths):
                cumulativePopulation = (i + 1)
                cumulativeWealth += wealth
                lorenzCurvePoints.append((cumulativePopulation / totalPopulation, cumulativeWealth / totalWealth))
            if lorenzCurvePoints[-1] != (1, 1):
                lorenzCurvePoints.append((1, 1))
        else:
            lorenzCurvePoints = [(0, 0), (1, 1)]

        self.graphStats["ageBins"] = ageBins
        self.graphStats["lorenzCurvePoints"] = lorenzCurvePoints
        self.graphStats["meanTribeTags"] = meanTribeTags
        self.graphStats["spiceBins"] = spiceBins
        self.graphStats["sugarBins"] = sugarBins

    def updateRuntimeStats(self):
        # Log separate stats for experimental and control groups
        if self.experimentalGroup != None:
            self.updateRuntimeStatsPerGroup(self.experimentalGroup)
            self.updateRuntimeStatsPerGroup(self.experimentalGroup, notInGroup=True)
        self.updateRuntimeStatsPerGroup()

    def updateRuntimeStatsPerGroup(self, group=None, notInGroup=False):
        meanAge = 0
        meanAgeismFactor = 0
        meanConflictHappiness = 0
        meanFamilyHappiness = 0
        meanHappiness = 0
        meanHealthHappiness = 0
        meanMetabolism = 0
        meanMovement = 0
        meanRacismFactor = 0
        meanSelfishness = 0
        meanSexismFactor = 0
        meanSocialHappiness = 0
        meanSpiceMetabolism = 0
        meanSugarMetabolism = 0
        combinedMetabolism = 0
        meanTradePrice = 0
        meanVision = 0
        meanWealth = 0
        meanWealthHappiness = 0

        loanVolume = 0
        maxRace = 0
        maxRaceSize = 0
        maxTribe = 0
        maxTribeSize = 0
        maxWealth = 0
        minWealth = sys.maxsize
        moveSpace = 0
        numAgents = 0
        numTraders = 0
        numTribes = 0
        sickAgents = 0
        tradeVolume = 0
        carryingCapacityWeight = 0.05
        carryingCapacity = math.ceil((carryingCapacityWeight * len(self.agents)) + ((1 - carryingCapacityWeight) * self.runtimeStats["carryingCapacity"]))
        if self.timestep == 0:
            carryingCapacity = len(self.agents)

        environmentWealthCreated = 0
        environmentWealthTotal = 0
        for i in range(self.environment.width):
            for j in range(self.environment.height):
                environmentWealthCreated += self.environment.grid[i][j].sugarLastProduced + self.environment.grid[i][j].spiceLastProduced
                environmentWealthTotal += self.environment.grid[i][j].sugar + self.environment.grid[i][j].spice
                if self.timestep == 1:
                    environmentWealthCreated += self.environment.grid[i][j].maxSugar + self.environment.grid[i][j].maxSpice

        agentAgingDeaths = 0
        agentCombatDeaths = 0
        agentDiseaseDeaths = 0
        agentLastMoveOptimalityPercentage = 0
        agentMoves = 0
        agentMeanTimeToLive = 0
        agentStarvationDeaths = 0
        agentTotalMetabolism = 0
        agentWealthBurnRate = 0
        agentWealthCollected = 0
        agentWealthTotal = 0

        meanDeathsPercentage = 0
        sickAgentsPercentage = 0

        diseaseEffectiveReproductionRate = 0
        diseaseIncidence = 0
        diseasePrevalence = 0
        infectors = set()

        combatControlToControl = 0
        combatControlToExperimental = 0
        combatExperimentalToControl = 0
        combatExperimentalToExperimental = 0
        diseaseControlToControl = 0
        diseaseControlToExperimental = 0
        diseaseExperimentalToControl = 0
        diseaseExperimentalToExperimental = 0
        lendingControlToControl = 0
        lendingControlToExperimental = 0
        lendingExperimentalToControl = 0
        lendingExperimentalToExperimental = 0
        reproductionControlToControl = 0
        reproductionControlToExperimental = 0
        reproductionExperimentalToControl = 0
        reproductionExperimentalToExperimental = 0
        tradeControlToControl = 0
        tradeControlToExperimental = 0
        tradeExperimentalToControl = 0
        tradeExperimentalToExperimental = 0

        agentsBorn = 0
        agentsReplaced = 0
        remainingRaces = 0
        races = {}
        remainingTribes = 0
        tribes = {}
        
        meanNeighbors = 0
        meanControlNeighbors = 0
        meanExperimentalNeighbors = 0
        meanValidMoves = 0
        meanMoveRank = 0
        meanMoveDifferenceFromOptimal = 0

        for agent in self.agents:
            if group != None and agent.isInGroup(group, notInGroup) == False:
                continue
            agentTimeToLive = agent.findTimeToLive()
            agentTimeToLiveAgeLimited = agent.findTimeToLive(True)
            agentWealth = agent.sugar + agent.spice
            meanSelfishness += agent.selfishnessFactor
            meanSugarMetabolism += agent.sugarMetabolism
            meanSpiceMetabolism += agent.spiceMetabolism
            meanMovement += agent.movement
            meanVision += agent.vision
            meanAge += agent.age
            meanAgeismFactor += agent.decisionModelAgeismFactor
            meanWealth += agentWealth
            meanHappiness += agent.happiness
            meanRacismFactor += agent.decisionModelRacismFactor
            meanSexismFactor += agent.decisionModelSexismFactor
            meanWealthHappiness += agent.wealthHappiness
            meanHealthHappiness += agent.healthHappiness
            meanFamilyHappiness += agent.familyHappiness
            meanSocialHappiness += agent.socialHappiness
            meanConflictHappiness += agent.conflictHappiness
            if agent.lastLendedTimestep == self.timestep:
                loanVolume += agent.lastLoans
            if agent.tradeVolume > 0:
                meanTradePrice += max(agent.spicePrice, agent.sugarPrice)
                tradeVolume += agent.tradeVolume
                numTraders += 1
            agentWealthTotal += agentWealth
            agentWealthCollected += agentWealth - (agent.lastSugar + agent.lastSpice)
            agentWealthBurnRate += agentTimeToLive
            agentMeanTimeToLive += agentTimeToLiveAgeLimited
            agentTotalMetabolism += agent.sugarMetabolism + agent.spiceMetabolism
            if agent.lastMoveOptimal == True:
                agentLastMoveOptimalityPercentage += 1
            agentMoves += 1

            meanNeighbors += len(agent.movementNeighborhood)
            if self.experimentalGroup != None:
                for neighbor in agent.movementNeighborhood:
                    if neighbor.isInGroup(self.experimentalGroup, notInGroup=True):
                        meanControlNeighbors += 1
                    else:
                        meanExperimentalNeighbors += 1
            moveSpace += agent.lastValidMoves
            meanValidMoves += len(agent.validMoves)
            for i in range(len(agent.validMoves)):
                cell = agent.validMoves[i]["cell"]
                if cell == agent.cell:
                    meanMoveDifferenceFromOptimal += agent.validMoves[0]["wealth"] - agent.validMoves[i]["wealth"]
                    meanMoveRank += i
                    break

            if agent.isSick():
                sickAgents += 1
            if agentWealth < minWealth:
                minWealth = agentWealth
            if agentWealth > maxWealth:
                maxWealth = agentWealth
            if agent.race not in races:
                races[agent.race] = 1
            else:
                races[agent.race] += 1
            if agent.tribe not in tribes:
                tribes[agent.tribe] = 1
            else:
                tribes[agent.tribe] += 1
            if group != None and agent.isInGroup(group):
                combatExperimentalToControl += agent.combatWithControlGroup
                combatExperimentalToExperimental += agent.combatWithExperimentalGroup
                diseaseExperimentalToControl += agent.diseaseWithControlGroup
                diseaseExperimentalToExperimental += agent.diseaseWithExperimentalGroup
                lendingExperimentalToControl += agent.lendingWithControlGroup
                lendingExperimentalToExperimental += agent.lendingWithExperimentalGroup
                reproductionExperimentalToControl += agent.reproductionWithControlGroup
                reproductionExperimentalToExperimental += agent.reproductionWithExperimentalGroup
                tradeExperimentalToControl += agent.tradeWithControlGroup
                tradeExperimentalToExperimental += agent.tradeWithExperimentalGroup
                agent.resetTimestepMetrics()
            elif group != None and agent.isInGroup(group, notInGroup=True):
                combatControlToControl += agent.combatWithControlGroup
                combatControlToExperimental += agent.combatWithExperimentalGroup
                diseaseControlToControl += agent.diseaseWithControlGroup
                diseaseControlToExperimental += agent.diseaseWithExperimentalGroup
                lendingControlToControl += agent.lendingWithControlGroup
                lendingControlToExperimental += agent.lendingWithExperimentalGroup
                reproductionControlToControl += agent.reproductionWithControlGroup
                reproductionControlToExperimental += agent.reproductionWithExperimentalGroup
                tradeControlToControl += agent.tradeWithControlGroup
                tradeControlToExperimental += agent.tradeWithExperimentalGroup
                agent.resetTimestepMetrics()
            numAgents += 1

            for disease in agent.diseases:
                # If in the experimental group for a specific disease, skip other diseases
                if group != None and self.isDiseaseExperimentalGroup(disease["disease"].ID) == False and notInGroup == False:
                    continue
                # If in the control group for a specific disease, skip the experimental disease
                elif group != None and self.isDiseaseExperimentalGroup(disease["disease"].ID) == True and notInGroup == True:
                    continue
                if disease["caught"] == self.timestep:
                    diseaseIncidence += 1
                    if self.timestep != 0:
                        infectors.add(disease["infector"])

        numDeadAgents = 0
        meanAgeAtDeath = 0
        for agent in self.deadAgents:
            if group != None and agent.isInGroup(group, notInGroup) == False:
                continue
            # If agent moved this timestep but died, count its movement optimality
            if agent.timestep == self.timestep:
                if agent.lastMoveOptimal == True:
                    agentLastMoveOptimalityPercentage += 1
                agentMoves += 1
            agentWealth = agent.sugar + agent.spice
            meanAgeAtDeath += agent.age
            agentWealthCollected += agentWealth - (agent.lastSugar + agent.lastSpice)
            agentAgingDeaths += 1 if agent.causeOfDeath == "aging" else 0
            agentCombatDeaths += 1 if agent.causeOfDeath == "combat" else 0
            agentDiseaseDeaths += 1 if agent.diseaseDeath == True else 0
            agentStarvationDeaths += 1 if agent.causeOfDeath == "starvation" else 0
            if group != None and agent.isInGroup(group):
                combatExperimentalToControl += agent.combatWithControlGroup
                combatExperimentalToExperimental += agent.combatWithExperimentalGroup
                diseaseExperimentalToControl += agent.diseaseWithControlGroup
                diseaseExperimentalToExperimental += agent.diseaseWithExperimentalGroup
                lendingExperimentalToControl += agent.lendingWithControlGroup
                lendingExperimentalToExperimental += agent.lendingWithExperimentalGroup
                reproductionExperimentalToControl += agent.reproductionWithControlGroup
                reproductionExperimentalToExperimental += agent.reproductionWithExperimentalGroup
                tradeExperimentalToControl += agent.tradeWithControlGroup
                tradeExperimentalToExperimental += agent.tradeWithExperimentalGroup
            elif group != None and agent.isInGroup(group, notInGroup=True):
                combatControlToControl += agent.combatWithControlGroup
                combatControlToExperimental += agent.combatWithExperimentalGroup
                diseaseControlToControl += agent.diseaseWithControlGroup
                diseaseControlToExperimental += agent.diseaseWithExperimentalGroup
                lendingControlToControl += agent.lendingWithControlGroup
                lendingControlToExperimental += agent.lendingWithExperimentalGroup
                reproductionControlToControl += agent.reproductionWithControlGroup
                reproductionControlToExperimental += agent.reproductionWithExperimentalGroup
                tradeControlToControl += agent.tradeWithControlGroup
                tradeControlToExperimental += agent.tradeWithExperimentalGroup
            numDeadAgents += 1

            for disease in agent.diseases:
                # If in the experimental group for a specific disease, skip other diseases
                if group != None and self.isDiseaseExperimentalGroup(disease["disease"].ID) == False and notInGroup == False:
                    continue
                # If in the control group for a specific disease, skip the experimental disease
                elif group != None and self.isDiseaseExperimentalGroup(disease["disease"].ID) == True and notInGroup == True:
                    continue
                if disease["caught"] == self.timestep:
                    diseaseIncidence += 1
                    if self.timestep != 0:
                        infectors.add(disease["infector"])
        meanAgeAtDeath = round(meanAgeAtDeath / numDeadAgents, 2) if numDeadAgents > 0 else 0

        for disease in self.diseases:
            # If in the experimental group for a specific disease, skip other diseases
            if group != None and self.isDiseaseExperimentalGroup(disease.ID) == False and notInGroup == False:
                continue
            # If in the control group for a specific disease, skip the experimental disease
            elif group != None and self.isDiseaseExperimentalGroup(disease.ID) == True and notInGroup == True:
                continue
            # Depression does not have an infection mechanism for transmission
            if disease.ID == "depression":
                continue
            diseasePrevalence += len(disease.infected)

        # Aggregated metrics captured here before dividing by population size
        totalHappiness = meanHappiness

        if numAgents > 0:
            agentMeanTimeToLive = round(agentMeanTimeToLive / numAgents, 2)
            agentWealthBurnRate = round(agentWealthBurnRate / numAgents, 2)
            agentWealthTotal = round(agentWealthTotal, 2)
            loanVolume = round(loanVolume, 2)
            maxRace = max(races, key=races.get)
            maxRaceSize = races[maxRace]
            maxTribe = max(tribes, key=tribes.get)
            maxTribeSize = tribes[maxTribe]
            maxWealth = round(maxWealth, 2)
            meanAge = round(meanAge / numAgents, 2)
            meanAgeismFactor = round(meanAgeismFactor / numAgents, 2)
            meanConflictHappiness = round(meanConflictHappiness / numAgents, 2)
            meanFamilyHappiness = round(meanFamilyHappiness / numAgents, 2)
            meanHappiness = round(meanHappiness / numAgents, 2)
            meanHealthHappiness = round(meanHealthHappiness / numAgents, 2)
            combinedMetabolism = meanSugarMetabolism + meanSpiceMetabolism
            if meanSugarMetabolism > 0 and meanSpiceMetabolism > 0:
                combinedMetabolism = round(combinedMetabolism / 2, 2)
            meanMetabolism = round(combinedMetabolism / numAgents, 2)
            meanMovement = round(meanMovement / numAgents, 2)
            meanRacismFactor = round(meanRacismFactor / numAgents, 2)
            meanSelfishness = round(meanSelfishness / numAgents, 2)
            meanSexismFactor = round(meanSexismFactor / numAgents, 2)
            meanSocialHappiness = round(meanSocialHappiness / numAgents, 2)
            meanTradePrice = round(meanTradePrice / numTraders, 2) if numTraders > 0 else 0
            meanVision = round(meanVision / numAgents, 2)
            meanWealth = round(meanWealth / numAgents, 2)
            meanWealthHappiness = round(meanWealthHappiness / numAgents, 2)
            minWealth = round(minWealth, 2)
            remainingRaces = len(races)
            remainingTribes = len(tribes)
            tradeVolume = round(tradeVolume, 2)
            meanDeathsPercentage = round((numDeadAgents / numAgents) * 100, 2)
            sickAgentsPercentage = round((sickAgents / numAgents) * 100, 2)
            diseaseEffectiveReproductionRate = round(diseaseIncidence / len(infectors), 2) if len(infectors) > 0 else 0
            agentLastMoveOptimalityPercentage = round((agentLastMoveOptimalityPercentage / agentMoves) * 100, 2)
            meanNeighbors = round(meanNeighbors / numAgents, 2)
            meanControlNeighbors = round(meanControlNeighbors / numAgents, 2)
            meanExperimentalNeighbors = round(meanExperimentalNeighbors / numAgents, 2)
            meanValidMoves = round(meanValidMoves / numAgents, 2)
            meanMoveRank = round(meanMoveRank / numAgents, 2)
            meanMoveDifferenceFromOptimal = round(meanMoveDifferenceFromOptimal / meanNeighbors, 2) if meanNeighbors > 0 else 0
        else:
            agentMeanTimeToLive = 0
            agentWealthBurnRate = 0
            maxRace = 0
            maxTribe = 0
            maxWealth = 0
            meanAge = 0
            meanAgeismFactor = 0
            meanConflictHappiness = 0
            meanFamilyHappiness = 0
            meanHappiness = 0
            meanHealthHappiness = 0
            meanMetabolism = 0
            meanMovement = 0
            meanRacismFactor = 0
            meanSelfishness = 0
            meanSexismFactor = 0
            meanSocialHappiness = 0
            meanVision = 0
            meanWealth = 0
            meanWealthHappiness = 0
            minWealth = 0
            remainingRaces = 0
            remainingTribes = 0
            totalHappiness = 0
            tradeVolume = 0
            diseaseEffectiveReproductionRate = 0

        for agent in self.replacedAgents:
            if group != None and agent.isInGroup(group, notInGroup) == False:
                continue
            agentsReplaced += 1

        for agent in self.bornAgents:
            if group != None and agent.isInGroup(group, notInGroup) == False:
                continue
            agentsBorn += 1

        # TODO: make clear whether agent or environment calculation
        runtimeStats = {"agentAgingDeaths": agentAgingDeaths, "agentCombatDeaths": agentCombatDeaths, "agentDeaths": numDeadAgents,
                        "agentDiseaseDeaths": agentDiseaseDeaths, "agentMeanTimeToLive": agentMeanTimeToLive, "agentsBorn": agentsBorn,
                        "agentsReplaced": agentsReplaced, "agentStarvationDeaths": agentStarvationDeaths, "agentTotalMetabolism": agentTotalMetabolism,
                        "agentWealthBurnRate": agentWealthBurnRate, "agentWealthCollected": agentWealthCollected, "agentWealthTotal": agentWealthTotal,
                        "carryingCapacity": carryingCapacity, "largestRace": maxRace, "largestRaceSize": maxRaceSize,
                        "largestTribe": maxTribe, "largestTribeSize": maxTribeSize, "maxWealth": maxWealth,
                        "meanAge": meanAge, "meanAgeAtDeath": meanAgeAtDeath, "meanConflictHappiness": meanConflictHappiness,
                        "meanFamilyHappiness": meanFamilyHappiness, "meanHappiness": meanHappiness, "meanHealthHappiness": meanHealthHappiness,
                        "meanMetabolism": meanMetabolism, "meanMovement": meanMovement, "meanMoveDifferenceFromOptimal": meanMoveDifferenceFromOptimal,
                        "meanMoveRank": meanMoveRank, "meanNeighbors": meanNeighbors, "meanSelfishness": meanSelfishness,
                        "meanSocialHappiness": meanSocialHappiness, "meanTradePrice": meanTradePrice, "meanWealth": meanWealth,
                        "meanWealthHappiness": meanWealthHappiness, "meanValidMoves": meanValidMoves, "meanVision": meanVision, "minWealth": minWealth,
                        "population": numAgents, "sickAgents": sickAgents, "totalHappiness": totalHappiness,
                        "remainingRaces": remainingRaces, 
                        "remainingTribes": remainingTribes,
                        "tradeVolume": tradeVolume, "meanDeathsPercentage": meanDeathsPercentage, "sickAgentsPercentage": sickAgentsPercentage,
                        "diseaseEffectiveReproductionRate": diseaseEffectiveReproductionRate, "diseaseIncidence": diseaseIncidence,
                        "diseasePrevalence": diseasePrevalence, "agentLastMoveOptimalityPercentage": agentLastMoveOptimalityPercentage,
                        "meanAgeismFactor": meanAgeismFactor, "meanRacismFactor": meanRacismFactor, "meanSexismFactor": meanSexismFactor,
                        "loanVolume": loanVolume, "moveSpace": moveSpace
                        }

        controlInteractionStats = {"combatControlGroupToControlGroup": combatControlToControl, "combatControlGroupToExperimentalGroup": combatControlToExperimental,
                                   "diseaseControlGroupToControlGroup": diseaseControlToControl, "diseaseControlGroupToExperimentalGroup": diseaseControlToExperimental,
                                   "lendingControlGroupToControlGroup": lendingControlToControl, "lendingControlGroupToExperimentalGroup": lendingControlToExperimental,
                                   "reproductionControlGroupToControlGroup": reproductionControlToControl, "reproductionControlGroupToExperimentalGroup": reproductionControlToExperimental,
                                   "tradeControlGroupToControlGroup": tradeControlToControl, "tradeControlGroupToExperimentalGroup": tradeControlToExperimental,
                                   }

        experimentalInteractionStats = {"combatExperimentalGroupToControlGroup": combatExperimentalToControl, "combatExperimentalGroupToExperimentalGroup": combatExperimentalToExperimental,
                                        "diseaseExperimentalGroupToControlGroup": diseaseExperimentalToControl, "diseaseExperimentalGroupToExperimentalGroup": diseaseExperimentalToExperimental,
                                        "lendingExperimentalGroupToControlGroup": lendingExperimentalToControl, "lendingExperimentalGroupToExperimentalGroup": lendingExperimentalToExperimental,
                                        "reproductionExperimentalGroupToControlGroup": reproductionExperimentalToControl, "reproductionExperimentalGroupToExperimentalGroup": reproductionExperimentalToExperimental,
                                        "tradeExperimentalGroupToControlGroup": tradeExperimentalToControl, "tradeExperimentalGroupToExperimentalGroup": tradeExperimentalToExperimental
                                        }

        groupMovementStats = {"meanControlNeighbors": meanControlNeighbors, "meanExperimentalNeighbors": meanExperimentalNeighbors}
        if self.experimentalGroup != None:
            runtimeStats.update(groupMovementStats)

        if group == None:
            self.runtimeStats["environmentWealthCreated"] = environmentWealthCreated
            self.runtimeStats["environmentWealthTotal"] = environmentWealthTotal
            self.runtimeStats["giniCoefficient"] = self.updateGiniCoefficient()
            self.runtimeStats["timestep"] = self.timestep
            self.bornAgents = []
            self.replacedAgents = []
            self.deadAgents = []
        else:
            # Convert keys to Pythonic case scheme
            groupString = group if notInGroup == False else "control"
            groupStats = {}
            for key in runtimeStats.keys():
                groupKey = groupString + key[0].upper() + key[1:]
                groupStats[groupKey] = runtimeStats[key]
            runtimeStats = groupStats
            if notInGroup == True:
                runtimeStats.update(controlInteractionStats)
            else:
                runtimeStats.update(experimentalInteractionStats)

        for key in runtimeStats.keys():
            self.runtimeStats[key] = runtimeStats[key]

    def writeToLog(self, log):
        if log == None:
            return
        stats = self.runtimeStats
        if log == self.agentLog:
            stats = self.agentRuntimeStats
            logString = ""
            for agentStats in stats:
                logString += f"\t{json.dumps(agentStats)},\n"
        else:
            logString = f"\t{json.dumps(stats)},\n"
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

    def __str__(self):
        string = f"{str(self.environment)}Seed: {self.seed}\nTimestep: {self.timestep}\nLiving Agents: {len(self.agents)}"
        return string
