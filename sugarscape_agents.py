"""Agent, disease, and environment operations for Sugarscape."""

import agent
import cell
import condition
import copy
import ethics
import json
import math
import random

class SugarscapeAgentMethods:
    def __deepcopy__(self, memo):
        selfClass = self.__class__
        result = selfClass.__new__(selfClass)
        memo[id(self)] = result
        skips = ["gui", "log", "agentLog", "agentLeader"]
        for key,value in self.__dict__.items():
            if key not in skips:
                setattr(result, key, copy.deepcopy(value, memo))
            else:
                setattr(result, key, None)
        return result

    def addAgent(self, agent):
        self.bornAgents.append(agent)
        self.agents.append(agent)

    def addRemainingDiseases(self):
        infectedDiseases = []
        for disease in self.remainingDiseases:
            if disease.startTimestep <= self.timestep:
                for agent in self.agents:
                    hammingDistance = agent.findNearestHammingDistanceInDisease(disease)["distance"]
                    if hammingDistance == 0:
                        continue
                    agent.catchDisease(disease, initial=True)
                    self.diseases.append(disease)
                    infectedDiseases.append(disease)
                    break
        for disease in infectedDiseases:
            self.remainingDiseases.remove(disease)

    def addResourcePeak(self, startX, startY, radius, maxValue, resource):
        height = self.environment.height
        width = self.environment.width
        radialDispersion = math.sqrt(max(startX, width - startX)**2 + max(startY, height - startY)**2) * (radius / width)
        for i in range(width):
            for j in range(height):
                euclideanDistanceToStart = math.sqrt((startX - i)**2 + (startY - j)**2)
                currDispersion = 1 + maxValue * (1 - euclideanDistanceToStart / radialDispersion)
                cellMaxCapacity = min(currDispersion, maxValue)
                cellMaxCapacity = math.ceil(cellMaxCapacity)
                if resource == "spice" and cellMaxCapacity > self.environment.findCell(i, j).maxSpice:
                    self.environment.findCell(i, j).maxSpice = cellMaxCapacity
                    self.environment.findCell(i, j).spice = cellMaxCapacity
                elif resource == "sugar" and cellMaxCapacity > self.environment.findCell(i, j).maxSugar:
                    self.environment.findCell(i, j).maxSugar = cellMaxCapacity
                    self.environment.findCell(i, j).sugar = cellMaxCapacity

    def configureAgents(self, numAgents, editCell=None):
        if self.environment == None:
            return
        if editCell != None and editCell.isOccupied():
            return
        if editCell != None and numAgents != 1:
            numAgents = 1

        if self.configuration["agentLeader"] == True and self.agentLeader == None:
            numAgents += 1

        emptyCells = [[cell for cell in quadrant if cell.agent == None] for quadrant in self.activeQuadrants]
        totalCells = sum(len(quadrant) for quadrant in emptyCells)
        quadrants = len(emptyCells)
        if totalCells == 0:
            return
        if numAgents > totalCells:
            if "all" in self.debug or "sugarscape" in self.debug:
                print(f"Could not allocate {numAgents} agents. Allocating maximum of {totalCells}.")
            numAgents = totalCells

        # Ensure agent endowments are randomized across initial agent count to make replacements follow same distributions
        if len(self.agentEndowments) == 0:
            self.agentEndowments = self.randomizeAgentEndowments(numAgents)
        quadrantIndices = []
        if editCell == None:
            for quadrant in emptyCells:
                random.shuffle(quadrant)
            quadrantIndices = [i for i in range(quadrants)]
            random.shuffle(quadrantIndices)

        for i in range(numAgents):
            placementCell = editCell
            if editCell == None:
                quadrantIndex = quadrantIndices[i % quadrants]
                placementCell = emptyCells[quadrantIndex].pop()
            agentConfiguration = self.agentEndowments[self.agentEndowmentIndex % len(self.agentEndowments)]
            self.agentEndowmentIndex += 1
            agentID = self.generateAgentID()
            a = agent.Agent(agentID, self.timestep, placementCell, agentConfiguration)
            if self.configuration["agentLeader"] == True and self.agentLeader == None:
                a = ethics.Leader(agentID, self.timestep, placementCell, agentConfiguration)
                a.gotoCell(self.environment.dummyCell)
                self.agentLeader = a
                self.environment.dummyCell.agent = a
                a.findCellsInRange()
                continue

            # If using a different decision model, replace new agent with instance of child class
            if "altruist" in agentConfiguration["decisionModel"]:
                a = ethics.Bentham(agentID, self.timestep, placementCell, agentConfiguration)
                a.selfishnessFactor = 0
            elif "bentham" in agentConfiguration["decisionModel"]:
                a = ethics.Bentham(agentID, self.timestep, placementCell, agentConfiguration)
                if agentConfiguration["selfishnessFactor"] < 0:
                    a.selfishnessFactor = 0.5
            elif "egoist" in agentConfiguration["decisionModel"]:
                a = ethics.Bentham(agentID, self.timestep, placementCell, agentConfiguration)
                a.selfishnessFactor = 1
            elif "negativeBentham" in agentConfiguration["decisionModel"]:
                a = ethics.Bentham(agentID, self.timestep, placementCell, agentConfiguration)
                a.selfishnessFactor = -1
            elif "temperance" in agentConfiguration["decisionModel"]:
                a = ethics.Temperance(agentID, self.timestep, placementCell, agentConfiguration)

            # If dynamic selfishness is desired but not defined, give a small degree of dynamic selfishness
            if "Dynamic" in agentConfiguration["decisionModel"] and self.configuration["agentDynamicSelfishnessFactor"] == [0.0, 0.0]:
                a.dynamicSelfishnessFactor = 0.01
            if "NoLookahead" in agentConfiguration["decisionModel"]:
                a.decisionModelLookaheadFactor = 0
            elif "HalfLookahead" in agentConfiguration["decisionModel"]:
                a.decisionModelLookaheadFactor = 0.5

            # If using a deontological decision model, replace new agent with instance of child class
            if "asimov" in agentConfiguration["decisionModel"]:
                a = ethics.Asimov(agentID, self.timestep, placementCell, agentConfiguration)

            # If using a virtue ethics decision model, replace new agent with instance of child class
            if "temperance" in agentConfiguration["decisionModel"]:
                if "PECS" in agentConfiguration["decisionModel"]:
                    a = ethics.Temperance(agentID, self.timestep, placementCell, agentConfiguration, pecs=True)
                else:
                    a = ethics.Temperance(agentID, self.timestep, placementCell, agentConfiguration, pecs=False)

            if self.configuration["environmentTribePerQuadrant"] == True:
                tribe = quadrantIndex
                tags = self.generateTribeTags(tribe)
                a.tags = tags
                a.tribe = a.findTribe()
            placementCell.agent = a
            self.agents.append(a)
            if self.timestep > 0:
                self.replacedAgents.append(a)

        for a in self.agents:
            a.findCellsInRange()
            a.findNeighborhood()

    def configureCell(self, cell, mode, delta):
        if cell == None or delta == 0:
            return
        if mode == "currentSpice":
            cell.spice = min(max(0, (cell.spice + delta)), self.environment.globalMaxSpice)
        elif mode == "currentSugar":
            cell.sugar = min(max(0, (cell.sugar + delta)), self.environment.globalMaxSugar)
        elif mode == "maximumSpice":
            cell.maxSpice = min(max(0, (cell.maxSpice + delta)), self.environment.globalMaxSpice)
        elif mode == "maximumSugar":
            cell.maxSugar = min(max(0, (cell.maxSugar + delta)), self.environment.globalMaxSugar)

    def configureDepression(self):
        if self.depression == True:
            self.depression = condition.Depression()
            self.diseases.append(self.depression)

    def configureDiseases(self, numDiseases, namedDiseases, editCell=None):
        numAgents = len(self.agents)
        if numAgents == 0:
            return
        elif numAgents < numDiseases:
            numDiseases = numAgents

        if len(self.diseaseEndowments) == 0:
            premadeDiseases  = []
            for disease in namedDiseases:
                validDisease = 0
                if "zombieVirus" in disease:
                    zombie = condition.ZombieVirus("zombieVirus", {})
                    premadeDiseases.append(zombie)
                    validDisease = 1
                if validDisease == 1:
                    numDiseases -= 1

            self.diseaseEndowments = self.randomizeDiseaseEndowments(numDiseases)
            random.shuffle(self.agents)
            initialDiseases = []
            for i in range(numDiseases):
                diseaseID = self.generateDiseaseID()
                diseaseConfiguration = self.diseaseEndowments[i]
                newDisease = condition.Disease(diseaseID, diseaseConfiguration)
                if diseaseConfiguration["startTimestep"] == 0:
                    initialDiseases.append(newDisease)
                else:
                    self.remainingDiseases.append(newDisease)
            initialDiseases.extend(premadeDiseases)

            startingDiseases = self.configuration["startingDiseasesPerAgent"]
            minStartingDiseases = startingDiseases[0]
            maxStartingDiseases = startingDiseases[1]
            currStartingDiseases = minStartingDiseases
            for agent in self.agents:
                random.shuffle(initialDiseases)
                for newDisease in initialDiseases:
                    if len(agent.diseases) >= currStartingDiseases and startingDiseases != [0, 0]:
                        currStartingDiseases += 1
                        break
                    if newDisease.tags != None:
                        hammingDistance = agent.findNearestHammingDistanceInDisease(newDisease)["distance"]
                        if hammingDistance == 0:
                            continue
                    agent.catchDisease(newDisease, initial=True)
                    self.diseases.append(newDisease)
                    if startingDiseases == [0, 0]:
                        initialDiseases.remove(newDisease)
                        break
                if currStartingDiseases > maxStartingDiseases:
                    currStartingDiseases = minStartingDiseases

            if startingDiseases == [0, 0] and len(initialDiseases) > 0 and ("all" in self.debug or "sugarscape" in self.debug):
                print(f"Could not place {len(self.diseases)} diseases.")
        elif editCell != None and editCell.isOccupied() == True:
            selectedAgent = editCell.agent
            diseaseID = self.generateDiseaseID()
            newDisease = condition.Disease(diseaseID, self.diseaseEndowments[self.diseaseEndowmentIndex % len(self.diseaseEndowments)])
            self.diseaseEndowmentIndex += 1
            selectedAgent.catchDisease(newDisease)

    def configureEnvironment(self, maxSugar, maxSpice, sugarPeaks, spicePeaks, environmentFile=None):
        height = self.environment.height
        width = self.environment.width
        # Dummy cell for debugging and for leader agent
        dummyCell = cell.Cell(-1, -1, self.environment)
        self.environment.dummyCell = dummyCell

        if environmentFile == None:
            for i in range(width):
                for j in range(height):
                    newCell = cell.Cell(i, j, self.environment)
                    self.environment.setCell(newCell, i, j)

            sugarRadiusScale = 2
            radius = math.ceil(math.sqrt(sugarRadiusScale * (height + width)))
            for peak in sugarPeaks:
                self.addResourcePeak(peak[0], peak[1], radius, peak[2], "sugar")

            spiceRadiusScale = 2
            radius = math.ceil(math.sqrt(spiceRadiusScale * (height + width)))
            for peak in spicePeaks:
                self.addResourcePeak(peak[0], peak[1], radius, peak[2], "spice")
        else:
            environmentFile = open(environmentFile)
            loadEnvironment = json.loads(environmentFile.read())
            environmentFile.close()
            height = len(loadEnvironment)
            width = len(loadEnvironment[0])
            self.environment.height = height
            self.environment.width = width
            self.environmentHeight = height
            self.environmentWidth = width
            for i in range(width):
                for j in range(height):
                    loadSpice = loadEnvironment[i][j]["spice"]
                    loadSugar = loadEnvironment[i][j]["sugar"]
                    newCell = cell.Cell(i, j, self.environment, loadSpice, loadSugar)
                    self.environment.setCell(newCell, i, j)
        self.environment.findCellNeighbors()
        self.environment.findCellRanges()

