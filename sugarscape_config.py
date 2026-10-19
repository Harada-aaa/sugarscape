"""Command-line configuration helpers for Sugarscape."""

import getopt
import json
import sys


def parseConfiguration(configFile, configuration):
    file = open(configFile)
    options = json.loads(file.read())
    if "sugarscapeOptions" in options:
        options = options["sugarscapeOptions"]

    optkeys = options.keys()
    if "agentEthicalTheory" in optkeys:
        options["agentDecisionModel"] = options["agentEthicalTheory"]
    if "agentEthicalFactor" in optkeys:
        options["agentDecisionModelFactor"] = options["agentEthicalFactor"]

    for opt in configuration:
        if opt in options:
            configuration[opt] = options[opt]
    return configuration


def parseOptions(configuration):
    commandLineArgs = sys.argv[1:]
    shortOptions = "c:h:"
    longOptions = ["conf=", "help"]
    try:
        args, vals = getopt.getopt(commandLineArgs, shortOptions, longOptions)
    except getopt.GetoptError as err:
        print(err)
        printHelp()
    nextArg = 0
    for currArg, currVal in args:
        nextArg += 1
        if currArg in ("-c", "--conf"):
            if currVal == "":
                print("No config file provided.")
                printHelp()
            parseConfiguration(currVal, configuration)
        elif currArg in ("-h", "--help"):
            printHelp()
    return configuration


def printHelp():
    print("Usage:\n\tpython sugarscape.py --conf config.json\n\nOptions:\n\t-c,--conf\tUse specified config file for simulation settings.\n\t-h,--help\tDisplay this message.")
    exit(0)


def sortConfigurationTimeframes(configuration, timeframe):
    config = configuration[timeframe]
    if configuration != [0, 0]:
        start = config[0]
        end = config[1]
        if start > end and end >= 0:
            swap = start
            start = end
            end = swap
            if "all" in configuration["debugMode"] or "sugarscape" in configuration["debugMode"] or "environment" in configuration["debugMode"]:
                print(f"Start and end values provided for {timeframe} in incorrect order. Switching values around.")
        if start < 0:
            if "all" in configuration["debugMode"] or "sugarscape" in configuration["debugMode"] or "environment" in configuration["debugMode"]:
                print(f"Start timestep {start} for {timeframe} is invalid. Setting {timeframe} start timestep to 0.")
            start = 0
        if end < 0:
            if "all" in configuration["debugMode"] or "sugarscape" in configuration["debugMode"] or "environment" in configuration["debugMode"]:
                print(f"End timestep {end} for {timeframe} is invalid. Setting {timeframe} end timestep to {configuration['timesteps']}.")
            end = configuration["timesteps"]
        config = [start, end]
    return config