"""Command-line entry point for running a Sugarscape simulation."""

import json
import os

from sugarscape import Sugarscape, parseOptions, verifyConfiguration, verifyRandomSeed


def main():
    config_path = os.path.join(os.path.dirname(__file__), "config", "default.json")
    with open(config_path, encoding="utf-8") as config_file:
        configuration = json.load(config_file)["sugarscapeOptions"]
    configuration["logfile"] = None
    configuration["agentLogfile"] = None
    configuration = parseOptions(configuration)
    verifyRandomSeed(configuration)
    configuration = verifyConfiguration(configuration)
    print("=== Simulation configuration ===")
    print("simulationMode: " + configuration["simulationMode"])
    print("agentDistributionMode: " + configuration["agentDistributionMode"])
    if configuration["headlessMode"] == False:
        import gui
    simulation = Sugarscape(configuration)
    if configuration["profileMode"] == True:
        import cProfile
        import tracemalloc
        tracemalloc.start()
        cProfile.run("simulation.runSimulation(configuration[\"timesteps\"])")
        snapshot = tracemalloc.take_snapshot()
        memoryStats = snapshot.statistics("lineno", True)
        for stat in memoryStats[:100]:
            print(stat)
    else:
        simulation.runSimulation(configuration["timesteps"])
    exit(0)


if __name__ == "__main__":
    main()