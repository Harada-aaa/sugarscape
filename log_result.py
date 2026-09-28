"""Create social-indicator graphs from Sugarscape JSON logs.

Usage:
	python log_result.py
	python log_result.py --input logs/default.json --output assets
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


INDICATOR_GROUPS = (
	(
		"Total wellbeing",
		(("totalHappiness", "Total happiness", "#d95f02"),),
	),
	(
		"Happiness components",
		(
			("meanHappiness", "Mean happiness", "#1b9e77"),
			("meanWealthHappiness", "Wealth happiness", "#7570b3"),
			("meanHealthHappiness", "Health happiness", "#e7298a"),
			("meanSocialHappiness", "Social happiness", "#66a61e"),
			("meanFamilyHappiness", "Family happiness", "#e6ab02"),
			("meanConflictHappiness", "Conflict happiness", "#a6761d"),
		),
	),
	(
		"Equality and wealth",
		(
			("giniCoefficient", "Gini coefficient", "#d95f02"),
			("meanWealth", "Mean wealth", "#1b9e77"),
			("agentWealthTotal", "Total agent wealth", "#7570b3"),
			("maxWealth", "Maximum wealth", "#e7298a"),
			("minWealth", "Minimum wealth", "#66a61e"),
		),
	),
	(
		"Population and social conditions",
		(
			("population", "Population", "#1b9e77"),
			("agentsBorn", "Agents born", "#7570b3"),
			("agentDeaths", "Agent deaths", "#d95f02"),
			("sickAgentsPercentage", "Sick agents (%)", "#e7298a"),
			("meanNeighbors", "Mean neighbors", "#66a61e"),
			("loanVolume", "Loan volume", "#e6ab02"),
		),
	),
)


def load_records(path: Path) -> list[dict]:
	"""Load a log containing a JSON list of timestep records."""
	with path.open(encoding="utf-8") as log_file:
		data = json.load(log_file)
	if isinstance(data, dict):
		data = data.get("records", data.get("log", []))
	if not isinstance(data, list):
		raise ValueError("the JSON root must be a list of timestep records")
	return [record for record in data if isinstance(record, dict)]


def numeric_series(records: list[dict], key: str) -> tuple[list[float], list[float]]:
	"""Return only numeric (timestep, value) pairs for one indicator."""
	points = []
	for record in records:
		timestep = record.get("timestep")
		value = record.get(key)
		if isinstance(timestep, (int, float)) and isinstance(value, (int, float)):
			points.append((float(timestep), float(value)))
	if not points:
		return [], []
	return [point[0] for point in points], [point[1] for point in points]


def plot_log(path: Path, output_dir: Path) -> Path:
	records = load_records(path)
	if not records:
		raise ValueError("no timestep records found")

	figure, axes = plt.subplots(len(INDICATOR_GROUPS), 1, figsize=(14, 19), sharex=True)
	figure.suptitle(f"Sugarscape social indicators: {path.stem}", fontsize=18, fontweight="bold")

	for axis, (group_name, indicators) in zip(axes, INDICATOR_GROUPS):
		plotted = 0
		for key, label, color in indicators:
			x_values, y_values = numeric_series(records, key)
			if not x_values:
				continue
			axis.plot(x_values, y_values, label=label, color=color, linewidth=2)
			plotted += 1
		axis.set_title(group_name, loc="left", fontweight="bold")
		axis.set_ylabel("Value")
		axis.grid(True, alpha=0.25)
		if plotted:
			axis.legend(loc="upper left", ncol=3, frameon=False, fontsize=9)
		else:
			axis.text(0.5, 0.5, "No matching indicators", ha="center", va="center", transform=axis.transAxes)

	axes[-1].set_xlabel("Timestep")
	figure.tight_layout(rect=(0, 0, 1, 0.97))
	output_dir.mkdir(parents=True, exist_ok=True)
	output_path = output_dir / f"{path.stem}_social_indicators.png"
	figure.savefig(output_path, dpi=160, bbox_inches="tight")
	plt.close(figure)
	return output_path


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--input",
		type=Path,
		default=Path("logs"),
		help="a JSON log file or a directory containing JSON logs (default: logs)",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=Path("assets"),
		help="directory for generated PNG files (default: assets)",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	input_paths = [args.input] if args.input.is_file() else sorted(args.input.glob("*.json"))
	if not input_paths:
		raise SystemExit(f"No JSON logs found in {args.input}")

	for input_path in input_paths:
		try:
			output_path = plot_log(input_path, args.output)
		except (OSError, ValueError, json.JSONDecodeError) as error:
			print(f"Skipping {input_path}: {error}")
			continue
		print(f"Saved {output_path}")


if __name__ == "__main__":
	main()
