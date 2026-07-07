from json.scanner import NUMBER_RE
from pathlib import Path
import re
import sys
import pandas as pd

CONDITIONS = {}
CONDITIONS['3'] = ["certain", "manipulated_1", "manipulated_2", "manipulated_3"]
CONDITIONS['3'] = ["certain", "manipulated_1", "manipulated_2", "manipulated_3", "manipulated_4", "manipulated_5"]
CONDITIONS['not_enough_info'] = ["certain", "not_enough_info"]
CONDITIONS['two_groups'] = ["certain", "partially_manipulated", "fully_manipulated"]


def to_float(value):
    match = NUMBER_RE.search(str(value))
    if match is None:
        raise ValueError(f"Keine Zahl gefunden in: {value!r}")
    return float(match.group(0))

# how fast do the values increase per step (on average over the steps)
def slope(values):
    diffs = [b - a for a, b in zip(values, values[1:])]
    return sum(diffs) / len(diffs)


def load_tables(manipulation_type, model):
    tables = {}
    model = model.replace("/","_")

    for condition in CONDITIONS[manipulation_type]:
        csv_path =   f"../results/Head_Div/{model}/{manipulation_type}/{condition}/head_div.csv"
        table = pd.read_csv(csv_path)
        table = table[table[table.columns[1:]].notna().all(axis=1)]

        for column in table.columns:
            if column != "Layer":
                table[column] = table[column].map(to_float)
        tables[condition] = table

    return tables


def build_layer_head_ranking(tables, manipulation_type):
    heads = [column for column in tables["certain"].columns if column != "Layer"]
    rows = []

    for row_index, layer in enumerate(tables["certain"]["Layer"]):
        for head in heads:
            values = [tables[condition].at[row_index, head] for condition in CONDITIONS[manipulation_type]]
            steps = [values[i + 1] - values[i] for i in range(len(values) - 1)]

            rows.append({
                "Layer": layer,
                "Head": head,
                "certain": values[0],
                "manipulated_1": values[1],
                "manipulated_2": values[2],
                "manipulated_3": values[3],
                "increase_certain_to_manipulated_1": steps[0],
                "increase_manipulated_1_to_manipulated_2": steps[1],
                "increase_manipulated_2_to_manipulated_3": steps[2],
                "total_increase_certain_to_manipulated_3": values[3] - values[0],
                "trend_slope_per_step": slope(values),
                "monotonic_increase": all(step >= 0 for step in steps),
            })

    ranking = pd.DataFrame(rows)
    ranking = ranking.sort_values(
        ["trend_slope_per_step", "total_increase_certain_to_manipulated_3"],
        ascending=False,
    ).reset_index(drop=True)
    ranking.insert(0, "Rank", ranking.index + 1)
    return ranking



def run(manipulation_type, model):
    path = Path(f"../results/Head_Div_Analysis/{model.replace('/','_')}/{manipulation_type}")
    results_dir = Path(path)


    tables = load_tables(manipulation_type, model)

    layer_ranking = build_layer_head_ranking(tables, manipulation_type)

    results_dir.mkdir(parents=True, exist_ok=True)
    layer_ranking.to_csv(results_dir / "head_increase_layer_head_ranking.csv", index=False)    
    best = layer_ranking.iloc[0]
    print(f"Fertig: bester einzelner Head = Layer {best['Layer']}, {best['Head']}")
    print(f"Alle Ergebnisse liegen direkt unter: {results_dir}")


run("not_enough_info","meta-llama/Meta-Llama-3-8B-Instruct")
