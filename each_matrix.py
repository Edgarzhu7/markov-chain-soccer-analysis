import os
import json
import pandas as pd
import numpy as np

# Event group map and functions as-is from your logic...
# 👇 (unchanged, include your current event_group_map and build_transition_matrix)

event_group_map = {
    "Pass_Long": "Long Pass",
    "Pass_Short": "Short Pass",
    "Ball Receipt": "Possession",
    "Carry": "Possession",
    "Dispossessed": "Possession",
    "Dribble": "Possession",
    "Dribbled Past": "Possession",
    "Error": "Possession",
    "Foul Won": "Possession",
    "Miscontrol": "Possession",
    "Block": "Defensive Action",
    "Clearance": "Defensive Action",
    "Interception": "Defensive Action",
    "50/50": "Duel",
    "Ball Recovery": "Duel",
    "Duel": "Duel",
    "Shield": "Duel",
    "Shot": "Shoot Attempt",
    "Goal Keeper": "Set Piece / Restart",
    "Half End": "Set Piece / Restart",
    "Half Start": "Set Piece / Restart",
    "Injury Stoppage": "Set Piece / Restart",
    "Player Off": "Set Piece / Restart",
    "Player On": "Set Piece / Restart",
    "Referee Ball-Drop": "Set Piece / Restart",
    "Starting XI": "Set Piece / Restart",
    "Substitution": "Set Piece / Restart",
    "Tactical Shift": "Set Piece / Restart",
    "Bad Behaviour": "Foul",
    "Foul Committed": "Foul",
    "Offside": "Foul",
    "Own Goal For": "Shoot Attempt",
    "Own Goal Against": "Shoot Attempt"
}

def extract_transitions_from(data):
    transitions = []

    for i in range(len(data) - 1):
        cur = data[i]
        nxt = data[i + 1]

        if "type" not in cur or "type" not in nxt:
            continue
        if "possession_team" not in cur or "possession_team" not in nxt:
            continue

        cur_team = cur["possession_team"]["id"]
        nxt_team = nxt["possession_team"]["id"]

        cur_type = cur["type"]["name"]
        nxt_type = nxt["type"]["name"]
        goal = None

        if cur_type == "Pass" and "pass" in cur and "length" in cur["pass"]:
            cur_type = "Pass_Long" if cur["pass"]["length"] >= 25 else "Pass_Short"
        if nxt_type == "Pass" and "pass" in nxt and "length" in nxt["pass"]:
            nxt_type = "Pass_Long" if nxt["pass"]["length"] >= 25 else "Pass_Short"

        if cur_type == "Own Goal Against" or cur_type == "Own Goal For":
            goal = "Goal"
        if cur_type == "Shot" and "shot" in cur:
            outcome = cur["shot"].get("outcome", {}).get("name")
            if outcome == "Goal":
                goal = "Goal"

        cur_group = event_group_map.get(cur_type)
        nxt_group = event_group_map.get(nxt_type)

        if cur_group and nxt_group:
            if goal == "Goal":
                transitions.append((cur_group, "Goal"))
                transitions.append(("Goal", "Set Piece / Restart"))
                transitions.append(("Set Piece / Restart", nxt_group))
            elif cur_team != nxt_team:
                transitions.append((cur_group, "Change of Possession"))
                transitions.append(("Change of Possession", nxt_group))
            else:
                transitions.append((cur_group, nxt_group))

    return transitions

def build_transition_matrix(events, return_counts=False):
    event_types = sorted(set(e[0] for e in events) | set(e[1] for e in events))
    event_index = {event: i for i, event in enumerate(event_types)}
    transition_counts = np.zeros((len(event_types), len(event_types)))

    for cur_group, nxt_group in events:
        i = event_index[cur_group]
        j = event_index[nxt_group]
        transition_counts[i, j] += 1

    row_sums = np.maximum(transition_counts.sum(axis=1, keepdims=True), 1)
    transition_probs = transition_counts / row_sums
    matrix_df = pd.DataFrame(transition_probs, index=event_types, columns=event_types)

    if return_counts:
        counts_df = pd.DataFrame(transition_counts, index=event_types, columns=event_types)
        return matrix_df, counts_df
    else:
        return matrix_df

# NEW: Process each file in folder
def process_matches(folder_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for file in os.listdir(folder_path):
        if file.endswith(".json"):
            full_path = os.path.join(folder_path, file)

            with open(full_path, "r") as f:
                data = json.load(f)

            transitions = extract_transitions_from(data)
            matrix = build_transition_matrix(transitions)

            # Save to CSV
            base_name = os.path.splitext(file)[0]
            output_file = os.path.join(output_folder, f"{base_name}_matrix.csv")
            matrix.to_csv(output_file)
            print(f"Saved: {output_file}")

# === Run it ===
if __name__ == "__main__":
    input_folder = "data"  # ← your match JSONs go here
    output_folder = "per_match_matrices"
    process_matches(input_folder, output_folder)
