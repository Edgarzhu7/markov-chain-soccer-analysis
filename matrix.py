import os
import json
import pandas as pd
import numpy as np

# Load JSON files
def load_json_folder(folder_path):
    data = []
    for file in os.listdir(folder_path):
        if file.endswith(".json"):
            with open(os.path.join(folder_path, file), "r") as f:
                data.extend(json.load(f))
    return data

# Categorize event location into Defensive, Midfield, or Attacking Third
def categorize_zone(x):
    if x < 40:
        return "Defensive"
    elif x < 80:
        return "Midfield"
    else:
        return "Attacking"

# # Map each event to a group
# event_group_map = {
#     # Possession
#     "Pass": "Possession",
#     "Ball Receipt": "Possession",
#     "Carry": "Possession",
#     "Clearance": "Possession",
#     "Ball Recovery": "Possession",

#     # Duel
#     "Duel": "Duel",
#     "50/50": "Duel",
#     "Shield": "Duel",
#     "Dispossessed": "Ball Loss",
#     "Dribbled Past": "Duel",

#     # Defensive Action
#     "Interception": "Defensive Action",
#     "Block": "Defensive Action",
#     "Pressure": "Defensive Action",
#     "Miscontrol": "Ball Loss",
#     "Error": "Defensive Action",

#     # Shot Event
#     "Shot": "Shot Event",
#     "Own Goal For": "Shot Event",
#     "Own Goal Against": "Shot Event",

#     # Foul
#     "Foul Committed": "Foul",
#     "Foul Won": "Foul",

#     # Set Piece / Restart
#     "Goal Keeper": "Set Piece/Restart",
#     "Kick Off": "Set Piece/Restart",
#     "Free Kick": "Set Piece/Restart",
#     "Throw-in": "Set Piece/Restart",
#     "Corner": "Set Piece/Restart",
#     "Referee Ball-Drop": "Set Piece/Restart",

#     # Game State Change
#     "Half Start": "Other",
#     "Half End": "Other",
#     "Substitution": "Other",
#     "Starting XI": "Other",
#     "Player On": "Other",
#     "Player Off": "Other",
#     "Tactical Shift": "Other",
#     "Injury Stoppage": "Other",

#     # Offside
#     "Offside": "Ball Loss",

#     # Other
#     "Bad Behaviour": "Other",
#     "Camera On": "Other",
#     "Camera off": "Other"
# }
event_group_map = {
    # Long vs Short Pass
    "Pass_Long": "Long Pass",
    "Pass_Short": "Short Pass",

    # Possession-related actions
    "Ball Receipt": "Possession",
    "Carry": "Possession",
    "Dispossessed": "Possession",
    "Dribble": "Possession",
    "Dribbled Past": "Possession",
    "Error": "Possession",
    "Foul Won": "Possession",
    "Miscontrol": "Possession",

    # Defensive Actions
    "Block": "Defensive Action",
    "Clearance": "Defensive Action",
    "Interception": "Defensive Action",

    # Duels
    "50/50": "Duel",
    "Ball Recovery": "Duel",
    "Duel": "Duel",
    "Shield": "Duel",

    # Shot Attempts
    "Shot": "Shoot Attempt",

    # Set Piece / Restart
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

    # Fouls and related infractions
    "Bad Behaviour": "Foul",
    "Foul Committed": "Foul",
    "Offside": "Foul",

    # Goals and Concessions
    "Own Goal For": "Shoot Attempt",
    "Own Goal Against": "Shoot Attempt"
}

def extract_transitions_from(data):
    transitions = []

    for i in range(len(data) - 1):
        cur = data[i]
        nxt = data[i + 1]

        # Skip if required keys are missing
        if "type" not in cur or "type" not in nxt:
            continue
        if "possession_team" not in cur or "possession_team" not in nxt:
            continue

        cur_team = cur["possession_team"]["id"]
        nxt_team = nxt["possession_team"]["id"]

        cur_type = cur["type"]["name"]
        nxt_type = nxt["type"]["name"]
        goal = None

        # Handle Pass classification
        if cur_type == "Pass" and "pass" in cur and "length" in cur["pass"]:
            cur_type = "Pass_Long" if cur["pass"]["length"] >= 25 else "Pass_Short"

        # Handle Own Goal Against
        if cur_type == "Own Goal Against" or cur_type == "Own Goal For":
            goal = "Goal"

        # Determine next event group
        if nxt_type == "Pass" and "pass" in nxt and "length" in nxt["pass"]:
            nxt_type = "Pass_Long" if nxt["pass"]["length"] >= 25 else "Pass_Short"

        
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
                # if cur_team != nxt_team:
                #     transitions.append(("Goal", "Change of Possession"))
                #     transitions.append(("Change of Possession", nxt_group))
                # else:
                #     transitions.append(("Goal", nxt_group))

            elif cur_team != nxt_team:
                # Possession change: insert Ball Loss in between
                transitions.append((cur_group, "Change of Possession"))
                transitions.append(("Change of Possession", nxt_group))
            else:
                transitions.append((cur_group, nxt_group))

    return transitions


# # Construct transition matrix
# def build_transition_matrix(events):
#     event_types = sorted(set(e["current_event"] for e in events) | set(e["next_event"] for e in events))

#     # Initialize transition matrix
#     transition_counts = np.zeros((len(event_types), len(event_types)))
    
#     # Map event types to matrix indices
#     event_index = {event: i for i, event in enumerate(event_types)}

#     for e in events:
#         row = event_index[e["current_event"]]
#         col = event_index[e["next_event"]]
#         transition_counts[row, col] += 1

#     # Normalize row-wise to get probabilities
#     transition_probs = transition_counts / np.maximum(transition_counts.sum(axis=1, keepdims=True), 1)

#     # Convert to Pandas DataFrame
#     transition_matrix = pd.DataFrame(transition_probs, index=event_types, columns=event_types)
    
#     return transition_matrix
def build_transition_matrix(events, return_counts=False):
    # Extract unique event group names from tuple pairs
    event_types = sorted(set(e[0] for e in events) | set(e[1] for e in events))

    # Create index map
    event_index = {event: i for i, event in enumerate(event_types)}

    # Initialize count matrix
    transition_counts = np.zeros((len(event_types), len(event_types)))

    # Count transitions
    for cur_group, nxt_group in events:
        i = event_index[cur_group]
        j = event_index[nxt_group]
        transition_counts[i, j] += 1

    # Normalize rows to get probabilities
    row_sums = np.maximum(transition_counts.sum(axis=1, keepdims=True), 1)
    transition_probs = transition_counts / row_sums

    # Return as DataFrame
    matrix_df = pd.DataFrame(transition_probs, index=event_types, columns=event_types)

    if return_counts:
        counts_df = pd.DataFrame(transition_counts, index=event_types, columns=event_types)
        return matrix_df, counts_df
    else:
        return matrix_df

# Load data and process
folder_path = "data"
data = load_json_folder(folder_path)
events = extract_transitions_from(data)
transition_matrix = build_transition_matrix(events)

# Save the transition matrix
file_name = "transition_matrix_zones.csv"
transition_matrix.to_csv(file_name, index=True)
print(f"Saved: {file_name}")
