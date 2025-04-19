import os
import pandas as pd
import numpy as np
import statsmodels.api as sm

# Directory containing per-match transition matrices (CSV files)
MATRIX_FOLDER = "per_match_matrices"

# List to collect match-level features
match_features = []

# Loop through each CSV matrix file
for file in os.listdir(MATRIX_FOLDER):
    if not file.endswith(".csv"):
        continue

    path = os.path.join(MATRIX_FOLDER, file)
    df = pd.read_csv(path, index_col=0)

    try:
        # Extract all required probabilities safely
        P_goal_given_shoot = df.loc["Shoot Attempt", "Goal"]
        P_cop_given_longpass = df.loc["Long Pass", "Change of Possession"]
        P_cop_given_shortpass = df.loc["Short Pass", "Change of Possession"]
        P_shoot_given_poss = df.loc["Possession", "Shoot Attempt"]
        P_cop_given_defense = df.loc["Defensive Action", "Change of Possession"]
        P_shortpass_given_poss = df.loc["Possession", "Short Pass"]
        P_poss_given_poss = df.loc["Possession", "Possession"]
        P_foul_given_defense = df.loc["Defensive Action", "Foul"]
        P_shortpass_given_shortpass = df.loc["Short Pass", "Short Pass"]
        # P_longpass_given_poss = df.loc["Possession", "Long Pass"]
        # P_longpass_given_longpass = df.loc["Long Pass", "Long Pass"]
        

        # Add row of extracted features
        match_features.append({
            "P_goal_given_shoot": P_goal_given_shoot,
            "retention_long_pass": 1 - P_cop_given_longpass,
            "retention_short_pass": 1 - P_cop_given_shortpass,
            "P_shoot_given_poss": P_shoot_given_poss,
            "P_cop_given_defense": P_cop_given_defense,
            "P_shortpass_given_poss": P_shortpass_given_poss,
            "P_poss_given_poss": P_poss_given_poss,
            "P_defense_non_foul": 1 - P_foul_given_defense,
            "P_shortpass_given_shortpass": P_shortpass_given_shortpass,
            # "P_longpass_given_poss": P_longpass_given_poss,
            # "P_longpass_given_longpass": P_longpass_given_longpass,
        })

    except KeyError as e:
        # Skip files missing any required transition
        print(f"Skipping {file}: missing {e}")
        continue

# Convert to DataFrame
df_reg = pd.DataFrame(match_features)

# Prepare data for regression
X = df_reg[[
    "retention_long_pass",
    "retention_short_pass",
    "P_shoot_given_poss",
    "P_cop_given_defense",
    "P_shortpass_given_poss",
    "P_poss_given_poss",
    "P_defense_non_foul",
    "P_shortpass_given_shortpass",
    # "P_longpass_given_poss",
    # "P_longpass_given_longpass",
]]
X = sm.add_constant(X)  # adds intercept β0
y = df_reg["P_goal_given_shoot"]

# Fit model
model = sm.OLS(y, X).fit()
print(model.summary())
