import os
import pandas as pd
import numpy as np
import statsmodels.api as sm

from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt


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
        # P_shortpass_given_poss = df.loc["Possession", "Short Pass"]
        P_poss_given_poss = df.loc["Possession", "Possession"]
        # P_foul_given_defense = df.loc["Defensive Action", "Foul"]
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
            # "P_shortpass_given_poss": P_shortpass_given_poss,
            "P_poss_given_poss": P_poss_given_poss,
            # "P_defense_non_foul": 1 - P_foul_given_defense,
            "P_shortpass_given_shortpass": P_shortpass_given_shortpass,
            # "P_longpass_given_poss": P_longpass_given_poss,
            # "P_longpass_given_longpass": P_longpass_given_longpass,
        })

    except KeyError as e:
        # Skip files missing any required transition
        print(f"Skipping {file}: missing {e}")
        continue


# Load your match_features like before
df = pd.DataFrame(match_features)

# Define predictors and target
X = df[[
    "retention_long_pass",
    "retention_short_pass",
    "P_shoot_given_poss",
    "P_cop_given_defense",
    # "P_shortpass_given_poss",
    "P_poss_given_poss",
    # "P_defense_non_foul",
    "P_shortpass_given_shortpass",
    # "P_longpass_given_poss",
    # "P_longpass_given_longpass",
]]
y = df["P_goal_given_shoot"]

# Optional: standardize + LASSO pipeline with built-in cross-validation
model = make_pipeline(StandardScaler(), LassoCV(cv=5, random_state=42))
model.fit(X, y)

# Get coefficients
lasso = model.named_steps['lassocv']
coef = pd.Series(lasso.coef_, index=X.columns)
print("\n📉 LASSO Coefficients:")
print(coef)

# Plot coefficients
coef.plot(kind="barh")
plt.axvline(0, color='k', linestyle='--')
plt.title("LASSO Regression Coefficients")
plt.tight_layout()
plt.show()

# Print model performance
print(f"\nR^2 on training set: {model.score(X, y):.3f}")
print(f"Best alpha (λ): {lasso.alpha_:.4f}")
