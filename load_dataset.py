import pandas as pd
import os
from termcolor import colored

def load_dataset(filename):
    print(colored("Loading dataset...", "green"))
    # Define the columns to be used in the dataset
    dataset_columns = ['Pre-Match PPG (Home)', 'Pre-Match PPG (Away)', 'home_team_goal_count',
                    'away_team_goal_count', 'Home Team Pre-Match xG', 'Away Team Pre-Match xG',
                    'average_goals_per_match_pre_match', 'average_corners_per_match_pre_match',
                    'btts_percentage_pre_match', 'over_15_percentage_pre_match', 'over_25_percentage_pre_match',
                    'over_35_percentage_pre_match', 'over_45_percentage_pre_match', 'points_per_goal_home',
                    'points_per_goal_away', 'over_15_HT_FHG_percentage_pre_match','over_05_HT_FHG_percentage_pre_match',
                    'over_15_2HG_percentage_pre_match','over_05_2HG_percentage_pre_match','label']
    # Load the dataset from a CSV file and select only the desired columns
    match_dataset = pd.read_csv(filename)
    match_dataset_cut = pd.DataFrame(match_dataset, columns=dataset_columns)

    # Label the dataset by calculating some features and setting the target variable
    for index, row in match_dataset_cut.iterrows():
        home_goals = row['home_team_goal_count']
        away_goals = row['away_team_goal_count']
        endscore = home_goals - away_goals
        
        ppg_h = row['Pre-Match PPG (Home)']  # Home team's points per game before the match
        ppg_a = row['Pre-Match PPG (Away)']  # Away team's points per game before the match
        home_goals_total = row['home_team_goal_count']  # Total number of goals scored by the home team before the match
        away_goals_total = row['away_team_goal_count']  # Total number of goals scored by the away team before the match
        
        # Initialize the points_per_goal_home and points_per_goal_away columns to zero
        match_dataset_cut.loc[index, 'points_per_goal_home'] = 0
        match_dataset_cut.loc[index, 'points_per_goal_away'] = 0
        
        # Calculate the points per goal for each team based on their previous performance
        if ppg_h != 0:
            match_dataset_cut.loc[index, 'points_per_goal_home'] = (home_goals_total / ppg_h)  
        if ppg_a != 0:
            match_dataset_cut.loc[index, 'points_per_goal_away'] = (away_goals_total / ppg_a)
            
        # Set the target variable (label) based on the match result
        if endscore > 0:
            match_dataset_cut.loc[index, 'label'] = 1  # Home team wins
        elif endscore < 0:
            match_dataset_cut.loc[index, 'label'] = -1  # Away team wins
        else:
            match_dataset_cut.loc[index, 'label'] = 0  # Draw

    
    # Remove the columns that were used to calculate the features or are not needed for modeling
    match_dataset_cut = match_dataset_cut.drop(['home_team_goal_count', 'away_team_goal_count'], axis=1)
    
    # Convert the label column from float to integer data type
    match_dataset_cut['label'] = match_dataset_cut['label'].astype(int)
    
    # Save the labeled dataset to a CSV file
    match_dataset_cut.to_csv('dataset_labeled.csv')
    
    print(colored("Dataset loaded!","light_green"))
    return match_dataset_cut