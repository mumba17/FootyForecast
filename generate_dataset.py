import pandas as pd
import numpy as np
import os

current_folder = os.getcwd()
folder_path = os.path.join(current_folder, 'datasets')

# Get a list of all files in the folder
file_list = os.listdir(folder_path)


filename2 = "dataset_footyf_v1.csv"
# opening the file with w+ mode truncates the file
f = open(filename2, "w+")
f.close()    

def generate_data(filename):
    dataset_columns = ['Wk', 'Home', 'Away', 'points_home', 'points_away', 'home_team_goal_count', 
                       'away_team_goal_count', 'average_goals_per_match_home', 'average_goals_per_match_away', 
                       'points_per_game_home', 'points_per_game_away', 'home_xG_total', 'away_xG_total', 
                       'home_xG_average', 'away_xG_average', 'btts_home_total', 'btts_away_total', 'btts_home_percentage',
                       'btts_away_percentage', 'points_per_goal_home', 'points_per_goal_away', 'goal_difference_home',
                       'goal_difference_away', 'goal_difference_home_per_game', 'goal_difference_away_per_game', 'total_xG_combined','total_xG_average_per_game','label']
    
    old_columns = ['Wk','Home','xG_home','Score','xG_away','Away']
    match_dataset = pd.read_csv(filename, on_bad_lines='skip',usecols=old_columns)
    match_dataset_cut = pd.DataFrame(match_dataset, columns=old_columns)
    
    
    #Adding IDs for every team instead of strings
    #Make ranking list for the specific team storing the corresponding values for the ranking
    team_df = pd.read_csv('team_ids.csv')
    
    ID_team_list = []
    if 'team' in team_df.columns:
        ID_team_list = team_df[['id', 'team']].values.tolist()

    ranking = [[team, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] for _, team in ID_team_list]

    value_to_id = {team: id for id, team in ID_team_list}

    for id, team in enumerate(match_dataset_cut['Home'].unique()):
        if team not in value_to_id:
            new_id = len(value_to_id) + 1
            value_to_id[team] = new_id

        home_id = value_to_id[team]

        # Update the team IDs in the 'Home' and 'Away' columns of match_dataset_cut
        match_dataset_cut['Home'] = match_dataset_cut['Home'].replace(team, home_id)
        match_dataset_cut['Away'] = match_dataset_cut['Away'].replace(team, home_id)

    team_df = pd.DataFrame.from_dict(value_to_id, orient='index', columns=['id'])
    team_df['team'] = team_df.index  # Add the 'team' column with original team names
    team_df.to_csv('team_ids.csv', index=False)
    
    for index, row in match_dataset_cut.iterrows():
        
        goalsHome,_,goalsAway = list(row['Score'])
        goalsHome = int(goalsHome)
        goalsAway = int(goalsAway)
        xgHome = int(row['xG_home'])
        xgAway = int(row['xG_away'])
        gameweek = int(row['Wk'])
        result_match = goalsHome - goalsAway
        
        home_id = row['Home']
        away_id = row['Away']
        
        # Check if home team ID is present in the ranking list
        if home_id not in [team[0] for team in ranking]:
            # If home team ID is not present, add it dynamically to the ranking list
            ranking.append([home_id, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        # Check if away team ID is present in the ranking list
        if away_id not in [team[0] for team in ranking]:
            # If away team ID is not present, add it dynamically to the ranking list
            ranking.append([away_id, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        
        #Init the first week for every stat to be 0 ahead of the game.
        if gameweek == 1:
            match_dataset_cut.loc[index, 'points_home'] = 0 #1
            match_dataset_cut.loc[index, 'points_away'] = 0
            match_dataset_cut.loc[index, 'home_team_goal_count'] = 0 #2
            match_dataset_cut.loc[index, 'away_team_goal_count'] = 0
            match_dataset_cut.loc[index, 'average_goals_per_match_home'] = 0 #3
            match_dataset_cut.loc[index, 'average_goals_per_match_away'] = 0
            match_dataset_cut.loc[index, 'points_per_game_home'] = 0 #4
            match_dataset_cut.loc[index, 'points_per_game_away'] = 0
            match_dataset_cut.loc[index, 'home_xG_total'] = 0 #5
            match_dataset_cut.loc[index, 'away_xG_total'] = 0
            match_dataset_cut.loc[index, 'home_xG_average'] = 0 #6
            match_dataset_cut.loc[index, 'away_xG_average'] = 0
            match_dataset_cut.loc[index, 'btts_home_total'] = 0 #7
            match_dataset_cut.loc[index, 'btts_away_total'] = 0 
            match_dataset_cut.loc[index, 'btts_home_percentage'] = 0 #8  
            match_dataset_cut.loc[index, 'btts_away_percentage'] = 0
            match_dataset_cut.loc[index, 'points_per_goal_home'] = 0 #9
            match_dataset_cut.loc[index, 'points_per_goal_away'] = 0
            match_dataset_cut.loc[index, 'goal_difference_home'] = 0 #10
            match_dataset_cut.loc[index, 'goal_difference_away'] = 0
            match_dataset_cut.loc[index, 'goal_difference_home_per_game'] = 0 #11
            match_dataset_cut.loc[index, 'goal_difference_away_per_game'] = 0
            match_dataset_cut.loc[index, 'total_xG_combined'] = 0 #12
            match_dataset_cut.loc[index, 'total_xG_average_per_game'] = 0 #13
    
        print(home_id)
        if result_match > 0:
            print(home_id, gameweek)
            ranking[home_id][1] += 3
            match_dataset_cut.loc[index, 'label'] = 1
            
        if result_match < 0:
            ranking[away_id][1] += 3
            match_dataset_cut.loc[index, 'label'] = 2
            
        if result_match == 0:
            ranking[home_id][1] += 1
            ranking[away_id][1] += 1
            match_dataset_cut.loc[index, 'label'] = 0
        
        #Amount of goals scored in the season
        ranking[home_id][2] += goalsHome
        ranking[away_id][2] += goalsAway
        
        #Average amount of goals per game
        ranking[home_id][3] = ranking[home_id][2] / gameweek
        ranking[away_id][3] = ranking[away_id][2] / gameweek
        
        #Average amount of points per game
        ranking[home_id][4] = ranking[home_id][1] / gameweek
        ranking[away_id][4] = ranking[away_id][1] / gameweek
        
        #total xG in season
        ranking[home_id][5] += xgHome
        ranking[away_id][5] += xgAway
        
        #Average xG per game during the season
        ranking[home_id][6] = ranking[home_id][5] / gameweek
        ranking[away_id][6] = ranking[away_id][5] / gameweek
        
        #Both teams to score
        if goalsHome > 0 and goalsAway > 0:
            ranking[home_id][7] += 1
            ranking[away_id][7] += 1
            
        #Both teams to score percentage per season
        ranking[home_id][8] = ranking[home_id][7] / gameweek
        ranking[away_id][8] = ranking[away_id][7] / gameweek
        
        #Points per goal
        if ranking[home_id][2] == 0:
            ranking[home_id][9] = 0
        else:
            ranking[home_id][9] = ranking[home_id][1] / ranking[home_id][2]
            
        if ranking[away_id][2] == 0:
            ranking[away_id][9] = 0
        else:
            ranking[away_id][9] = ranking[away_id][1] / ranking[away_id][2]
        
        #Total goal difference over the season
        ranking[home_id][10] += goalsHome - goalsAway
        ranking[away_id][10] += goalsAway - goalsHome 
        
        #Average goal difference per game
        ranking[home_id][11] = ranking[home_id][10] / gameweek
        ranking[away_id][11] = ranking[away_id][10] / gameweek   
        
        ranking[home_id][12] = ranking[home_id][5] + ranking[away_id][5]
        ranking[away_id][12] = ranking[home_id][5] + ranking[away_id][5]
        
        ranking[home_id][13] = (ranking[home_id][5] + ranking[away_id][5]) / gameweek
        ranking[away_id][13] = (ranking[home_id][5] + ranking[away_id][5]) / gameweek
            
        if gameweek > 1:
            match_dataset_cut.loc[index, 'points_home'] = ranking[home_id][1] 
            match_dataset_cut.loc[index, 'points_away'] = ranking[away_id][1] 
            match_dataset_cut.loc[index, 'home_team_goal_count'] = ranking[home_id][2]
            match_dataset_cut.loc[index, 'away_team_goal_count'] = ranking[away_id][2]
            match_dataset_cut.loc[index, 'average_goals_per_match_home'] = ranking[home_id][3]
            match_dataset_cut.loc[index, 'average_goals_per_match_away'] = ranking[away_id][3] 
            match_dataset_cut.loc[index, 'points_per_game_home'] = ranking[home_id][4]
            match_dataset_cut.loc[index, 'points_per_game_away'] = ranking[away_id][4] 
            match_dataset_cut.loc[index, 'home_xG_total'] = ranking[home_id][5]
            match_dataset_cut.loc[index, 'away_xG_total'] = ranking[away_id][5] 
            match_dataset_cut.loc[index, 'home_xG_average'] = ranking[home_id][6]
            match_dataset_cut.loc[index, 'away_xG_average'] = ranking[away_id][6] 
            match_dataset_cut.loc[index, 'btts_home_total'] = ranking[home_id][7]
            match_dataset_cut.loc[index, 'btts_away_total'] = ranking[away_id][7] 
            match_dataset_cut.loc[index, 'btts_home_percentage'] = ranking[home_id][8]  
            match_dataset_cut.loc[index, 'btts_away_percentage'] = ranking[away_id][8] 
            match_dataset_cut.loc[index, 'points_per_goal_home'] = ranking[home_id][9]
            match_dataset_cut.loc[index, 'points_per_goal_away'] = ranking[away_id][9] 
            match_dataset_cut.loc[index, 'goal_difference_home'] = ranking[home_id][10]
            match_dataset_cut.loc[index, 'goal_difference_away'] = ranking[away_id][10] 
            match_dataset_cut.loc[index, 'goal_difference_home_per_game'] = ranking[home_id][11]
            match_dataset_cut.loc[index, 'goal_difference_away_per_game'] = ranking[away_id][11]
            match_dataset_cut.loc[index, 'total_xG_combined'] = ranking[home_id][12]
            match_dataset_cut.loc[index, 'total_xG_average_per_game'] = ranking[home_id][13]
    
    
    match_dataset_cut = match_dataset_cut.drop(['xG_home', 'Score', 'xG_away'], axis=1)  
    match_dataset_cut['label'] = match_dataset_cut['label'].astype(int)  
    print(match_dataset_cut.tail())
    match_dataset_cut.to_csv(filename2, index=False, mode='a', header=False)
            
# Iterate over each file in the folder
for file_name in file_list:
    # Check if the file has the .csv extension
    if file_name.endswith('.csv'):
        # Construct the full file path
        file_path = os.path.join(folder_path, file_name)
        
        # Call the generate_data function with the file path
        generate_data(file_path)