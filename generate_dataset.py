import pandas as pd
import numpy as np
import os

current_folder = os.getcwd()
folder_path = os.path.join(current_folder, 'datasets')

# Get a list of all files in the folder
file_list = os.listdir(folder_path)


filename2 = "dataset_footyf_v5.csv"
# opening the file with w+ mode truncates the file
f = open(filename2, "w+")
f.close()    

def generate_data(filename):
    dataset_columns = ['Wk', 'Home', 'Away', 'points_home', 'points_away', 'home_team_goal_count', 
                       'away_team_goal_count', 'average_goals_per_match_home', 'average_goals_per_match_away', 
                       'points_per_game_home', 'points_per_game_away', 'home_xG_total', 'away_xG_total', 
                       'home_xG_average', 'away_xG_average', 'btts_home_total', 'btts_away_total', 'btts_home_percentage',
                       'btts_away_percentage', 'points_per_goal_home', 'points_per_goal_away', 'goal_difference_home',
                       'goal_difference_away', 'goal_difference_home_per_game', 'goal_difference_away_per_game', 'total_xG_combined',
                       'total_xG_average_per_game','home_progessive_carries_per_game', 'away_progessive_carries_per_game',
                        'home_progessive_passes_per_game', 'away_progessive_passes_per_game','home_prg_pass_carries_per_goal', 
                        'away_prg_pass_carries_per_goal', 'home_wins', 'away_wins', 'home_draws', 'away_draws', 'home_loss', 'away_loss', 'label']
    
    old_columns = ['Wk','Home','xG_home','Score','xG_away','Away']
    squad_cols = ['Squad', 'PrgC', 'PrgP', 'MP']
    misc_cols = ['Squad', 'Recov', ]
    squad_file = filename.replace(".csv","")+'_squad_stat.csv'
    misc_file = filename.replace(".csv","")+'_squad_misc.csv'
    match_dataset = pd.read_csv(filename, on_bad_lines='skip',usecols=old_columns)
    match_dataset_cut = pd.DataFrame(match_dataset, columns=old_columns)
    squad_stats = pd.read_csv(squad_file, on_bad_lines='skip', usecols=squad_cols)
    misc_stats = pd.read_csv(misc_file, on_bad_lines='skip', usecols=misc_cols)
    
    
    #Adding IDs for every team instead of strings
    #Make ranking list for the specific team storing the corresponding values for the ranking
    team_df = pd.read_csv('team_ids.csv')
    
    ID_team_list = []
    if 'team' in team_df.columns:
        ID_team_list = team_df[['id', 'team']].values.tolist()

    ranking = [[team, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] for _, team in ID_team_list]

    value_to_id = {team: id for id, team in ID_team_list}

    for id, team in enumerate(match_dataset_cut['Home'].unique()):
        if team not in value_to_id:
            new_id = len(value_to_id) + 1
            value_to_id[team] = new_id

        home_id = value_to_id[team]

        # Update the team IDs in the 'Home' and 'Away' columns of match_dataset_cut
        match_dataset_cut['Home'] = match_dataset_cut['Home'].replace(team, home_id)
        match_dataset_cut['Away'] = match_dataset_cut['Away'].replace(team, home_id)
        squad_stats['Squad'] = squad_stats['Squad'].replace(team, home_id)
        
    squad_stats['Squad'] = squad_stats['Squad'].astype(int) 
    squad_stats = squad_stats.sort_values(by='Squad', ascending=True)
    squad_stats = squad_stats.set_index('Squad')

    team_df = pd.DataFrame.from_dict(value_to_id, orient='index', columns=['id'])
    team_df['team'] = team_df.index  # Add the 'team' column with original team names
    team_df.to_csv('team_ids.csv', index=False)
    
    for index, row in match_dataset_cut.iterrows():
        
        goalsHome,_,goalsAway = list(row['Score'])
        goalsHome = int(goalsHome)
        goalsAway = int(goalsAway)
        xgHome = float(row['xG_home'])
        xgAway = float(row['xG_away'])
        gameweek = int(row['Wk'])
        result_match = goalsHome - goalsAway
        
        home_id = row['Home']
        away_id = row['Away']
        
        # Check if home team ID is present in the ranking list
        if home_id not in [team[0] for team in ranking]:
            # If home team ID is not present, add it dynamically to the ranking list
            ranking.append([home_id, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        # Check if away team ID is present in the ranking list
        if away_id not in [team[0] for team in ranking]:
            # If away team ID is not present, add it dynamically to the ranking list
            ranking.append([away_id, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        
        #Init the first week for every stat to be 0 ahead of the game.
        if gameweek == 1:
            match_dataset_cut.loc[index, 'points_home'] = 0
            match_dataset_cut.loc[index, 'points_away'] = 0
            match_dataset_cut.loc[index, 'home_team_goal_count'] = 0
            match_dataset_cut.loc[index, 'away_team_goal_count'] = 0
            match_dataset_cut.loc[index, 'average_goals_per_match_home'] = 0
            match_dataset_cut.loc[index, 'average_goals_per_match_away'] = 0
            match_dataset_cut.loc[index, 'points_per_game_home'] = 0
            match_dataset_cut.loc[index, 'points_per_game_away'] = 0
            match_dataset_cut.loc[index, 'home_xG_total'] = 0
            match_dataset_cut.loc[index, 'away_xG_total'] = 0
            match_dataset_cut.loc[index, 'home_xG_average'] = 0
            match_dataset_cut.loc[index, 'away_xG_average'] = 0
            match_dataset_cut.loc[index, 'btts_home_total'] = 0
            match_dataset_cut.loc[index, 'btts_away_total'] = 0
            match_dataset_cut.loc[index, 'btts_home_percentage'] = 0  
            match_dataset_cut.loc[index, 'btts_away_percentage'] = 0
            match_dataset_cut.loc[index, 'points_per_goal_home'] = 0
            match_dataset_cut.loc[index, 'points_per_goal_away'] = 0 
            match_dataset_cut.loc[index, 'goal_difference_home'] = 0
            match_dataset_cut.loc[index, 'goal_difference_away'] = 0 
            match_dataset_cut.loc[index, 'goal_difference_home_per_game'] = 0
            match_dataset_cut.loc[index, 'goal_difference_away_per_game'] = 0
            match_dataset_cut.loc[index, 'total_xG_combined'] = 0
            match_dataset_cut.loc[index, 'total_xG_average_per_game'] = 0
            match_dataset_cut.loc[index, 'home_progessive_passes_per_game'] = 0
            match_dataset_cut.loc[index, 'away_progessive_passes_per_game'] = 0
            match_dataset_cut.loc[index, 'home_progessive_carries_per_game'] = 0
            match_dataset_cut.loc[index, 'away_progessive_carries_per_game'] = 0
            match_dataset_cut.loc[index, 'home_prg_pass_carries_per_goal'] = 0
            match_dataset_cut.loc[index, 'away_prg_pass_carries_per_goal'] = 0
            match_dataset_cut.loc[index, 'home_team_favourite'] = 0
            match_dataset_cut.loc[index, 'away_team_favourite'] = 0
            match_dataset_cut.loc[index, 'home_wins'] = 0
            match_dataset_cut.loc[index, 'away_wins'] = 0
            match_dataset_cut.loc[index, 'home_draws'] = 0
            match_dataset_cut.loc[index, 'away_draws'] = 0
            match_dataset_cut.loc[index, 'home_loss'] = 0
            match_dataset_cut.loc[index, 'away_loss'] = 0
    
        if result_match > 0:
            ranking[home_id][1] += 3
            ranking[home_id][18] += 1
            ranking[away_id][20] += 1
            match_dataset_cut.loc[index, 'label'] = 1
            
        if result_match < 0:
            ranking[away_id][1] += 3
            ranking[home_id][20] += 1
            ranking[away_id][18] += 1
            match_dataset_cut.loc[index, 'label'] = 2
            
        if result_match == 0:
            ranking[home_id][1] += 1
            ranking[away_id][1] += 1
            ranking[home_id][19] += 1
            ranking[home_id][19] += 1
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
        
        ranking[home_id][14] = squad_stats.loc[home_id, 'PrgP'] / squad_stats.loc[home_id, 'MP']
        ranking[away_id][14] = squad_stats.loc[away_id, 'PrgP'] / squad_stats.loc[away_id, 'MP']
        
        ranking[home_id][15] = squad_stats.loc[home_id, 'PrgC'] / squad_stats.loc[home_id, 'MP']
        ranking[away_id][15] = squad_stats.loc[away_id, 'PrgC'] / squad_stats.loc[away_id, 'MP']
        
        if ranking[home_id][2] != 0:
            ranking[home_id][16] = (ranking[home_id][14] + ranking[home_id][15]) / ranking[home_id][2]
        else:
            ranking[home_id][16] = 0
        if ranking[away_id][2] != 0:
            ranking[away_id][16] = (ranking[away_id][14] + ranking[away_id][15]) / ranking[away_id][2]
        else:
            ranking[away_id][16] = 0
            
        if ranking[home_id][9] > ranking[away_id][9] and ranking[home_id][5] > ranking[away_id][5] and ranking[home_id][18] > ranking[away_id][18]:
            ranking[home_id][17] = 1
            ranking[away_id][17] = 0
        elif ranking[home_id][9] < ranking[away_id][9] and ranking[home_id][5] < ranking[away_id][5] and not ranking[home_id][18] > ranking[away_id][18]:
            ranking[home_id][17] = 0
            ranking[away_id][17] = 1
        else:
            ranking[home_id][17] = 0.5
            ranking[away_id][17] = 0.5
            
        
            
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
            match_dataset_cut.loc[index, 'home_progessive_passes_per_game'] = ranking[home_id][14]
            match_dataset_cut.loc[index, 'away_progessive_passes_per_game'] = ranking[away_id][14]
            match_dataset_cut.loc[index, 'home_progessive_carries_per_game'] = ranking[home_id][15]
            match_dataset_cut.loc[index, 'away_progessive_carries_per_game'] = ranking[away_id][15]
            match_dataset_cut.loc[index, 'home_prg_pass_carries_per_goal'] = ranking[home_id][16]
            match_dataset_cut.loc[index, 'away_prg_pass_carries_per_goal'] = ranking[away_id][16]
            match_dataset_cut.loc[index, 'home_team_favourite'] = ranking[home_id][17]
            match_dataset_cut.loc[index, 'away_team_favourite'] = ranking[away_id][17]
            match_dataset_cut.loc[index, 'home_wins'] = ranking[home_id][18]
            match_dataset_cut.loc[index, 'away_wins'] = ranking[away_id][18]
            match_dataset_cut.loc[index, 'home_draws'] = ranking[home_id][19]
            match_dataset_cut.loc[index, 'away_draws'] = ranking[away_id][19]
            match_dataset_cut.loc[index, 'home_loss'] = ranking[home_id][20]
            match_dataset_cut.loc[index, 'away_loss'] = ranking[away_id][20]
    
    
    match_dataset_cut = match_dataset_cut[match_dataset_cut['Wk'] != 1]
    match_dataset_cut = match_dataset_cut.drop(['xG_home', 'Score', 'xG_away'], axis=1)  
    match_dataset_cut['label'] = match_dataset_cut['label'].astype(int)  
    match_dataset_cut.to_csv(filename2, index=False, mode='a', header=False)
    print(match_dataset_cut.head())
            
# Iterate over each file in the folder
for file_name in file_list:
    # Check if the file has the .csv extension
    if file_name.endswith('.csv') and not file_name.endswith('squad_stat.csv'):
        # Construct the full file path
        file_path = os.path.join(folder_path, file_name)
        
        # Call the generate_data function with the file path
        generate_data(file_path)