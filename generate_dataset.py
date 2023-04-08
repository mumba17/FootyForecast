# import requests
# from bs4 import BeautifulSoup
# import csv
import pandas as pd

# url = "https://fbref.com/en/comps/23/schedule/Eredivisie-Scores-and-Fixtures"
# response = requests.get(url)

# soup = BeautifulSoup(response.content, 'html.parser')
# table = soup.find('table')

# rows = table.find_all('tr')
# data = []
# for row in rows:
#     cols = row.find_all('td')
#     cols = [col.text.strip() for col in cols]
#     data.append(cols)
    
# with open('data.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerows(data)

def generate_data(filename):
    dataset_columns = ['PPG_H', 'PPG_A', 'home_team_goal_count',
                        'away_team_goal_count', 'home_xG', 'away_xG',
                        'average_goals_per_match_home', 'btts_percentage_pre_match',
                        'points_per_goal_home','points_per_goal_away','PH','PA','label']
    
    old_columns = ['Round','Home','xG_home','Score','xG_away','Away']
    match_dataset = pd.read_csv(filename, on_bad_lines='skip')
    match_dataset_cut = pd.DataFrame(match_dataset, columns=old_columns)
    for index, row in match_dataset_cut.iterrows():
        goalsHome,_,goalsAway = list(row['Score'])
        endscore = int(goalsHome) - int(goalsAway)
        totalGoals = goalsHome + goalsAway
        if int(row['Round']) == 1:
            match_dataset_cut.loc[index, 'PH'] = 0
            match_dataset_cut.loc[index, 'PA'] = 0
            match_dataset_cut.loc[index, 'home_team_goal_count'] = 0
            match_dataset_cut.loc[index, 'away_team_goal_count'] = 0
            match_dataset_cut.loc[index, 'average_goals_per_match_home'] = 0
            match_dataset_cut.loc[index, 'PPG_H'] = 0
            match_dataset_cut.loc[index, 'PPG_A'] = 0
        if endscore > 0:
            match_dataset_cut.loc[index, 'label'] = 1
            match_dataset_cut.loc[index, 'PH'] = (int(row['PH']) + 3)
            match_dataset_cut.loc[index, 'PPG_H'] = int(row['PH']) / int(row['Round'])
            match_dataset_cut.loc[index, 'home_team_goal_count'] = int(row['home_team_goal_count']) + goalsHome
            match_dataset_cut.loc[index, 'average_goals_per_match_home'] = ((int(row['Round'] - 1) * int(row['average_goals_per_match'])) + totalGoals) / int(row['Round'])
            match_dataset_cut.loc[index, 'points_per_goal_home'] = int(row['PH']) / int(row['home_team_goal_count'])
            
generate_data('eredivisie_21_22.csv')