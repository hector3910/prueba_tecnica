import pandas as pd
import numpy as np
from collections import Counter

# =============================================================================
# CASO 1: Análisis de la Copa Mundial Femenina
# =============================================================================

# -----------------------------------------------------------------------------
# 1.1 Carga de datos
# -----------------------------------------------------------------------------

wc = pd.read_csv('https://raw.githubusercontent.com/daramireh/simonBolivarCienciaDatos/refs/heads/main/world_cup_women.csv')
matches = pd.read_csv('https://raw.githubusercontent.com/daramireh/simonBolivarCienciaDatos/refs/heads/main/matches_1991_2023.csv')

pd.set_option('display.max_columns', None)

# -----------------------------------------------------------------------------
# 1.2 Análisis de nulos, duplicados y tipos de variables
# -----------------------------------------------------------------------------

print(f"Valores nulos por variable en Wc:\n{wc.isnull().sum()}")
print('\n')
print(f"Valores nulos por variable en Matches:\n{matches.isnull().sum()}")

print(f"Valores duplicados en Wc:\n{wc.duplicated().sum()}")
print('\n')
print(f"Valores duplicados en Matches:\n{matches.duplicated().sum()}")

print(f"Tipos de datos en Wc:\n{wc.dtypes}")
print('\n')
print(f"Tipos de datos en Matches:\n{matches.dtypes}")

# -----------------------------------------------------------------------------
# 1.3 Tabla de posiciones del mundial 1991
# -----------------------------------------------------------------------------

wc1991 = matches.loc[matches['Year']==1991, ['home_team', 'away_team', 'home_score', 'away_score', 'home_red_card', 'away_red_card', 'home_yellow_card_long', 'away_yellow_card_long', 'Round']]
wc1991['home_win']  = (wc1991['home_score'] >  wc1991['away_score']).astype(int)
wc1991['home_draw'] = (wc1991['home_score'] == wc1991['away_score']).astype(int)
wc1991['home_loss'] = (wc1991['home_score'] <  wc1991['away_score']).astype(int)
wc1991['home_points'] = wc1991['home_win']*3 + wc1991['home_draw']
wc1991['away_win']  = (wc1991['away_score'] >  wc1991['home_score']).astype(int)
wc1991['away_draw'] = (wc1991['away_score'] == wc1991['home_score']).astype(int)
wc1991['away_loss'] = (wc1991['away_score'] <  wc1991['home_score']).astype(int)
wc1991['away_points'] = wc1991['away_win']*3 + wc1991['away_draw']

wc1991['Round'] = wc1991['Round'].replace({
    'Group stage': 0,
    'Quarter-finals': 1,
    'Semi-finals': 2,
    'Third-place match': 3,
    'Final': 4
})

cols = ['home_yellow_card_long', 'away_yellow_card_long', 'home_red_card', 'away_red_card']
for col in cols:
    wc1991[col] = wc1991[col].str.split(',').str.len().fillna(0).astype(int)

home = wc1991[['home_team', 'home_score', 'away_score',
               'home_win', 'home_draw', 'home_loss', 'home_points', 'Round', 'home_red_card', 'home_yellow_card_long']].copy()
home.columns = ['team', 'gf', 'gc', 'win', 'draw', 'loss', 'points', 'Round', 'red_card', 'yellow_card']

away = wc1991[['away_team', 'away_score', 'home_score',
               'away_win', 'away_draw', 'away_loss', 'away_points', 'Round', 'away_red_card', 'away_yellow_card_long']].copy()
away.columns = ['team', 'gf', 'gc', 'win', 'draw', 'loss', 'points', 'Round', 'red_card', 'yellow_card']

all_matches = pd.concat([home, away], ignore_index=True)

standings = all_matches.groupby('team').agg(
    PJ     = ('team',        'count'),
    PG     = ('win',         'sum'),
    PE     = ('draw',        'sum'),
    PP     = ('loss',        'sum'),
    GF     = ('gf',          'sum'),
    GC     = ('gc',          'sum'),
    round  = ('Round',       'max'),
    yellow = ('yellow_card', 'sum'),
    red    = ('red_card',    'sum'),
    PTS    = ('points',      'sum')
).reset_index()

standings['DG']  = standings['GF'] - standings['GC']
standings['JL']  = -standings['yellow'] - 2 * standings['red']
standings['PTS'] = standings.pop('PTS')

standings = standings.sort_values(
    by        = ['round', 'PTS', 'DG', 'GF'],
    ascending = [False, False, False, False]
).reset_index(drop=True)
standings.index += 1

print(standings.drop(columns=['round', 'yellow', 'red']).to_markdown())

# -----------------------------------------------------------------------------
# 1.4 Tabla de goleadoras del mundial 2023
# -----------------------------------------------------------------------------

wc2023 = matches.loc[matches['Year']==2023, ['home_goal', 'away_goal', 'home_own_goal', 'away_own_goal', 'home_penalty_goal', 'away_penalty_goal']]

cols = wc2023.columns
for col in cols:
    wc2023[col] = wc2023[col].str.split('|')

for col in cols:
    for i in range(len(wc2023)):
        if wc2023[col].iloc[i] is not np.nan:
            n = len(wc2023[col].iloc[i])
            for j in range(n):
                wc2023[col].iloc[i][j] = wc2023[col].iloc[i][j].split(' ·')[0].split(' (P)')[0].strip()

contador = Counter()
goal_cols = ['home_goal', 'away_goal', 'home_penalty_goal', 'away_penalty_goal']
for col in goal_cols:
    for i in range(len(wc2023)):
        if wc2023[col].iloc[i] is not np.nan:
            contador.update(wc2023[col].iloc[i])

for col in ['home_own_goal', 'away_own_goal']:
    for i in range(len(wc2023)):
        if wc2023[col].iloc[i] is not np.nan:
            contador.update({'Autogol': len(wc2023[col].iloc[i])})

goles = (pd.DataFrame.from_dict(contador, orient='index', columns=['Goles'])
           .reset_index()
           .rename(columns={'index': 'Jugadora'})
           .sort_values('Goles', ascending=False)).reset_index(drop=True)

print(goles)

# -----------------------------------------------------------------------------
# 1.5 Tabla general (todos los años)
# -----------------------------------------------------------------------------

table = matches[['Year', 'Host', 'home_team', 'away_team', 'home_score', 'away_score', 'Attendance']]
table['home_win']  = (table['home_score'] >  table['away_score']).astype(int)
table['home_draw'] = (table['home_score'] == table['away_score']).astype(int)
table['home_loss'] = (table['home_score'] <  table['away_score']).astype(int)
table['away_win']  = (table['away_score'] >  table['home_score']).astype(int)
table['away_draw'] = (table['away_score'] == table['home_score']).astype(int)
table['away_loss'] = (table['away_score'] <  table['home_score']).astype(int)


home = table[['Year', 'Host', 'Attendance', 'home_team', 'home_score', 'away_score',
              'home_win', 'home_draw', 'home_loss']].copy()
home.columns = ['year', 'host', 'attendance', 'team', 'gf', 'gc', 'win', 'draw', 'loss']


away = table[['Year', 'Host', 'Attendance', 'away_team', 'away_score', 'home_score',
              'away_win', 'away_draw', 'away_loss']].copy()
away.columns = ['year', 'host', 'attendance', 'team', 'gf', 'gc', 'win', 'draw', 'loss']


matches_concat = pd.concat([home, away], ignore_index=True)


standings = matches_concat.groupby(['year', 'host', 'team']).agg(
    PJ         = ('team',       'count'),
    GF         = ('gf',         'sum'),
    GC         = ('gc',         'sum'),
    PG         = ('win',        'sum'),
    PE         = ('draw',       'sum'),
    PP         = ('loss',       'sum'),
    Attendance = ('attendance', 'sum')
).round(2).reset_index()

standings['AVG_GF']          = (standings['GF']         / standings['PJ']).round(2)
standings['AVG_GC']          = (standings['GC']         / standings['PJ']).round(2)
standings['AVG_Attendance']  = (standings['Attendance'] / standings['PJ']).round(0).astype(int)

print(standings.sort_values(by=['year', 'team'], ascending=[False, True]).drop(columns=['Attendance']).reset_index(drop=True))
