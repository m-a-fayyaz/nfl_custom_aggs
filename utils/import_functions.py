from utils.pbp_functions import *
from utils.helper_functions import *
import pandas as pd
import nfl_data_py as nfl
import logging
import time


# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(name)s:%(asctime)s:%(filename)s:%(funcName)s:%(message)s')

file_handler = logging.FileHandler('logs/import.log')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)


def import_pbp_participation(years):
    """
    Imports participation data for plays from next-gen stats through nfl_data-py module

    Args:
        years (list[int], range): years to return participation data
    Returns:
        DataFrame
    """

    start_time = time.time()

    # Check for valid inputs
    years_error_catcher(years=years, min_year=2016)

    # import data
    url = r'https://github.com/nflverse/nflverse-data/releases/download/pbp_participation/pbp_participation_{0}.parquet'

    df = pd.concat([pd.read_parquet(url.format(x), engine='auto') for x in years])

    end_time = time.time()
    memory = round((df.memory_usage(deep=True).sum()/1000000),4)
    runtime = round((end_time-start_time),4)
    rate = round((memory/runtime),4)
    logger.info('{}sec:{}MB:{}MB/s'.format(runtime, memory, rate))

    return df


def participation_expander(pbp_part_df, players_df=None):
    """
    Split offensive and defensive personnel into seperate columns

    Args:
        pbp_part_df (dataframe): pre-loaded participation-enriched play-by-play dataframe
        players_df (dataframe): pre-loaded dataframe storing player info by season, pulls from nfl-data-py module if empty
    Returns:
        dataframe
    """

    start_time = time.time()

    # Import players data if not passed through
    if players_df == None:
        min_year = pbp_part_df['season'].min()
        max_year = pbp_part_df['season'].max()
        year_list = list(range(min_year, max_year))
        players_df = import_players_data(years=year_list)

    players_df_trunc = players_df[['player_id','season','player_name','depth_chart_position']].add_prefix('p_')

    # Create labels for split columns
    defense_player_columns = []
    offense_player_columns = []

    for i in range(1,13):
        defense_string = 'defense_player_'+str(i)
        offense_string = 'offense_player_'+str(i)
        defense_player_columns.extend([defense_string])
        offense_player_columns.extend([offense_string])

    all_player_columns = offense_player_columns + defense_player_columns

    # Split defensive players
    def_split_df = pbp_part_df['defense_players'].str.split(';', expand=True) \
                    .fillna(np.nan) \
                    .rename(columns={i: col for i,col in enumerate(defense_player_columns)})

    # Split offesnive players
    off_split_df = pbp_part_df['offense_players'].str.split(';', expand=True) \
                    .fillna(np.nan) \
                    .rename(columns={i: col for i,col in enumerate(offense_player_columns)})

    # Join splits to pbp data
    pbp_part_df = pbp_part_df.join(def_split_df).join(off_split_df)

    # Add player info
    for column in all_player_columns:
        left_columns = ['season', column]
        right_columns = ['p_season', 'p_player_id']
        new_name = column + '_name'
        new_position = column + '_position'
        pbp_part_df = pbp_part_df.merge(players_df_trunc, how='left', left_on=left_columns, right_on=right_columns) \
                                 .drop(columns=right_columns) \
                                 .rename(columns={'p_player_name':new_name, 'p_depth_chart_position':new_position})

    end_time = time.time()
    memory = round((pbp_part_df.memory_usage(deep=True).sum()/1000000),4)
    runtime = round((end_time-start_time),4)
    rate = round((memory/runtime),4)
    logger.info('{}sec:{}MB:{}MB/s'.format(runtime, memory, rate))

    return pbp_part_df


def import_enrich_pbp_data(years, regular=True):
    """
    nfl_data_py play-by-play data import with fantasy and participation enrichment

    Args:
        years (list[int], range): years to return enriched pbp df
        regular (boolean): True if restricting to regular season
    Returns:
        dataframe
    """

    start_time = time.time()

    # Check for valid inputs
    years_error_catcher(years=years, min_year=1999)

    if not isinstance(regular, bool):
        raise ValueError('Input must be a Boolean.')

    # Participation data only goes back to 2016
    years_part = list(filter(lambda years: years >= 2016, years))

    # Import using nfl_data_py modeule
    pbp_df = nfl.import_pbp_data(years=years, downcast=True, cache=False, alt_path=None)
    if regular==True:
        pbp_df = pbp_df[pbp_df['season_type']=='REG']

    # Import participation data
    part_raw_df = import_pbp_participation(years_part)
    part_df = part_raw_df.rename(columns={'nflverse_game_id':'game_id'})

    print('Adding participation data.')

    # Merge pbp and participation
    pbp_part_df = pbp_df.merge(part_df, how='left', on=['game_id', 'play_id'])

    print('Adding fantasy data.')

    # Apply fantasy and quarter-calc enrichment
    final_df = fantasy_pass_enrich(fantasy_rush_enrich(fantasy_rec_enrich(misc_enrich(pbp_part_df))))

    end_time = time.time()
    memory = round((final_df.memory_usage(deep=True).sum()/1000000),4)
    runtime = round((end_time-start_time),4)
    rate = round((memory/runtime),4)
    logger.info('{}sec:{}MB:{}MB/s'.format(runtime, memory, rate))

    return final_df


def import_off_agg_data(years, regular=True, level='season'):
    """
    Returns an aggregation of several offesnvie metrics at the specfiied level

    Args:
        years (list[int], range): years to return enriched pbp df
        regular (boolean): True if restricting pull to regular season
        level (string): aggregation level (season, week, half, quarter)
    Returns:
        dataframe
    """

    start_time = time.time()

    # Check for valid inputs
    years_error_catcher(years=years, min_year=1999)

    if not isinstance(regular, bool):
        raise ValueError('Input for regular must be a Boolean.')

    if level not in level_list():
        raise ValueError('Input for level must be one of the following: '+str(level_list()))

    # Create index for pbp_functions
    agg_index = agg_indexer(level, add_index=['player_id'])

    # Pull enriched pbp dataframe
    pbp_enrich_df = import_enrich_pbp_data(years=years, regular=regular)

    # Pull players data
    players_df = import_players_data(years)

    print('Aggregating data.')

    # Aggregate dataframe
    pbp_agg_df = fantasy_player_off_agg(pbp_enrich_df, level=level) \
                    .merge(player_rush_agg(pbp_enrich_df, level=level), how='outer', on=agg_index) \
                    .merge(player_rec_agg(pbp_enrich_df, level=level), how='outer', on=agg_index) \
                    .merge(players_df, how='left', on=['player_id','season'])

    end_time = time.time()
    memory = round((pbp_agg_df.memory_usage(deep=True).sum()/1000000),4)
    runtime = round((end_time-start_time),4)
    rate = round((memory/runtime),4)
    logger.info('{}sec:{}MB:{}MB/s'.format(runtime, memory, rate))

    return pbp_agg_df


def import_players_data(years):
    """
    Returns select columns & ensures distinct entries for a player per season from the nfl_data_py module

    Args:
        years (list[int], range): years to return player data
    Returns:
        dataframe
    """

    start_time = time.time()

    # Check for valid inputs
    years_error_catcher(years=years, min_year=1999)

    # Import using nfl_data_py module
    players_df = nfl.import_rosters(years=years)

    # Add estimated start-of-season age
    players_df['age_sos'] = np.where(players_df['birth_date'].dt.month < 9, \
                                     players_df['season'] - players_df['birth_date'].dt.year, \
                                     players_df['season'] - players_df['birth_date'].dt.year - 1)

    # Pull unique rows for the specified index
    players_df_index = [
        'season','player_id','first_name','last_name','player_name','height','weight','years_exp', \
        'draft_club','draft_number','age_sos','position','depth_chart_position','jersey_number'
    ]

    players_distinct_df = players_df.groupby(by=players_df_index) \
                                    .agg({'pfr_id':'count'}).reset_index() \
                                    .drop(columns=['pfr_id'])

    end_time = time.time()
    memory = round((players_distinct_df.memory_usage(deep=True).sum()/1000000),4)
    runtime = round((end_time-start_time),4)
    rate = round((memory/runtime),4)
    logger.info('{}sec:{}MB:{}MB/s'.format(runtime, memory, rate))

    return players_distinct_df


def main():
    print("Data import functions are defined.")


if __name__ == "__main__":
    main()
