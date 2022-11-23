import pandas as pd
import numpy as np
from utils.helper_functions import *
import logging
import time


# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(name)s:%(asctime)s:%(filename)s:%(funcName)s:%(message)s')

file_handler = logging.FileHandler('logs/pbp.log')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)


def misc_enrich(pbp_df):
    """
    Adds columns for quarter & targets to play-by-play dataframe

    Args:
        pbp_df (dataframe): pre-loaded play-by-play dataframe
    Returns:
        dataframe
    """

    start_time = time.time()

    # Check for valid input
    if not isinstance(pbp_df, pd.DataFrame):
        raise ValueError('Input must be a dataframe.')

    # Calculate quarter (0=first quarter... 4=OT quarter)
    pbp_df['quarter_raw'] = pbp_df.sort_values(by=['game_id','play_id'], ascending=[True,True]) \
                                  .groupby(by=['game_id'])['quarter_end'].cumsum().astype('int')

    # Add one to start at 1 (1=first quarter... 5=OT quarter)
    pbp_df['game_quarter'] = pbp_df['quarter_raw'] + 1
    pbp_df = pbp_df.drop(columns=['quarter_raw'])

    # Add flag for targets
    pbp_df['target'] = np.where((pbp_df['play_type']=='pass'), 1, 0)

    # Add unique play_id
    pbp_df['play_id_unique'] = pbp_df['game_id'] + '_' + pbp_df['play_id'].astype(int).astype(str)

    # Make kick-off receiving team the posession team
    pbp_df['posteam_adjusted'] = np.where(
        pbp_df['play_type_nfl']=='KICK_OFF', 
        pbp_df['defteam'], 
        pbp_df['posteam']
    )
    pbp_df['defteam_adjusted'] = np.where(
        pbp_df['play_type_nfl']=='KICK_OFF', 
        pbp_df['posteam'], 
        pbp_df['defteam']
    )
    
    # Make passer_id the rusher_id for QB scrambles
    pbp_df['rusher_id_adjusted'] = np.where(
        pbp_df['qb_scramble']==1,
        pbp_df['passer_id'],
        pbp_df['rusher_id']
    )

    end_time = time.time()
    memory = round((pbp_df.memory_usage(deep=True).sum()/1000000),4)
    runtime = round((end_time-start_time),4)
    rate = round((memory/runtime),4)
    logger.info('{}sec:{}MB:{}MB/s'.format(runtime, memory, rate))

    return pbp_df


def fantasy_rec_enrich(pbp_df, rec_rec=1, rec_yd=0.1, rec_td=6, rec_twoPointConv=2, rec_lostFumble=-2):
    """
    Adds fantasy receiving metrics to play-by-play dataframe without any aggregation

    Args:
        pbp_df (dataframe): pre-loaded play-by-play dataframe
        rec_rec (int, float): points per reception (0 for standard)
        rec_yd (int, float): points per receiving yard
        rec_td (int, float): points per receiving touchdown
        rec_twoPointConv (int, float): points for receiving two point conversion
        rec_lostFumble (int, float): points for receiving lost fumble
    Returns:
        dataframe
    """

    start_time = time.time()

    # Check for valid input
    if not isinstance(pbp_df, pd.DataFrame):
        raise ValueError('Input must be a dataframe.')

    for x in (rec_rec, rec_yd, rec_td, rec_twoPointConv):
        if not isinstance(x, (int, float)):
            raise ValueError(str(x)+' must be an integer or float.')

    # Add individual columns for receiving events
    pbp_df['fantasy_rec_rec_pts'] = np.where(
        (pbp_df['complete_pass']==1), 
        rec_rec, 
        0
    )
    pbp_df['fantasy_rec_yd_pts'] = np.where(
        (pbp_df['receiving_yards']!=0)&(pbp_df['receiving_yards'].notnull()),
        (pbp_df['receiving_yards']*rec_yd), 
        0
    )
    pbp_df['fantasy_rec_td_pts'] = np.where(
        (pbp_df['td_player_id'].notnull())&(pbp_df['play_type']=='pass')&(pbp_df['td_team']==pbp_df['posteam']),
        rec_td, 
        0
    )
    pbp_df['fantasy_rec_twoPointConv_pts'] = np.where(
        (pbp_df['two_point_conv_result']=='success')&(pbp_df['play_type']=='pass'),
        rec_twoPointConv, 
        0
    )
    pbp_df['fantasy_re _lostFumble_pts'] = np.where(
        (pbp_df['fumble_lost']==1)&(pbp_df['fumbled_1_player_id']==pbp_df['receiver_id']),
        rec_lostFumble, 
        0
    )

    # Sum all receiving events
    pbp_df['fantasy_rec_pts'] = pbp_df['fantasy_rec_rec_pts'] + \
                                pbp_df['fantasy_rec_yd_pts'] + \
                                pbp_df['fantasy_rec_td_pts'] + \
                                pbp_df['fantasy_rec_twoPointConv_pts'] + \
                                pbp_df['fantasy_re _lostFumble_pts']

    end_time = time.time()
    memory = round((pbp_df.memory_usage(deep=True).sum()/1000000),4)
    runtime = round((end_time-start_time),4)
    rate = round((memory/runtime),4)
    logger.info('{}sec:{}MB:{}MB/s'.format(runtime, memory, rate))

    return pbp_df


def fantasy_rush_enrich(pbp_df, rush_yd=0.1, rush_td=6, rush_twoPointConv=2, rush_lostFumble=-2):
    """
    Adds fantasy rushing metrics to play-by-play dataframe without any aggregation

    Args:
        pbp_df (dataframe): pre-loaded play-by-play dataframe
        rush_yd (int): points per rushing yard
        rush_td (int): points per passing touchdown
        rush_twoPointConv (int): points for rushing two point conversion
        rush_lostFumble (int): points per lost fumble (negative or 0)
    Returns:
        dataframe
    """

    start_time = time.time()

    # Check for valid input
    if not isinstance(pbp_df, pd.DataFrame):
        raise ValueError('Input must be a dataframe.')

    for x in (rush_yd, rush_td, rush_twoPointConv, rush_lostFumble):
        if not isinstance(x, (int, float)):
            raise ValueError(str(x)+' must be an integer or float.')

    # Add individual columns for rushing events
    pbp_df['fantasy_rush_yd_pts'] = np.where(
        (pbp_df['rushing_yards']!=0)&(pbp_df['rushing_yards'].notnull()),
        pbp_df['rushing_yards']*rush_yd, 
        0
    )
    pbp_df['fantasy_rush_td_pts'] = np.where(
        (pbp_df['td_player_id'].notnull())&(pbp_df['play_type']=='run')&(pbp_df['td_team']==pbp_df['posteam']),
        rush_td, 
        0
    )
    pbp_df['fantasy_rush_lostFumble_pts'] = np.where(
        (pbp_df['fumble_lost']==1)&(pbp_df['play_type']=='run'),
        rush_lostFumble, 
        0
    )
    pbp_df['fantasy_rush_twoPointConv_pts'] = np.where(
        (pbp_df['two_point_conv_result']=='success')&(pbp_df['play_type']=='run'),
        rush_twoPointConv, 
        0
    )

    # Sum all rushing events
    pbp_df['fantasy_rush_pts'] = pbp_df['fantasy_rush_yd_pts'] + \
                                 pbp_df['fantasy_rush_td_pts'] + \
                                 pbp_df['fantasy_rush_lostFumble_pts'] + \
                                 pbp_df['fantasy_rush_twoPointConv_pts']

    end_time = time.time()
    memory = round((pbp_df.memory_usage(deep=True).sum()/1000000),4)
    runtime = round((end_time-start_time),4)
    rate = round((memory/runtime),4)
    logger.info('{}sec:{}MB:{}MB/s'.format(runtime, memory, rate))

    return pbp_df


def fantasy_pass_enrich(pbp_df, pass_yd=0.04, pass_td=4, pass_twoPointConv=2, pass_lostFumble=-2, pass_int=-1):
    """
    Adds fantasy passing metrics to play-by-play dataframe without any aggregation

    Args:
        pbp_df (dataframe): pre-loaded play-by-play dataframe
        pass_yd (int): points per passing yard
        pass_td (int): points per passing touchdown
        pass_twoPointConv (int): points for passing for two point conversion
        pass_int (int): points per interception (negative or 0)
    Returns:
        dataframe
    """

    start_time = time.time()

    # Check for valid input
    if not isinstance(pbp_df, pd.DataFrame):
        raise ValueError('Input must be a dataframe.')

    for x in (pass_yd, pass_td, pass_twoPointConv, pass_int):
        if not isinstance(x, (int, float)):
            raise ValueError(str(x)+' must be an integer or float.')

    # Add individual columns for passing events
    pbp_df['fantasy_pass_yd_pts'] = np.where(
        (pbp_df['passing_yards']!=0)&(pbp_df['passing_yards'].notnull()),
        pbp_df['passing_yards']*pass_yd, 
        0
    )
    pbp_df['fantasy_pass_td_pts'] = np.where(
        (pbp_df['td_player_id'].notnull())&(pbp_df['play_type']=='pass')&(pbp_df['td_team']==pbp_df['posteam']),
        pass_td, 
        0
    )
    pbp_df['fantasy_pass_int_pts'] = np.where(
        pbp_df['interception']==1, 
        pass_int, 
        0
    )
    pbp_df['fantasy_pass_lostFumble_pts'] = np.where(
        (pbp_df['fumble_lost']==1)&(pbp_df['fumbled_1_player_id']==pbp_df['passer_id']),
        pass_lostFumble, 
        0
    )
    pbp_df['fantasy_pass_twoPointConv_pts'] = np.where(
        (pbp_df['two_point_conv_result']=='success')&(pbp_df['play_type']=='pass'),
        pass_twoPointConv, 
        0
    )

    # Sum all passing events
    pbp_df['fantasy_pass_pts'] = pbp_df['fantasy_pass_yd_pts'] + \
                                 pbp_df['fantasy_pass_td_pts'] + \
                                 pbp_df['fantasy_pass_int_pts'] + \
                                 pbp_df['fantasy_pass_lostFumble_pts'] + \
                                 pbp_df['fantasy_pass_twoPointConv_pts']

    end_time = time.time()
    memory = round((pbp_df.memory_usage(deep=True).sum()/1000000),4)
    runtime = round((end_time-start_time),4)
    rate = round((memory/runtime),4)
    logger.info('{}sec:{}MB:{}MB/s'.format(runtime, memory, rate))

    return pbp_df


def fantasy_player_off_agg(pbp_fantasy_df, level='season'):
    """
    Aggregates play-by-play fantasy-enriched dataframe by several offensive fantasy metrics at the specified level

    Args:
        pbp_fantasy_df (dataframe): pre-loaded play-by-play fantasy-enriched dataframe
        level (string): aggregation level (season, week, half, quarter)
    Returns:
        dataframe
    """

    start_time = time.time()

    # Check for valid input
    if not isinstance(pbp_fantasy_df, pd.DataFrame):
        raise ValueError('Input for pbp_fantasy_df must be a dataframe.')

    if level not in level_list():
        raise ValueError('Input for level must be one of the following: '+str(level_list()))

    # Create indexes
    rush_agg_index = agg_indexer(level, add_index=['rusher_id_adjusted'])
    rec_agg_index = agg_indexer(level, add_index=['receiver_id'])
    pass_agg_index = agg_indexer(level, add_index=['passer_id'])
    agg_index = agg_indexer(level, add_index=['player_id'])

    # Aggregate rushing, receiving, and passing metrics
    rush_fant_agg_df = pbp_fantasy_df.groupby(by=rush_agg_index) \
                                     .agg({'fantasy_rush_pts':'sum'}).reset_index() \
                                     .rename(columns={'rusher_id_adjusted':'player_id'})
    rec_fant_agg_df = pbp_fantasy_df.groupby(by=rec_agg_index) \
                                    .agg({'fantasy_rec_pts':'sum'}).reset_index() \
                                    .rename(columns={'receiver_id':'player_id'})
    pass_fant_agg_df = pbp_fantasy_df.groupby(by=pass_agg_index) \
                                     .agg({'fantasy_pass_pts':'sum'}).reset_index() \
                                     .rename(columns={'passer_id':'player_id'})

    # Merge aggreations
    final_df = rush_fant_agg_df.merge(rec_fant_agg_df, how='outer', on=agg_index) \
                               .merge(pass_fant_agg_df, how='outer', on=agg_index) \
                               .fillna(0)

    final_df['fantasy_total_pts'] = final_df['fantasy_rush_pts'] + final_df['fantasy_rec_pts'] + final_df['fantasy_pass_pts']

    end_time = time.time()
    memory = round((final_df.memory_usage(deep=True).sum()/1000000),4)
    runtime = round((end_time-start_time),4)
    rate = round((memory/runtime),4)
    logger.info('{}sec:{}MB:{}MB/s'.format(runtime, memory, rate))

    return final_df


def player_rush_agg(pbp_df, level='season'):
    """
    Aggregates pbp dataframe by key rushing metrics at the specified level

    Args:
        pbp_df (dataframe): pre-loaded play-by-play dataframe
        level (string): aggregation level (season, week, half, quarter)
    Returns:
        dataframe
    """

    start_time = time.time()

    # Check for valid inputs
    if not isinstance(pbp_df, pd.DataFrame):
        raise ValueError('Input for pbp_df must be a dataframe.')

    if level not in level_list():
        raise ValueError('Input for level must be one of the following: '+str(level_list()))

    # Create indexes
    agg_index = agg_indexer(level, add_index=['rusher_id_adjusted'])
    td_agg_index = agg_indexer(level, add_index=['td_player_id'])

    # Aggregate rushing metrics excluding touchdowns
    filter_df = pbp_df[pbp_df['play_type']=='run']
    agg_raw_df = pd.DataFrame(
        filter_df.groupby(by=agg_index) \
                 .agg({'play_id_unique':'count','rushing_yards':'sum'})
    )
    agg_df = agg_raw_df.reset_index() \
                       .rename(columns={'play_id_unique':'carries'})

    # Aggregate rushing touchdowns
    filter_td_df = filter_df[filter_df['td_player_id'].notnull()]
    agg_raw_td_df = pd.DataFrame(
        filter_td_df.groupby(by=td_agg_index) \
                    .agg({'play_id_unique':'count'})
    )
    agg_td_df = agg_raw_td_df.reset_index() \
                             .rename(columns={'play_id_unique':'rushing_tds', 'td_player_id':'rusher_id_adjusted'})

    # Merge aggregations
    final_df = agg_df.merge(agg_td_df, how='left', on=agg_index) \
                     .rename(columns={'rusher_id_adjusted':'player_id'})

    end_time = time.time()
    memory = round((final_df.memory_usage(deep=True).sum()/1000000),4)
    runtime = round((end_time-start_time),4)
    rate = round((memory/runtime),4)
    logger.info('{}sec:{}MB:{}MB/s'.format(runtime, memory, rate))

    return final_df


def player_rec_agg(pbp_df, level='season'):
    """
    Aggregates pbp dataframe by key receiving metrics at the specified level

    Args:
        pbp_df (dataframe): pre-loaded play-by-play dataframe
        level (string): aggregation level (season, week, half, quarter)
    Returns:
        dataframe
    """

    start_time = time.time()

    # Check for valid input
    if not isinstance(pbp_df, pd.DataFrame):
        raise ValueError('Input for pbp_df must be a dataframe.')

    if level not in level_list():
        raise ValueError('Input for level must be one of the following: '+str(level_list()))

    # Create indexes
    agg_index = agg_indexer(level, add_index=['receiver_id'])
    td_agg_index = agg_indexer(level, add_index=['td_player_id'])

    # Aggregate receiving metrics excluding targets and touchdowns
    filter_df = pbp_df[(pbp_df['play_type']=='pass')&(pbp_df['complete_pass']==1)]
    agg_raw_df = pd.DataFrame(
        filter_df.groupby(by=agg_index) \
                 .agg({'play_id_unique':'count','receiving_yards':'sum'})
    )
    agg_df = agg_raw_df.reset_index().rename(columns={'play_id_unique':'receptions'})

    # Aggregate receiving targets
    filter_targ_df = pbp_df[(pbp_df['play_type']=='pass')]
    agg_raw_targ_df = pd.DataFrame(
        filter_targ_df.groupby(by=agg_index) \
                      .agg({'play_id_unique':'count'})
    )
    agg_targ_df = agg_raw_targ_df.reset_index().rename(columns={'play_id_unique':'targets'})

    # Aggregate receiving toucdowns
    filter_td_df = filter_df[filter_df['td_player_id'].notnull()]
    agg_raw_td_df = pd.DataFrame(
        filter_td_df.groupby(by=td_agg_index) \
                    .agg({'play_id_unique':'count'})
    )
    agg_td_df = agg_raw_td_df.reset_index().rename(columns={'play_id_unique':'receiving_tds', 'td_player_id':'receiver_id'})

    # Merge aggregations
    final_df = agg_targ_df.merge(agg_df, how='left', on=agg_index) \
                          .merge(agg_td_df, how='left', on=agg_index) \
                          .fillna({"receptions": 0, "targets": 0}) \
                          .astype({"receptions": int, "targets": int}) \
                          .rename(columns={'receiver_id':'player_id'})

    end_time = time.time()
    memory = round((final_df.memory_usage(deep=True).sum()/1000000),4)
    runtime = round((end_time-start_time),4)
    rate = round((memory/runtime),4)
    logger.info('{}sec:{}MB:{}MB/s'.format(runtime, memory, rate))

    return final_df


def off_agg_enrich(off_agg_df, level='season'):
    """
    Enriches offensive aggregated dataframe with additional columns

    Args:
        off_agg_df (dataframe): pbp dataframe aggregated for offensive data
        level (string): level of aggregation
    Returns:
        dataframe
    """

    start_time = time.time()

    # Check for valid input
    if not isinstance(off_agg_df, pd.DataFrame):
        raise ValueError('Input for pbp_df must be a dataframe.')

    if level not in level_list():
        raise ValueError('Input for level must be one of the following: '+str(level_list()))

    # Fetch agg index
    agg_index = agg_indexer(level=level)

    # Fill NA aggregated columns
    column_list = ['fantasy_total_pts','fantasy_rush_pts','fantasy_rec_pts','fantasy_pass_pts']
    off_agg_df[column_list] = off_agg_df[column_list].fillna(value=0)

    # Add rank columns
    fantasy_overall_rank = 'fantasy_overall_rank_' + level
    off_agg_df[fantasy_overall_rank] = off_agg_df.groupby(agg_index)['fantasy_total_pts'] \
                                                 .rank(method='dense', ascending=False) \
                                                 .astype(int)

    fantasy_rush_rank = 'fantasy_rush_rank_' + level
    off_agg_df[fantasy_rush_rank] = off_agg_df.groupby(agg_index)['fantasy_rush_pts'] \
                                                .rank(method='dense', ascending=False) \
                                                .astype(int)

    fantasy_rec_rank = 'fantasy_rec_rank_' + level
    off_agg_df[fantasy_rec_rank] = off_agg_df.groupby(agg_index)['fantasy_rec_pts'] \
                                             .rank(method='dense', ascending=False) \
                                             .astype(int)

    fantasy_pass_rank = 'fantasy_pass_rank_' + level
    off_agg_df[fantasy_pass_rank] = off_agg_df.groupby(agg_index)['fantasy_pass_pts'] \
                                              .rank(method='dense', ascending=False) \
                                              .astype(int)

    fantasy_position_rank = 'fantasy_position_rank_' + level
    off_agg_df[fantasy_position_rank] = off_agg_df.groupby(agg_index+['position'])['fantasy_total_pts'] \
                                                  .rank(method='dense', ascending=False) \
                                                  .astype(int)

    # Add next date level fantasy total 
    next_col = 'fantasy_total_pts_next' + level.capitalize()
    next_col_diff = next_col + 'Diff'

    off_agg_df[next_col] = off_agg_df.sort_values(by=agg_index+['player_id'], ascending=False) \
                                     .groupby(['player_id'])['fantasy_total_pts'] \
                                     .shift(1)

    off_agg_df[next_col_diff] = off_agg_df['fantasy_total_pts'] - off_agg_df[next_col]

    end_time = time.time()
    memory = round((off_agg_df.memory_usage(deep=True).sum()/1000000),4)
    runtime = round((end_time-start_time),4)
    rate = round((memory/runtime),4)
    logger.info('{}sec:{}MB:{}MB/s'.format(runtime, memory, rate))

    return off_agg_df


def participation_expander(pbp_part_df):
    """
    Split offensive and defensive personnel into seperate columns

    Args:
        pbp_part_df (dataframe): pre-loaded participation-enriched play-by-play dataframe
    Returns:
        dataframe
    """

    start_time = time.time()

    # Check for valid input
    if not isinstance(pbp_part_df, pd.DataFrame):
        raise ValueError('Input for pbp_expand_df must be a dataframe.')

    # Fetch generic position lists
    defense_player_columns, offense_player_columns = position_list_generator()

    # Split defensive players
    def_split_df = pbp_part_df['defense_players'].str.split(';', expand=True).fillna(np.nan)

    print('Splitting personnel.')

    # Drop players beyond 11
    for column in def_split_df.columns:
        if int(column) > 10:
            def_split_df = def_split_df.drop(columns=[column])

    # Ensure at least 11 players are included
    while def_split_df.columns.max() < 10:
        max_split = def_split_df.columns.max()
        new_col = int(max_split + 1)
        def_split_df.insert(new_col,new_col,None)

    # Rename defensive columns
    def_split_df = def_split_df.rename(columns={i: col for i,col in enumerate(defense_player_columns)})

    # Split offesnive players
    off_split_df = pbp_part_df['offense_players'].str.split(';', expand=True).fillna(np.nan)

    # Drop players beyond 11
    for column in off_split_df.columns:
        if int(column) > 10:
            off_split_df = off_split_df.drop(columns=[column])

    # Ensure at least 11 players are included
    while off_split_df.columns.max() < 10:
        max_split = off_split_df.columns.max()
        new_col = int(max_split + 1)
        off_split_df.insert(new_col,new_col,None)

    # Rename defensive columns
    off_split_df = off_split_df.rename(columns={i: col for i,col in enumerate(offense_player_columns)})

    print('Joining split data.')

    # Join splits to pbp data
    pbp_part_df = pbp_part_df.join(def_split_df).join(off_split_df)

    end_time = time.time()
    memory = round((pbp_part_df.memory_usage(deep=True).sum()/1000000),4)
    runtime = round((end_time-start_time),4)
    rate = round((memory/runtime),4)
    logger.info('{}sec:{}MB:{}MB/s'.format(runtime, memory, rate))

    return pbp_part_df


def snap_counter(pbp_expand_df: pd.DataFrame, level='season'):
    """
    Count snaps for players and teams at the specified level

    Args:
        pbp_expand_df (dataframe): pre-loaded pariticipation-enriched-expanded play-by-play dataframe
        level (string): level of aggregation
    Returns:
        dataframe
    """
    
    start_time = time.time()

     # Check for valid input
    if not isinstance(pbp_expand_df, pd.DataFrame):
        raise ValueError('Input for pbp_expand_df must be a dataframe.')

    if level not in level_list():
        raise ValueError('Input for level must be one of the following: '+str(level_list()))

    # Generate position lists
    defense_player_columns, offense_player_columns = position_list_generator()
    all_player_columns = defense_player_columns + offense_player_columns

    # Set melt parameters
    id_vars_col = [
        'season','game_id','game_half','game_quarter','fixed_drive','play_id_unique',
        'play_type','play_type_nfl','special_teams_play','posteam_adjusted','defteam_adjusted'
    ]

    play_types = ['KICK_OFF','RUSH','PASS','PUNT','FIELD_GOAL','SACK','XP_KICK','PENALTY']

    pbp_melt_df = pbp_expand_df[
        (pbp_expand_df['play_type_nfl'].isin(play_types))&
        (pbp_expand_df['play_type']!='no_play')
    ].melt(id_vars=id_vars_col, value_vars=all_player_columns, var_name='player_category', value_name='player_id')

    pbp_melt_df['player_side'] = pbp_melt_df['player_category'].str[:7]

    # Generate aggregation indexes
    agg_index_off_player = agg_indexer(level=level, add_index=['player_id','posteam_adjusted'])
    agg_index_off = agg_indexer(level=level, add_index=['posteam_adjusted'])
    # agg_index_def_player = agg_indexer(level=level, add_index=['player_id','defteam_adjusted'])
    # agg_index_def = agg_indexer(level=level, add_index=['defteam_adjusted'])

    # Offense, no special teams
    offense_player_snaps = pbp_melt_df[
        (pbp_melt_df['player_side']=='offense')&
        (pbp_melt_df['special_teams_play']==0)
    ].groupby(agg_index_off_player) \
     .agg({'play_id_unique':pd.Series.nunique}) \
     .reset_index() \
     .rename(columns={'play_id_unique':'offense_player_snaps'})

    offense_snaps = pbp_melt_df[
        (pbp_melt_df['special_teams_play']==0)
    ].groupby(agg_index_off) \
     .agg({'play_id_unique':pd.Series.nunique}) \
     .reset_index() \
     .rename(columns={'play_id_unique':'offense_team_snaps'})

    # Offense, with special teams (kicking)
    st_offense_player_snaps = pbp_melt_df[
        (pbp_melt_df['player_side']=='offense')&
        (pbp_melt_df['special_teams_play']==1)
    ].groupby(agg_index_off_player) \
     .agg({'play_id_unique':pd.Series.nunique}) \
     .reset_index() \
     .rename(columns={'play_id_unique':'st_offense_player_snaps'})

    st_offense_snaps = pbp_melt_df[
        (pbp_melt_df['special_teams_play']==1)
    ].groupby(agg_index_off) \
     .agg({'play_id_unique':pd.Series.nunique}) \
     .reset_index() \
     .rename(columns={'play_id_unique':'st_offense_team_snaps'})

    # # Defense, no special teams
    # defense_player_snaps = pbp_melt_df[
    #     (pbp_melt_df['player_side']=='defense')&
    #     (pbp_melt_df['special_teams_play']==0)
    # ].groupby(agg_index_def_player) \
    #  .agg({'play_id_unique':pd.Series.nunique}) \
    #  .reset_index() \
    #  .rename(columns={'play_id_unique':'defense_player_snaps'})

    # defense_snaps = pbp_melt_df[
    #     (pbp_melt_df['special_teams_play']==0)
    # ].groupby(agg_index_def) \
    #  .agg({'play_id_unique':pd.Series.nunique}) \
    #  .reset_index() \
    #  .rename(columns={'play_id_unique':'defense_team_snaps'})

    # # Defense, with special teams (receiving)
    # st_defense_player_snaps = pbp_melt_df[
    #     (pbp_melt_df['player_side']=='defense')&
    #     (pbp_melt_df['special_teams_play']==1)
    # ].groupby(agg_index_def_player) \
    #  .agg({'play_id_unique':pd.Series.nunique}) \
    #  .reset_index() \
    #  .rename(columns={'play_id_unique':'st_defense_player_snaps'})

    # st_defense_snaps = pbp_melt_df[
    #     (pbp_melt_df['special_teams_play']==1)
    # ].groupby(agg_index_def) \
    #  .agg({'play_id_unique':pd.Series.nunique}) \
    #  .reset_index() \
    #  .rename(columns={'play_id_unique':'st_defense_team_snaps'})
    
    # Join offensive data
    offense_join_df = offense_player_snaps.merge(offense_snaps, how='inner', on=agg_index_off) \
                                          .merge(st_offense_player_snaps, how='outer', on=agg_index_off_player) \
                                          .merge(st_offense_snaps, how='inner', on=agg_index_off)
    
    offense_join_df['offense_player_snap_pct'] = offense_join_df['offense_player_snaps']/offense_join_df['offense_team_snaps']
    offense_join_df['st_offense_player_snap_pct'] = offense_join_df['st_offense_player_snaps']/offense_join_df['st_offense_team_snaps']

    end_time = time.time()
    memory = round((pbp_melt_df.memory_usage(deep=True).sum()/1000000),4)
    runtime = round((end_time-start_time),4)
    rate = round((memory/runtime),4)
    logger.info('{}sec:{}MB:{}MB/s'.format(runtime, memory, rate))

    return offense_join_df


def complete_agg(pbp_fantasy_df, level='season'):
    """
    Applies all offensive aggregations to a fasntasy_enriched play-by-play dataframe

    Args:
        pbp_fantasy_df (dataframe): pre-loaded fantasy-enriched play-by-play dataframe
    Returns:
        dataframe
    """

    start_time = time.time()

    # Check for valid input
    if not isinstance(pbp_fantasy_df, pd.DataFrame):
        raise ValueError('Input for pbp_fantasy_df must be a dataframe.')

    # Create index for pbp_functions
    agg_index = agg_indexer(level, add_index=['player_id'])

    final_df = fantasy_player_off_agg(pbp_fantasy_df).merge(player_rush_agg(pbp_fantasy_df), how='outer', on=agg_index) \
                                                     .merge(player_rec_agg(pbp_fantasy_df), how='outer', on=agg_index)

    end_time = time.time()
    memory = round((final_df.memory_usage(deep=True).sum()/1000000),4)
    runtime = round((end_time-start_time),4)
    rate = round((memory/runtime),4)
    logger.info('{}sec:{}MB:{}MB/s'.format(runtime, memory, rate))

    return final_df


def main():
    print("PBP data functions are defined.")


if __name__ == "__main__":
    main()
