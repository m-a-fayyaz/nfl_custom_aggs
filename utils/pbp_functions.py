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
    pbp_df.drop(columns=['quarter_raw'])

    # Add flag for targets
    pbp_df['target'] = np.where((pbp_df['play_type']=='pass'), 1, 0)

    end_time = time.time()
    memory = round((pbp_df.memory_usage(deep=True).sum()/1000000),4)
    runtime = round((end_time-start_time),4)
    rate = round((memory/runtime),4)
    logger.info('{}sec:{}MB:{}MB/s'.format(runtime, memory, rate))

    return pbp_df


def fantasy_rec_enrich(pbp_df, rec_rec=1, rec_yd=0.1, rec_td=6, rec_twoPointConv=2):
    """
    Adds fantasy receiving metrics to play-by-play dataframe without any aggregation

    Args:
        pbp_df (dataframe): pre-loaded play-by-play dataframe
        rec_rec (int): points per reception (0 for standard)
        rec_yd (int): points per receiving yard
        rec_td (int): points per receiving touchdown
        rec_twoPointConv (int): points for receiving two point conversion
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
    pbp_df['fantasy_rec_yd_pts'] = (pbp_df['receiving_yards']*rec_yd) + rec_rec

    pbp_df['fantasy_rec_td_pts'] = np.where((pbp_df['td_player_id'].notnull())&(pbp_df['play_type']=='pass'),
                                            rec_td, 0)

    pbp_df['fantasy_rec_twoPointConv_pts'] = np.where((pbp_df['two_point_conv_result']=='success')&(pbp_df['play_type']=='pass'),
                                                      rec_twoPointConv, 0)

    # Sum all receiving events
    pbp_df['fantasy_rec_pts'] = pbp_df['fantasy_rec_yd_pts'] + \
                                pbp_df['fantasy_rec_td_pts'] + \
                                pbp_df['fantasy_rec_twoPointConv_pts']

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
    pbp_df['fantasy_rush_yd_pts'] = pbp_df['rushing_yards']*rush_yd
    pbp_df['fantasy_rush_td_pts'] = np.where((pbp_df['td_player_id'].notnull())&(pbp_df['play_type']=='run'),
                                             rush_td, 0)
    pbp_df['fantasy_rush_lostFumble_pts'] = np.where(pbp_df['fumble_lost']==1,
                                                     rush_lostFumble, 0)
    pbp_df['fantasy_rush_twoPointConv_pts'] = np.where((pbp_df['two_point_conv_result']=='success')&(pbp_df['play_type']=='run'),
                                                       rush_twoPointConv, 0)

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


def fantasy_pass_enrich(pbp_df, pass_yd=0.04, pass_td=4, pass_twoPointConv=2, pass_int=-1):
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
    pbp_df['fantasy_pass_yd_pts'] = pbp_df['passing_yards']*pass_yd
    pbp_df['fantasy_pass_td_pts'] = np.where((pbp_df['td_player_id'].notnull())&(pbp_df['play_type']=='pass'),
                                             pass_td, 0)
    pbp_df['fantasy_pass_int_pts'] = np.where(pbp_df['interception']==1,
                                              pass_int, 0)
    pbp_df['fantasy_pass_twoPointConv_pts'] = np.where((pbp_df['two_point_conv_result']=='success')&(pbp_df['play_type']=='pass'),
                                                       pass_twoPointConv, 0)

    # Sum all passing events
    pbp_df['fantasy_pass_pts'] = pbp_df['fantasy_pass_yd_pts'] + \
                                 pbp_df['fantasy_pass_td_pts'] + \
                                 pbp_df['fantasy_pass_int_pts'] + \
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
    rush_agg_index = agg_indexer(level, add_index=['rusher_id'])
    rec_agg_index = agg_indexer(level, add_index=['receiver_id'])
    pass_agg_index = agg_indexer(level, add_index=['passer_id'])
    agg_index = agg_indexer(level, add_index=['player_id'])

    # Aggregate rushing, receiving, and passing metrics
    rush_fant_agg_df = pbp_fantasy_df.groupby(by=rush_agg_index) \
                                     .agg({'fantasy_rush_pts':'sum'}).reset_index() \
                                     .rename(columns={'rusher_id':'player_id'})
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
    agg_index = agg_indexer(level, add_index=['rusher_id'])
    td_agg_index = agg_indexer(level, add_index=['td_player_id'])

    # Aggregate rushing metrics excluding touchdowns
    filter_df = pbp_df[pbp_df['play_type']=='run']
    agg_raw_df = pd.DataFrame(
        filter_df.groupby(by=agg_index) \
                 .agg({'play_id':'count','rushing_yards':'sum'})
    )
    agg_df = agg_raw_df.reset_index() \
                       .rename(columns={'play_id':'carries'})

    # Aggregate rushing touchdowns
    filter_td_df = filter_df[filter_df['td_player_id'].notnull()]
    agg_raw_td_df = pd.DataFrame(
        filter_td_df.groupby(by=td_agg_index) \
                    .agg({'play_id':'count'})
    )
    agg_td_df = agg_raw_td_df.reset_index() \
                             .rename(columns={'play_id':'rushing_tds', 'td_player_id':'rusher_id'})

    # Merge aggregations
    final_df = agg_df.merge(agg_td_df, how='left', on=agg_index) \
                     .rename(columns={'rusher_id':'player_id'})

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
                 .agg({'play_id':'count','receiving_yards':'sum'})
    )
    agg_df = agg_raw_df.reset_index().rename(columns={'play_id':'receptions'})

    # Aggregate receiving targets
    filter_targ_df = pbp_df[(pbp_df['play_type']=='pass')]
    agg_raw_targ_df = pd.DataFrame(
        filter_targ_df.groupby(by=agg_index) \
                      .agg({'play_id':'count'})
    )
    agg_targ_df = agg_raw_targ_df.reset_index().rename(columns={'play_id':'targets'})

    # Aggregate receiving toucdowns
    filter_td_df = filter_df[filter_df['td_player_id'].notnull()]
    agg_raw_td_df = pd.DataFrame(
        filter_td_df.groupby(by=td_agg_index) \
                    .agg({'play_id':'count'})
    )
    agg_td_df = agg_raw_td_df.reset_index().rename(columns={'play_id':'receiving_tds', 'td_player_id':'receiver_id'})

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
