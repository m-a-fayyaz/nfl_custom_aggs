import collections
import yaml
import pandas as pd
import numpy as np


def level_list():
    """
    Central point to control possible aggregation levels

    Returns:
        list
    """

    level_list = ['season', 'week', 'half', 'quarter']

    return level_list


def flatten_dict(d, parent_key='', sep='_'):
    """
    Flatten nested dicitonary into a single layer of keys

    Args:
        d (dictionary): nested dictionary
        parent_key (string): inital key to attach for first value (defaults to none)
        sep (string): seperator between nested keys
    Returns:
        dictionary
    """

    # Check for valid inputs
    if d is None:
        raise ValueError('Input must be a dictionary.')

    if not isinstance(d, dict):
        raise ValueError('Input must be a dictionary.')

    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))

    return dict(items)


def read_yaml(filepath):
    """
    Read yaml file using safe load method

    Args:
        filepath (string): location of yaml file
    Returns:
        dictionary
    """

    # Check for valid input
    if filepath == None:
        ValueError('Must specify filepath.')

    with open(filepath, 'r') as stream:
        try:
            new_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    return new_dict


def custom_pivot(df, values, columns, index):
    """
    Pivot a dataframe aggregating by multiple values, returns flat dataframe only for single value in columns

    Args:
        df (dataframe): flat dataframe
        values (string or List[string]) : columns to aggregate
        columns (string or List[string]): keys to group aggregations by on the pivot table
        index (string or List[string]): keys to group by on the pivot table
    Returns:
        dataframe
    """

    # Check for valid input
    if not isinstance(df, pd.DataFrame):
        raise ValueError('Input must be a dataframe.')

    # Apply pivot from pandas module
    pivotTable_df = pd.pivot_table(df, values=values, index=index, columns=columns, aggfunc=np.max)

    # Reset indeces for each value
    merge_df = pd.DataFrame
    counter = 0
    last_value = ''
    for value in values:
        if counter == 0:
            merge_df = pivotTable_df[value].reset_index()
            merge_df = merge_df.reset_index()
        else:
            temp_df = pivotTable_df[value].reset_index()
            temp_df = temp_df.reset_index()
            left_suffix = "_" + last_value
            right_suffix = "_" + value
            merge_df = merge_df.merge(temp_df, how='outer', on=index, suffixes=(left_suffix, right_suffix))
        counter+=1
        last_value = value

    return merge_df


def column_binner(df, column, start, end, step):
    """
    Groups values from a column into distinct bins

    Args:
        df (dataframe): flat dataframe
        column (column): column containing values to be binned
        start (integer): initial value for binning
        end (integer): terminating value for binning
        step (integer): length of bines
    Returns:
        dataframe
    """
    # Check for valid inputs
    if not isinstance(df, pd.DataFrame):
        raise ValueError('Input must be a dataframe.')

    if not isinstance(column, str):
        raise ValueError('Column name must be a string.')

    for x in (start, end, step):
        if not isinstance(x, int):
            raise ValueError(str(x)+' must be an integer.')


    # Create distinct bins
    bins = np.arange(start, end+1, step)

    # Create binned column
    new_col_name = column + '_binned'
    df[new_col_name] = pd.cut(df[column], bins)

    return df


def agg_indexer(level='quarter', add_index=None):
    """
    Returns the index columns for a pbp dataframe at the specified level of aggregation

    Args:
        level (string): aggregation level (season, week, half, quarter)
        add_index (list[string]): extra columns to add to index
    Returns:
        list
    """

    # Check for valid inputs
    if level not in level_list():
        raise ValueError('Input for level must be one of the following: '+str(level_list()))

    if not isinstance(add_index, (list, tuple, None)):
        raise ValueError('Input for add_index must be a list, tuple, or empty.')

    if add_index!=None:
        for x in add_index:
            if not isinstance(x, str):
                raise ValueError('Invalid input \''+str(x)+'\': inputs for add_index must be strings')

    # Initialize index
    agg_index = ['season']

    # Extend for level
    if level=='season':
        pass
    elif level=='week':
        agg_index.extend(['game_id'])
    elif level =='half':
        agg_index.extend(['game_id', 'game_half'])
    elif level=='quarter':
        agg_index.extend(['game_id', 'game_half', 'game_quarter'])
    else:
        raise ValueError('Input for level must be one of the following: '+str(level_list()))

    # Extend for add_index
    if add_index!=None: agg_index.extend(add_index)

    return agg_index


def years_error_catcher(years, min_year=1999):
    """
    Central point to control error thrown for the years parameter

    Returns:
        void
    """

    if years is None:
        raise ValueError('Input for years must be a list or range of integers.')

    if not isinstance(years, (list, range)):
        raise ValueError('Input for years must be a list or range of integers.')

    if isinstance(years, list):
        for year in years:
            if not isinstance(year, int):
                raise ValueError('Invalid input '+str(year)+': inputs for years must be integers in a list or range')

    if len(years) > 0:
        if min(years) < min_year:
            raise ValueError('Data not available before '+str(min_year))


def main():
    print("Helper functions are defined.")


if __name__ == "__main__":
    main()
