## nfl_custom_aggs

Built on the nfl-data-py module, the nfl_custom_agg functions allow you to import aggregated statistics for individual offensive players at the level of detail you want (season, week, half, quarter, or drive). You can also choose to import the raw play-by-play data with the enrichments used to create the aggregations (fantasy data, individual play lineups, etc.).

### Usage
Clone this repository to your local desktop.

```bash
$ git clone https://github.com/m-a-fayyaz/nfl_custom_aggs.git
```

In a python file or Jypyter notebook in the root folder, import the functions in the **utils** folder.

```python
from utils import *
```

#### Enriched PBP Data
```python
import_enrich_pbp_data(years, regular=True)
```
Returns PBP data from the nfl-data-py module with additional columns (fantasy data, individual play lineups, etc.).

years
: required, list of seasons to pull data for (as early as 1999)

regular
: optional, specify if you only want regular season data

#### Custom Offensive Aggregations
```python
import_off_agg_data(years, regular=True, level='season')
```
Returns aggregated statistics for individual offensive players at the level of detail you want (season, week, half, quarter, or drive).

years
: required, list of seasons to pull data for (as early as 1999)

regular
: optional, specify if you only want regular season data

level
: optional, level of aggregation (season, week, half, quarter, or drive)

