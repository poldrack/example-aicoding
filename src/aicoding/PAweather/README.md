# PAweather (#27)

NOAA weather API — monthly max temperature timeseries for Palo Alto (1960–2000).

## Approach

- Uses the NOAA Climate Data Online (CDO) API v2 to fetch daily TMAX data.
- API key is read from `noaa_api_key.txt`.
- Data is paginated (1000 records per request) and collected year by year.
- Monthly max temperatures computed via pandas groupby on year-month.
- Timeseries plotted with matplotlib.
