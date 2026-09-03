# FR-Nuclear-Output

A Python script that plots France's nuclear electricity generation using live data from the [ENTSO-E Transparency Platform](https://transparency.entsoe.eu).

Register for a free API key at transparency.entsoe.eu

## What it shows

| Line | Style | Description |
|------|-------|-------------|
| This year actual | Solid blue | Daily nuclear output (GW), 7-day rolling avg |
| This year forecast | Dotted blue | Available capacity for remainder of year, derived from EDF's filed reactor outage declarations |
| Previous year actual | Solid orange | Same metric, prior year |
| 5-year average | Dashed green | Day-of-year mean over 2021–2025 |

The forecast line is calculated as: **installed capacity (63 GW) minus planned unavailability**, sourced from ENTSO-E's generation unit outage records (document type A80). This reflects EDF's publicly declared maintenance windows per reactor.

## Requirements

```bash
pip install entsoe-py pandas matplotlib
