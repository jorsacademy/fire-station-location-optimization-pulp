# Fire Station Location Optimization with PuLP

A mixed-integer linear programming (MILP) example for locating fire stations and assigning demand points under response-time, budget, and station-count constraints.

The project uses synthetic city data so it can run without external datasets. Demand points carry population and fire-risk weights, while candidate fire stations have construction and annual operating costs.

## Model

Decision variables:

- `y[j] = 1` if candidate fire station `j` is opened.
- `x[i,j] = 1` if demand point `i` is assigned to station `j`.

Objective:

- Minimize weighted average emergency response time.

Main constraints:

- Every demand point is assigned to exactly one station.
- Demand can only be assigned to an opened station.
- Assignments exceeding the maximum response time are prohibited.
- The number of opened stations cannot exceed the configured limit.
- Total construction cost cannot exceed the budget.

## Improvements over the initial version

The sensitivity analysis rebuilds and solves a fresh optimization model for every scenario instead of deleting constraints by dictionary position. This avoids modifying the wrong constraints and prevents stale PuLP variable references from contaminating scenario results.

The coverage-radius calculation also uses dimensionally consistent units:

```text
coverage distance = (maximum response time - dispatch time) × truck speed
```

The solver status is checked before solution values are processed, and infeasible scenarios are reported safely.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python fire_station_location_optimization.py
```

The script generates:

- `fire_station_optimization.png` — static visualization of demand points, selected stations, assignments, and coverage radii.
- `fire_station_map.html` — interactive Folium map.
- `sensitivity_analysis.png` — station-count versus weighted response-time analysis.
- Console tables for station-count and budget sensitivity scenarios.

## Default scenario

- City grid: 10 km × 10 km
- Demand points: 50
- Candidate station locations: 15
- Maximum stations: 5
- Maximum response time: 5 minutes
- Construction budget: $12,000k
- Fire-truck speed assumption: 50 km/h
- Dispatch time: 1 minute

## Notes

This is an educational facility-location model based on synthetic Euclidean distances. A production implementation would normally use real GIS coordinates, road-network travel times, existing station locations, station capacities, traffic effects, terrain, shift availability, and uncertainty in incident demand.

## Tech stack

- Python
- PuLP / CBC
- NumPy
- pandas
- Matplotlib
- Folium
