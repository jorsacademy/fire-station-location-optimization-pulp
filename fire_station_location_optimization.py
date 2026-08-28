import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import folium
from pulp import LpBinary, LpMinimize, LpProblem, LpStatus, LpVariable, PULP_CBC_CMD, lpSum, value


SEED = 42
GRID_SIZE_KM = 10
NUM_DEMAND_POINTS = 50
NUM_POTENTIAL_STATIONS = 15
MAX_STATIONS = 5
MAX_RESPONSE_TIME_MIN = 5
BUDGET_K = 12_000
TRUCK_SPEED_KM_PER_MIN = 50 / 60
DISPATCH_TIME_MIN = 1


rng = np.random.default_rng(SEED)


def generate_data():
    demand = []
    for i in range(NUM_DEMAND_POINTS):
        population = int(rng.integers(500, 5000))
        risk_factor = float(rng.uniform(1, 10))
        demand.append(
            {
                "id": i,
                "x": float(rng.uniform(0, GRID_SIZE_KM)),
                "y": float(rng.uniform(0, GRID_SIZE_KM)),
                "population": population,
                "risk_factor": risk_factor,
                "weight": population * risk_factor / 1000,
            }
        )

    stations = []
    for j in range(NUM_POTENTIAL_STATIONS):
        stations.append(
            {
                "id": j,
                "x": float(rng.uniform(0, GRID_SIZE_KM)),
                "y": float(rng.uniform(0, GRID_SIZE_KM)),
                "construction_cost": float(rng.uniform(1500, 3000)),
                "operating_cost": float(rng.uniform(800, 1200)),
            }
        )

    return pd.DataFrame(demand), pd.DataFrame(stations)


def build_response_time_matrix(df_demand, df_stations):
    dx = df_demand["x"].to_numpy()[:, None] - df_stations["x"].to_numpy()[None, :]
    dy = df_demand["y"].to_numpy()[:, None] - df_stations["y"].to_numpy()[None, :]
    distances = np.sqrt(dx**2 + dy**2)
    response_times = DISPATCH_TIME_MIN + distances / TRUCK_SPEED_KM_PER_MIN
    return distances, response_times


def solve_location_model(
    df_demand,
    df_stations,
    response_time_matrix,
    max_stations=MAX_STATIONS,
    budget_k=BUDGET_K,
    max_response_time=MAX_RESPONSE_TIME_MIN,
):
    num_demand = len(df_demand)
    num_stations = len(df_stations)

    model = LpProblem("Fire_Station_Location_Problem", LpMinimize)

    y = LpVariable.dicts("Station", range(num_stations), cat=LpBinary)
    x = LpVariable.dicts(
        "Assignment",
        [(i, j) for i in range(num_demand) for j in range(num_stations)],
        cat=LpBinary,
    )

    total_weight = float(df_demand["weight"].sum())
    model += (
        lpSum(
            df_demand.loc[i, "weight"]
            * response_time_matrix[i, j]
            * x[(i, j)]
            for i in range(num_demand)
            for j in range(num_stations)
        )
        / total_weight
    )

    for i in range(num_demand):
        model += (
            lpSum(x[(i, j)] for j in range(num_stations)) == 1,
            f"assign_demand_{i}",
        )

    for i in range(num_demand):
        for j in range(num_stations):
            model += x[(i, j)] <= y[j], f"open_link_{i}_{j}"

            # If a station cannot reach demand point i within the required time,
            # prohibit that assignment directly.
            if response_time_matrix[i, j] > max_response_time:
                model += x[(i, j)] == 0, f"response_limit_{i}_{j}"

    model += (
        lpSum(y[j] for j in range(num_stations)) <= max_stations,
        "max_station_count",
    )

    model += (
        lpSum(df_stations.loc[j, "construction_cost"] * y[j] for j in range(num_stations))
        <= budget_k,
        "construction_budget",
    )

    model.solve(PULP_CBC_CMD(msg=False))

    status = LpStatus[model.status]
    result = {
        "status": status,
        "selected_stations": [],
        "assignments": {},
        "total_construction_cost": None,
        "total_operating_cost": None,
        "coverage_percentage": None,
        "avg_weighted_response_time": None,
        "objective_value": None,
    }

    if status != "Optimal":
        return result

    selected = [j for j in range(num_stations) if y[j].value() and y[j].value() > 0.5]
    assignments = {}
    for i in range(num_demand):
        for j in range(num_stations):
            if x[(i, j)].value() and x[(i, j)].value() > 0.5:
                assignments[i] = j
                break

    weighted_time = sum(
        df_demand.loc[i, "weight"] * response_time_matrix[i, j]
        for i, j in assignments.items()
    )

    covered = sum(
        response_time_matrix[i, j] <= max_response_time
        for i, j in assignments.items()
    )

    result.update(
        {
            "selected_stations": selected,
            "assignments": assignments,
            "total_construction_cost": sum(
                df_stations.loc[j, "construction_cost"] for j in selected
            ),
            "total_operating_cost": sum(
                df_stations.loc[j, "operating_cost"] for j in selected
            ),
            "coverage_percentage": 100 * covered / num_demand,
            "avg_weighted_response_time": weighted_time / total_weight,
            "objective_value": value(model.objective),
        }
    )
    return result


def plot_solution(df_demand, df_stations, result, max_response_time=MAX_RESPONSE_TIME_MIN):
    plt.figure(figsize=(10, 10))

    plt.scatter(
        df_demand["x"],
        df_demand["y"],
        alpha=0.5,
        s=df_demand["population"] / 100,
        label="Demand Points",
    )
    plt.scatter(
        df_stations["x"],
        df_stations["y"],
        marker="s",
        s=100,
        label="Potential Stations",
    )

    selected = result["selected_stations"]
    if selected:
        selected_df = df_stations.loc[selected]
        plt.scatter(
            selected_df["x"],
            selected_df["y"],
            marker="*",
            s=220,
            label="Selected Stations",
        )

    for i, j in result["assignments"].items():
        plt.plot(
            [df_demand.loc[i, "x"], df_stations.loc[j, "x"]],
            [df_demand.loc[i, "y"], df_stations.loc[j, "y"]],
            alpha=0.15,
        )

    # response_time = dispatch + distance/speed
    coverage_radius_km = max(0.0, (max_response_time - DISPATCH_TIME_MIN) * TRUCK_SPEED_KM_PER_MIN)
    for j in selected:
        circle = plt.Circle(
            (df_stations.loc[j, "x"], df_stations.loc[j, "y"]),
            coverage_radius_km,
            fill=False,
            alpha=0.3,
        )
        plt.gca().add_patch(circle)

    plt.title("Fire Station Location Optimization")
    plt.xlabel("X coordinate (km)")
    plt.ylabel("Y coordinate (km)")
    plt.grid(True)
    plt.legend()
    plt.axis("equal")
    plt.xlim(0, GRID_SIZE_KM)
    plt.ylim(0, GRID_SIZE_KM)
    plt.tight_layout()
    plt.savefig("fire_station_optimization.png", dpi=300)
    plt.close()


def create_interactive_map(df_demand, df_stations, result, max_response_time=MAX_RESPONSE_TIME_MIN):
    base_lat, base_lng = 40.7128, -74.0060
    lat_scale = 0.01
    lng_scale = 0.01

    map_center = [
        base_lat + GRID_SIZE_KM / 2 * lat_scale,
        base_lng + GRID_SIZE_KM / 2 * lng_scale,
    ]
    fmap = folium.Map(location=map_center, zoom_start=12)

    for i, row in df_demand.iterrows():
        lat = base_lat + row["y"] * lat_scale
        lng = base_lng + row["x"] * lng_scale
        radius = np.sqrt(row["population"]) / 10
        red = int(255 * row["risk_factor"] / 10)
        color = f"#{red:02x}0000"

        folium.CircleMarker(
            location=[lat, lng],
            radius=radius,
            color=color,
            fill=True,
            fill_opacity=0.4,
            tooltip=f"Demand {i}: Pop={int(row['population'])}, Risk={row['risk_factor']:.1f}",
        ).add_to(fmap)

    coverage_radius_km = max(0.0, (max_response_time - DISPATCH_TIME_MIN) * TRUCK_SPEED_KM_PER_MIN)

    for j, row in df_stations.iterrows():
        lat = base_lat + row["y"] * lat_scale
        lng = base_lng + row["x"] * lng_scale

        if j in result["selected_stations"]:
            folium.Marker(
                location=[lat, lng],
                icon=folium.Icon(color="red", icon="fire-extinguisher", prefix="fa"),
                tooltip=f"Station {j}: Selected",
            ).add_to(fmap)
            folium.Circle(
                location=[lat, lng],
                radius=coverage_radius_km * 1000,
                color="red",
                fill=True,
                fill_opacity=0.1,
            ).add_to(fmap)
        else:
            folium.Marker(
                location=[lat, lng],
                icon=folium.Icon(color="gray", icon="building", prefix="fa"),
                tooltip=f"Station {j}: Not Selected",
            ).add_to(fmap)

    for i, j in result["assignments"].items():
        start = [
            base_lat + df_demand.loc[i, "y"] * lat_scale,
            base_lng + df_demand.loc[i, "x"] * lng_scale,
        ]
        end = [
            base_lat + df_stations.loc[j, "y"] * lat_scale,
            base_lng + df_stations.loc[j, "x"] * lng_scale,
        ]
        folium.PolyLine([start, end], weight=1, opacity=0.5).add_to(fmap)

    fmap.save("fire_station_map.html")


def run_sensitivity_analysis(df_demand, df_stations, response_time_matrix):
    station_results = []
    for station_limit in range(1, 8):
        result = solve_location_model(
            df_demand,
            df_stations,
            response_time_matrix,
            max_stations=station_limit,
            budget_k=BUDGET_K,
        )
        station_results.append(
            {
                "max_stations": station_limit,
                "status": result["status"],
                "num_stations_built": len(result["selected_stations"]) if result["status"] == "Optimal" else np.nan,
                "total_cost": result["total_construction_cost"],
                "avg_response_time": result["avg_weighted_response_time"],
            }
        )

    station_df = pd.DataFrame(station_results)
    print("\nResponse time vs. number of stations:")
    print(station_df)

    valid = station_df[station_df["status"] == "Optimal"]
    if not valid.empty:
        plt.figure(figsize=(10, 6))
        plt.plot(valid["max_stations"], valid["avg_response_time"], "o-")
        plt.xlabel("Maximum Number of Stations Allowed")
        plt.ylabel("Average Weighted Response Time (minutes)")
        plt.title("Trade-off between Number of Stations and Response Time")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("sensitivity_analysis.png", dpi=300)
        plt.close()

    budget_results = []
    for multiplier in [0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8]:
        test_budget = BUDGET_K * multiplier
        result = solve_location_model(
            df_demand,
            df_stations,
            response_time_matrix,
            max_stations=MAX_STATIONS,
            budget_k=test_budget,
        )
        budget_results.append(
            {
                "budget": test_budget,
                "status": result["status"],
                "num_stations_built": len(result["selected_stations"]) if result["status"] == "Optimal" else np.nan,
                "total_cost": result["total_construction_cost"],
                "avg_response_time": result["avg_weighted_response_time"],
            }
        )

    budget_df = pd.DataFrame(budget_results)
    print("\nResponse time vs. budget:")
    print(budget_df)
    return station_df, budget_df


def main():
    df_demand, df_stations = generate_data()
    _, response_time_matrix = build_response_time_matrix(df_demand, df_stations)

    result = solve_location_model(df_demand, df_stations, response_time_matrix)
    print(f"Status: {result['status']}")

    if result["status"] != "Optimal":
        print("No feasible optimal solution found for the current parameters.")
        return

    print(f"Selected stations: {result['selected_stations']}")
    print(f"Total construction cost: ${result['total_construction_cost']:,.2f}k")
    print(f"Annual operating cost: ${result['total_operating_cost']:,.2f}k")
    print(f"Coverage within {MAX_RESPONSE_TIME_MIN} minutes: {result['coverage_percentage']:.2f}%")
    print(f"Weighted average response time: {result['avg_weighted_response_time']:.2f} minutes")

    plot_solution(df_demand, df_stations, result)
    create_interactive_map(df_demand, df_stations, result)
    run_sensitivity_analysis(df_demand, df_stations, response_time_matrix)

    print("\n======= Fire Station Location Optimization Report =======")
    print(f"City grid: {GRID_SIZE_KM} km x {GRID_SIZE_KM} km")
    print(f"Number of demand points: {NUM_DEMAND_POINTS}")
    print(f"Maximum allowed stations: {MAX_STATIONS}")
    print(f"Budget: ${BUDGET_K:,.2f}k")
    print(f"Maximum response time: {MAX_RESPONSE_TIME_MIN} minutes")
    print(f"Number of stations to build: {len(result['selected_stations'])}")
    print(f"Selected station locations: {result['selected_stations']}")
    print(f"Total construction cost: ${result['total_construction_cost']:,.2f}k")
    print(f"Total annual operating cost: ${result['total_operating_cost']:,.2f}k")
    print(f"Coverage: {result['coverage_percentage']:.2f}%")
    print(f"Weighted average response time: {result['avg_weighted_response_time']:.2f} minutes")
    print("=========================================================")


if __name__ == "__main__":
    main()
