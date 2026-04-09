import sqlite3
import pulp
import pandas as pd
import matplotlib

matplotlib.use('Agg')  # Background plotting mode
import matplotlib.pyplot as plt
import numpy as np
import os


def solve_milp(scenario_id="S0", f_cap_override=None, demand_multiplier=1.0, cost_multiplier=1.0):
    """Dynamic function to solve the MILP model with specific sensitivity scenarios."""
    conn = sqlite3.connect('Land_Logistics.db')
    cursor = conn.cursor()

    # Amortization Period (N)
    cursor.execute("SELECT Setting_Value FROM System_Settings WHERE Setting_Name='Amortization_Period_Years'")
    N = cursor.fetchone()[0]

    # Facility Data (Supports Setup Cost and Capacity Overrides)
    cursor.execute("SELECT Facility_ID, Setup_Cost_USD, Annual_Capacity_Tons FROM Facilities")
    f, Cap = {}, {}
    for row in cursor.fetchall():
        f[row[0]] = row[1]
        Cap[row[0]] = row[2] if f_cap_override is None or row[0] not in f_cap_override else f_cap_override[row[0]]
    J = list(f.keys())

    # Waste Demands (Supports Demand Surge Overrides)
    cursor.execute("SELECT City_ID, Category_ID, Annual_Waste_Tons FROM Waste_Demands")
    D = {}
    I_set, K_set = set(), set()
    for row in cursor.fetchall():
        I_set.add(row[0])
        K_set.add(row[1])
        D[(row[0], row[1])] = row[2] * demand_multiplier
    I, K = list(I_set), list(K_set)

    # Transportation Costs (Supports Cost Surge Overrides)
    cursor.execute("SELECT City_ID, Facility_ID, Unit_Transport_Cost FROM Transport_Costs")
    beta = {(row[0], row[1]): row[2] * cost_multiplier for row in cursor.fetchall()}
    conn.close()

    # Optimization Model Initialization
    model = pulp.LpProblem(f"Supply_Chain_Optimization_{scenario_id}", pulp.LpMinimize)
    y = pulp.LpVariable.dicts(f"Facility_{scenario_id}", J, cat='Binary')
    x = pulp.LpVariable.dicts(f"Flow_{scenario_id}", [(i, j, k) for i in I for j in J for k in K], lowBound=0)

    # Objective Function: Annualized CAPEX + OPEX
    model += pulp.lpSum([(f[j] / N) * y[j] for j in J]) + \
             pulp.lpSum([beta[(i, j)] * x[(i, j, k)] for i in I for j in J for k in K])

    # Constraint 1: Fulfill all demand
    for i in I:
        for k in K:
            model += pulp.lpSum([x[(i, j, k)] for j in J]) == D[(i, k)]

    # Constraint 2: Facility Capacity limits
    for j in J:
        model += pulp.lpSum([x[(i, j, k)] for i in I for k in K]) <= Cap[j] * y[j]

    # Solve the model (Suppressing verbose output)
    model.solve(pulp.PULP_CBC_CMD(msg=0))
    return model, y, x, I, J, K, Cap, beta


def main():
    print("=" * 70)
    print(" GLOBAL LOGISTICS INC. - HIERARCHICAL DECISION SUPPORT SYSTEM (AHP-TOPSIS-MILP)")
    print(" (Memory-Safe Dynamic Sensitivity Analysis Module Active)")
    print("=" * 70)

    # ---------------------------------------------------------
    # STAGE 1: AHP (Analytic Hierarchy Process)
    # ---------------------------------------------------------
    print("\n[+] STAGE 1: AHP ANALYSIS (Criteria Weights)")
    ahp_matrix = np.array([
        [1, 4, 3, 5],  # Cost
        [1 / 4, 1, 2, 3],  # Distance
        [1 / 3, 1 / 2, 1, 2],  # Capacity
        [1 / 5, 1 / 3, 1 / 2, 1]  # Environmental Impact
    ])
    eigvals, eigvecs = np.linalg.eig(ahp_matrix)
    weights = np.real(eigvecs[:, np.argmax(eigvals)])
    weights /= np.sum(weights)
    print(
        f" -> Cost: {weights[0] * 100:.1f}% | Distance: {weights[1] * 100:.1f}% | Capacity: {weights[2] * 100:.1f}% | Environment: {weights[3] * 100:.1f}%")

    # ---------------------------------------------------------
    # STAGE 2: TOPSIS (Candidate Facility Ranking)
    # ---------------------------------------------------------
    print("\n[+] STAGE 2: TOPSIS RANKING (Strategic Evaluation)")
    # Randomized/Anonymized values for GitHub portfolio
    decision_matrix = np.array([
        [60000, 150.00, 200, 2900.00],  # F1 (Alpha)
        [45000, 135.50, 150, 2500.00],  # F2 (Beta)
        [30000, 250.00, 100, 4300.00],  # F3 (Gamma)
        [35000, 390.00, 120, 6800.00]  # F4 (Delta)
    ])
    norm_matrix = decision_matrix / np.sqrt((decision_matrix ** 2).sum(axis=0))
    weighted_norm = norm_matrix * weights

    ideal = [np.min(weighted_norm[:, 0]), np.min(weighted_norm[:, 1]), np.max(weighted_norm[:, 2]),
             np.min(weighted_norm[:, 3])]
    n_ideal = [np.max(weighted_norm[:, 0]), np.max(weighted_norm[:, 1]), np.min(weighted_norm[:, 2]),
               np.max(weighted_norm[:, 3])]

    s_plus = np.sqrt(((weighted_norm - ideal) ** 2).sum(axis=1))
    s_minus = np.sqrt(((weighted_norm - n_ideal) ** 2).sum(axis=1))
    topsis_scores = s_minus / (s_plus + s_minus)

    for i, s in enumerate(topsis_scores):
        print(f" -> F{i + 1} TOPSIS Score: {s:.4f}")

    # ---------------------------------------------------------
    # STAGE 3: MILP (Base Scenario Optimization)
    # ---------------------------------------------------------
    print("\n" + "-" * 70)
    print(" STAGE 3: MILP OPTIMIZATION (Network Design)")
    print("-" * 70)

    print("[*] Solving Scenario 0 (Base Case)...")
    res0, y0, x0, I, J, K, Cap0, beta0 = solve_milp("S0")
    cost0 = pulp.value(res0.objective)
    active_facilities0 = [j for j in J if y0[j].varValue is not None and y0[j].varValue > 0.5]
    print(f"    Result: ${cost0:,.2f} USD | Active Facilities: {active_facilities0}")

    # ---------------------------------------------------------
    # STAGE 4: SENSITIVITY ANALYSIS (Scenario Testing)
    # ---------------------------------------------------------
    print("\n" + "-" * 70)
    print(" STAGE 4: SENSITIVITY ANALYSIS (Scenario Testing)")
    print("-" * 70)

    print("[*] Scenario 1: Transportation costs decreased by 15%...")
    res1, y1, _, _, _, _, _, _ = solve_milp("S1", cost_multiplier=0.85)
    print(
        f"    Result: ${pulp.value(res1.objective):,.2f} USD | Active Facilities: {[j for j in J if y1[j].varValue is not None and y1[j].varValue > 0.5]}")

    print("\n[*] Scenario 2: Transportation costs increased by 20%...")
    res2, y2, _, _, _, _, _, _ = solve_milp("S2", cost_multiplier=1.20)
    print(
        f"    Result: ${pulp.value(res2.objective):,.2f} USD | Active Facilities: {[j for j in J if y2[j].varValue is not None and y2[j].varValue > 0.5]}")

    print("\n[*] Scenario 3: Facility Alpha (F1) capacity dropped to 160 tons...")
    res3, y3, _, _, _, _, _, _ = solve_milp("S3", f_cap_override={'F1': 160})
    print(
        f"    Result: ${pulp.value(res3.objective):,.2f} USD | Active Facilities: {[j for j in J if y3[j].varValue is not None and y3[j].varValue > 0.5]} <--- STRATEGY SHIFT!")

    print("\n[*] Scenario 4: Waste generation increased by 25% across all nodes...")
    res4, y4, _, _, _, _, _, _ = solve_milp("S4", demand_multiplier=1.25)
    print(
        f"    Result: ${pulp.value(res4.objective):,.2f} USD | Active Facilities: {[j for j in J if y4[j].varValue is not None and y4[j].varValue > 0.5]} <--- STRATEGY SHIFT!")

    # ---------------------------------------------------------
    # STAGE 5: DETAILED VISUALIZATION & REPORTING
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print(" STAGE 5: GENERATING CHARTS & REPORTS")
    print("=" * 70)

    scenario_names = ['Base (S0)', 'Low Freight (S1)', 'High Freight (S2)', 'Cap Drop (S3)', 'High Demand (S4)']
    costs = [pulp.value(res0.objective), pulp.value(res1.objective), pulp.value(res2.objective),
             pulp.value(res3.objective), pulp.value(res4.objective)]

    facility_labels = {'F1': 'F1 (Alpha)', 'F2': 'F2 (Beta)', 'F3': 'F3 (Gamma)', 'F4': 'F4 (Delta)'}
    active_facility_names_s0 = [facility_labels.get(j, j) for j in active_facilities0]

    # --- Chart 1: Scenario Cost Comparison ---
    plt.figure(figsize=(9, 5))
    colors = ['#1f77b4', '#1f77b4', '#1f77b4', '#d62728', '#d62728']
    bars1 = plt.bar(scenario_names, costs, color=colors, edgecolor='black')
    plt.ylim(0, max(costs) * 1.2)
    plt.ylabel('Total System Cost (USD)', fontweight='bold')
    plt.title('Chart 1: Sensitivity Analysis Scenarios', fontsize=12, fontweight='bold')

    for bar in bars1:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 1500, f'${yval:,.0f}', ha='center', va='bottom',
                 fontweight='bold')

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("Chart_1_Scenarios.png", dpi=300)
    plt.close()

    # --- Chart 2: Financial Breakdown (CAPEX vs OPEX) ---
    conn = sqlite3.connect('Land_Logistics.db')
    cursor = conn.cursor()
    cursor.execute("SELECT Setting_Value FROM System_Settings WHERE Setting_Name='Amortization_Period_Years'")
    N = cursor.fetchone()[0]
    cursor.execute("SELECT Facility_ID, Setup_Cost_USD FROM Facilities")
    f = {row[0]: row[1] for row in cursor.fetchall()}
    conn.close()

    capex_s0 = sum((f[j] / N) * y0[j].varValue for j in J)
    opex_s0 = cost0 - capex_s0

    plt.figure(figsize=(6, 6))
    pie_labels = [f'Freight Operations (OPEX)\n${opex_s0:,.0f}', f'Facility Amortization (CAPEX)\n${capex_s0:,.0f}']
    pie_values = [opex_s0, capex_s0]
    pie_colors = ['#ff7f0e', '#2ca02c']

    plt.pie(pie_values, labels=pie_labels, colors=pie_colors, autopct='%1.1f%%', startangle=140,
            textprops={'fontweight': 'bold'})
    plt.title('Chart 2: Base Case (S0) Financial Breakdown', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig("Chart_2_Financial_Analysis.png", dpi=300)
    plt.close()

    # --- Chart 3: Capacity Utilization ---
    plt.figure(figsize=(7, 5))
    utilization_rates = [(sum(x0[(i, j, k)].varValue for i in I for k in K) / Cap0[j]) * 100 for j in
                         active_facilities0]

    bars3 = plt.bar(active_facility_names_s0, utilization_rates, color='#9467bd', edgecolor='black', width=0.5)
    plt.axhline(y=100, color='r', linestyle='--', label='100% Capacity Limit')
    plt.ylim(0, 115)
    plt.ylabel('Utilization Rate (%)', fontweight='bold')
    plt.title('Chart 3: Capacity Utilization of Active Facilities', fontsize=12, fontweight='bold')

    for bar in bars3:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 2, f'{yval:.1f}%', ha='center', va='bottom',
                 fontweight='bold')

    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("Chart_3_Capacity.png", dpi=300)
    plt.close()

    # --- Chart 4: Waste Category Distribution ---
    plt.figure(figsize=(8, 6))
    category_colors = {'Paper': '#e1aa7a', 'Plastic': '#5b9bd5', 'Glass': '#70ad47', 'Metal': '#a5a5a5'}
    bottom_data = np.zeros(len(active_facilities0))

    for k in K:
        k_values = [sum(x0[(i, j, k)].varValue for i in I) for j in active_facilities0]
        plt.bar(active_facility_names_s0, k_values, bottom=bottom_data, label=k,
                color=category_colors.get(k, '#333333'), edgecolor='white', width=0.5)

        for idx, val in enumerate(k_values):
            if val is not None and val > 0.01:
                plt.text(idx, bottom_data[idx] + (val / 2), f'{val:.1f}t\n({k})', ha='center', va='center', fontsize=9,
                         fontweight='bold')
        bottom_data += np.array(k_values)

    plt.ylabel('Waste Amount (Tons)', fontweight='bold')
    plt.title('Chart 4: Received Waste Categories by Facility', fontsize=12, fontweight='bold')
    plt.legend(title='Categories', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("Chart_4_Waste_Distribution.png", dpi=300)
    plt.close()

    # --- EXCEL REPORT EXPORT ---
    report_data = []
    for i in I:
        for j in active_facilities0:
            for k in K:
                amount = x0[(i, j, k)].varValue
                if amount is not None and amount > 0.01:
                    report_data.append({
                        "Source Node": i,
                        "Target Facility": facility_labels[j],
                        "Waste Category": k,
                        "Amount (Tons)": round(amount, 2),
                        "Freight Cost (USD)": round(amount * beta0[(i, j)], 2)
                    })
    pd.DataFrame(report_data).to_excel("Optimization_Results.xlsx", index=False)

    print(f"\n[✓] SUCCESS! 4 Charts and Excel Report generated successfully:")
    print(f" 1. {os.path.abspath('Chart_1_Scenarios.png')}")
    print(f" 2. {os.path.abspath('Chart_2_Financial_Analysis.png')}")
    print(f" 3. {os.path.abspath('Chart_3_Capacity.png')}")
    print(f" 4. {os.path.abspath('Chart_4_Waste_Distribution.png')}")
    print(f" 5. Excel Report: {os.path.abspath('Optimization_Results.xlsx')}")


if __name__ == "__main__":
    main()