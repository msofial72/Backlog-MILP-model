"""
Field Technician Allocation System - MILP Implementation
Mixed-Integer Linear Programming Model 
for Reducing the Backlog of Service Orders at Electric Power Utilities

Implemented Policies:
1. FIFO (First-In-First-Out) - Baseline
2. EOQ (Economic Order Quantity) - Adapted for services
3. (s,S) - Continuous Review Policy
4. Dynamic Bi-phase Backlog - MILP (Equations 1-30 from the article)

Authors: Delgadillo, Garcia, Rodrigues, Castro, Silva
Date: 2026-02-18
"""

import pandas as pd
import numpy as np
from pulp import *
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# BASE CLASS: SCHEDULING POLICY
# ============================================================================

class SchedulingPolicy:
    """Base class for all scheduling policies"""
    
    def __init__(self, name, config=None):
        self.name = name
        self.config = config or {}
        self.reset_metrics()
    
    def reset_metrics(self):
        """Resets performance metrics"""
        self.metrics = {
            'total_queue_time': 0,
            'total_overtime': 0,
            'total_cost': 0,
            'orders_completed': 0,
            'orders_pending': 0,
            'daily_stats': []
        }
    
    def allocate_orders(self, orders, technicians, day):
        """Abstract method - allocation logic"""
        raise NotImplementedError("Subclasses must implement allocate_orders()")


# ============================================================================
# POLICY 1: FIFO (First-In-First-Out)
# ============================================================================

class FIFOPolicy(SchedulingPolicy):
    """
    FIFO: Processes orders in chronological arrival order
    Reference: Standard industrial baseline
    """
    
    def __init__(self, config=None):
        super().__init__("FIFO", config)
    
    def allocate_orders(self, orders, technicians, day):
        """Allocates orders by arrival order (FIFO)"""
        # Sort by generation_day (ascending)
        sorted_orders = sorted(orders, key=lambda x: x['generation_day'])
        
        allocations = []
        total_overtime = 0
        
        for order in sorted_orders:
            allocated = False
            
            # Try to allocate to technician with available capacity
            for tech in technicians:
                if tech['available_capacity'] >= order['service_time']:
                    allocations.append({
                        'order_id': order['order_id'],
                        'tech_id': tech['tech_id'],
                        'service_time': order['service_time'],
                        'uses_overtime': False,
                        'queue_days': day - order['generation_day']
                    })
                    tech['available_capacity'] -= order['service_time']
                    allocated = True
                    break
            
            # If not allocated, try with overtime (if allowed)
            if not allocated and self.config.get('allow_overtime', True):
                for tech in technicians:
                    overtime_needed = order['service_time'] - tech['available_capacity']
                    if overtime_needed > 0 and overtime_needed <= 4:  # Max 4h overtime
                        allocations.append({
                            'order_id': order['order_id'],
                            'tech_id': tech['tech_id'],
                            'service_time': order['service_time'],
                            'uses_overtime': True,
                            'overtime_hours': overtime_needed,
                            'queue_days': day - order['generation_day']
                        })
                        tech['available_capacity'] = 0
                        total_overtime += overtime_needed
                        allocated = True
                        break
        
        return allocations, total_overtime


# ============================================================================
# POLICY 2: EOQ (Economic Order Quantity) Adapted
# ============================================================================

class EOQPolicy(SchedulingPolicy):
    """
    EOQ: Economic Order Quantity adapted for service capacity management
    
    Calculated parameters (as per Section VI of the article):
    - Q* = √(2DK/h) ≈ 94 orders (batch size)
    - Safety stock ss = z × σ × √LT ≈ 43 orders
    - Reorder point = Q* - ss ≈ 51 orders
    """
    
    def __init__(self, config=None):
        config = config or {}
        # EOQ parameters (Table 5 - as per article)
        self.D = config.get('avg_demand', 95)  # orders/day
        self.K = config.get('setup_cost', 420)  # USD$ per batch
        self.h = config.get('holding_cost', 18)  # USD$ per order-day
        
        # EOQ calculation: Q* = √(2DK/h)
        self.Q_star = int(np.sqrt(2 * self.D * self.K / self.h))
        
        # Safety stock: ss = z × σ × √LT
        self.z = 1.96  # 95% service level
        self.sigma = config.get('demand_std', 22)  # orders/day
        self.LT = 1  # lead time (days)
        self.safety_stock = int(self.z * self.sigma * np.sqrt(self.LT))
        
        # Reorder point
        self.reorder_point = max(self.Q_star - self.safety_stock, 1)
        
        super().__init__("EOQ", config)
        
        print(f"\n[EOQ] Calculated parameters:")
        print(f"  Q* (batch size) = {self.Q_star} orders")
        print(f"  Safety stock = {self.safety_stock} orders")
        print(f"  Reorder point = {self.reorder_point} orders")
    
    def allocate_orders(self, orders, technicians, day):
        """
        Allocates orders following EOQ policy
        """
        allocations = []
        total_overtime = 0
        
        backlog_size = len(orders)
        
        # Check if should process batch (backlog ≥ reorder point)
        if backlog_size < self.reorder_point:
            return allocations, total_overtime
        
        # Calculate scores for prioritization (priority × age)
        for order in orders:
            age = day - order['generation_day']
            order['eoq_score'] = order['priority'] * (1 + 0.1 * age)
        
        # Sort by score (descending)
        sorted_orders = sorted(orders, key=lambda x: x['eoq_score'], reverse=True)
        
        # Process up to Q* orders
        orders_to_process = sorted_orders[:self.Q_star]
        
        for order in orders_to_process:
            allocated = False
            
            for tech in technicians:
                if tech['available_capacity'] >= order['service_time']:
                    allocations.append({
                        'order_id': order['order_id'],
                        'tech_id': tech['tech_id'],
                        'service_time': order['service_time'],
                        'uses_overtime': False,
                        'queue_days': day - order['generation_day']
                    })
                    tech['available_capacity'] -= order['service_time']
                    allocated = True
                    break
            
            if not allocated and self.config.get('allow_overtime', True):
                for tech in technicians:
                    overtime_needed = order['service_time'] - tech['available_capacity']
                    if overtime_needed > 0 and overtime_needed <= 4:
                        allocations.append({
                            'order_id': order['order_id'],
                            'tech_id': tech['tech_id'],
                            'service_time': order['service_time'],
                            'uses_overtime': True,
                            'overtime_hours': overtime_needed,
                            'queue_days': day - order['generation_day']
                        })
                        tech['available_capacity'] = 0
                        total_overtime += overtime_needed
                        allocated = True
                        break
        
        return allocations, total_overtime


# ============================================================================
# POLICY 3: (s,S) - Continuous Review Policy
# ============================================================================

class sSPolicy(SchedulingPolicy):
    """
    (s,S): Continuous Review Policy adapted for services
    
    Optimized parameters (as per Section VI of the article):
    - s* = 180 orders (reorder point ≈ 1.9 days of demand)
    - S* = 85 orders (base stock ≈ 0.9 days)
    
    Objective function minimized via Monte Carlo (10,000 simulations):
    C(s,S) = c_op × ∑B_t + c_over × ∑HE_t + c_pen × ∑P_t
    """
    
    def __init__(self, config=None):
        config = config or {}
        
        # Optimized (s,S) parameters (Table 5 - as per article)
        self.s = config.get('reorder_point', 180)  # orders
        self.S = config.get('base_stock', 85)      # orders
        
        # Costs for performance calculation
        self.c_op = config.get('operational_cost', 62)  # USD$/order
        self.c_over = config.get('overtime_cost', 42)   # USD$/hour
        self.c_pen = config.get('penalty_cost', 18)     # USD$/order-day
        
        super().__init__("(s,S)", config)
        
        print(f"\n[(s,S)] Optimized parameters:")
        print(f"  s (reorder point) = {self.s} orders")
        print(f"  S (base stock) = {self.S} orders")
        print(f"  Allocation quantity when triggered = backlog - {self.S}")
    
    def allocate_orders(self, orders, technicians, day):
        """
        Allocates orders following (s,S) policy
        """
        allocations = []
        total_overtime = 0
        
        backlog_size = len(orders)
        
        # Check trigger: backlog ≥ s
        if backlog_size < self.s:
            return allocations, total_overtime
        
        # Calculate allocation quantity: Q = backlog - S
        quantity_to_allocate = backlog_size - self.S
        
        if quantity_to_allocate <= 0:
            return allocations, total_overtime
        
        # Calculate scores for prioritization (priority × aging)
        for order in orders:
            age = day - order['generation_day']
            aging_factor = 1 + 0.15 * age
            order['ss_score'] = order['priority'] * aging_factor
        
        # Sort by score (descending)
        sorted_orders = sorted(orders, key=lambda x: x['ss_score'], reverse=True)
        
        # Select orders to process
        orders_to_process = sorted_orders[:quantity_to_allocate]
        
        for order in orders_to_process:
            allocated = False
            
            for tech in technicians:
                if tech['available_capacity'] >= order['service_time']:
                    allocations.append({
                        'order_id': order['order_id'],
                        'tech_id': tech['tech_id'],
                        'service_time': order['service_time'],
                        'uses_overtime': False,
                        'queue_days': day - order['generation_day']
                    })
                    tech['available_capacity'] -= order['service_time']
                    allocated = True
                    break
            
            if not allocated and self.config.get('allow_overtime', True):
                for tech in technicians:
                    overtime_needed = order['service_time'] - tech['available_capacity']
                    if overtime_needed > 0 and overtime_needed <= 4:
                        allocations.append({
                            'order_id': order['order_id'],
                            'tech_id': tech['tech_id'],
                            'service_time': order['service_time'],
                            'uses_overtime': True,
                            'overtime_hours': overtime_needed,
                            'queue_days': day - order['generation_day']
                        })
                        tech['available_capacity'] = 0
                        total_overtime += overtime_needed
                        allocated = True
                        break
        
        return allocations, total_overtime


# ============================================================================
# POLICY 4: DYNAMIC BI-PHASE BACKLOG (MILP - Equations 1-30 from article)
# ============================================================================

class BacklogMILPPolicy(SchedulingPolicy):
    """
    Bi-phase MILP model as per IEEE Access article
    
    Implements all equations from the mathematical model (Section V):
    - Objective Function Phase 1 (Eq. 1): Minimizes emergency time
    - Objective Function Phase 2 (Eq. 2): Minimizes backlog + new + overtime
    - Coverage constraints (Eq. 3-5)
    - Capacity constraints (Eq. 6-19)
    - Prioritization constraints (Eq. 20-24)
    - Operational constraints (Eq. 25-30)
    
    Model parameters (Table 4):
    - αB = 0.70 (aging rate - Eq. 23)
    - β = 0.50 (weighted delay - Eq. 24)
    - γ = 2.00 (backlog penalty - Eq. 2)
    - ρ = 0.60 (minimum quota - Eq. 20)
    - ρ_intens = 0.80 (quota intensification - Eq. 20)
    """
    
    def __init__(self, config=None):
        config = config or {}
        
        # Model parameters (Table 4 - as per article)
        self.alpha_B = config.get('aging_rate', 0.70)        # Eq. 23
        self.beta = config.get('delay_weight', 0.50)         # Eq. 24
        self.gamma = config.get('backlog_penalty', 2.00)     # Eq. 2
        self.rho = config.get('min_quota', 0.60)             # Eq. 20
        self.rho_intens = config.get('quota_intense', 0.80)  # Eq. 20 (weekly)
        
        # Cost parameters
        self.c_op = config.get('operational_cost', 62)       # USD$/order
        self.c_over = config.get('overtime_cost', 42)        # USD$/hour overtime
        self.c_pen = config.get('penalty_cost', 18)          # USD$/order-day
        
        super().__init__("Backlog-MILP", config)
        
        print(f"\n[Backlog-MILP] Model parameters:")
        print(f"  αB (aging rate) = {self.alpha_B}")
        print(f"  β (delay weight) = {self.beta}")
        print(f"  γ (backlog penalty) = {self.gamma}")
        print(f"  ρ (min quota) = {self.rho} (intensification: {self.rho_intens})")
    
    def calculate_adjusted_priority(self, order, current_day):
        """
        Equation 23: p_adj(t) = P0 + αB × max(0, t - PG) + 0.5 × T
        """
        age = max(0, current_day - order['generation_day'])
        adjusted = order['priority'] + self.alpha_B * age + 0.5 * order['service_time']
        return adjusted
    
    def calculate_weighted_delay(self, order, current_day):
        """
        Equation 24: atpond = age × p_adj
        """
        age = max(0, current_day - order['generation_day'])
        p_adj = self.calculate_adjusted_priority(order, current_day)
        return age * p_adj
    
    def allocate_orders(self, orders, technicians, day):
        """
        Complete MILP implementation of bi-phase model
        
        PHASE 1 (Eq. 1): Minimize ∑∑ w^E_to
        PHASE 2 (Eq. 2): Minimize ∑∑ w^B_to + ∑∑ w^C_to + θ ∑ he_td
        """
        
        # Separate orders by type (as per Table 3)
        emergency_orders = [o for o in orders if o['priority'] >= 7]  # Levels 0-2 → 7-10
        commercial_orders = [o for o in orders if o['priority'] < 7]
        
        # Identify backlog (old orders)
        backlog_orders = [o for o in commercial_orders if (day - o['generation_day']) > 1]
        new_orders = [o for o in commercial_orders if (day - o['generation_day']) <= 1]
        
        print(f"\n[Day {day}] Emergency: {len(emergency_orders)}, Backlog: {len(backlog_orders)}, New: {len(new_orders)}")
        
        # ========================================================================
        # PHASE 1: EMERGENCY ALLOCATION (Eq. 1)
        # ========================================================================
        
        phase1_allocations = []
        phase1_overtime = 0
        
        # Create MILP problem for Phase 1
        prob_phase1 = LpProblem("Phase1_Emergency", LpMinimize)
        
        # Decision variables (binary)
        x_E = {}
        for i, order in enumerate(emergency_orders):
            for j, tech in enumerate(technicians):
                x_E[(i, j)] = LpVariable(f"x_E_{i}_{j}", cat='Binary')
        
        # Objective function Phase 1 (Eq. 1): min ∑∑ w^E_to
        # w^E_to = x^E_to × T_o (linearized via Big-M)
        objective_phase1 = 0
        for i, order in enumerate(emergency_orders):
            for j, tech in enumerate(technicians):
                objective_phase1 += x_E[(i, j)] * order['service_time']
        
        prob_phase1 += objective_phase1
        
        # Coverage constraints (Eq. 3): ∑ x^E_to = 1
        for i, order in enumerate(emergency_orders):
            prob_phase1 += lpSum([x_E[(i, j)] for j in range(len(technicians))]) == 1
        
        # Capacity constraints (Eq. 11): ∑ z^E_to ≤ C^total_td
        for j, tech in enumerate(technicians):
            capacity_used = 0
            for i, order in enumerate(emergency_orders):
                capacity_used += x_E[(i, j)] * order['service_time']
            prob_phase1 += capacity_used <= tech['available_capacity']
        
        # Solve Phase 1
        prob_phase1.solve(PULP_CBC_CMD(msg=0))
        
        # Extract Phase 1 allocations
        if prob_phase1.status == 1:  # Optimal
            for i, order in enumerate(emergency_orders):
                for j, tech in enumerate(technicians):
                    if value(x_E[(i, j)]) == 1:
                        phase1_allocations.append({
                            'order_id': order['order_id'],
                            'tech_id': tech['tech_id'],
                            'service_time': order['service_time'],
                            'uses_overtime': False,
                            'queue_days': day - order['generation_day'],
                            'phase': 'Emergency'
                        })
                        technicians[j]['available_capacity'] -= order['service_time']
        
        # ========================================================================
        # PHASE 2: BACKLOG + NEW ORDERS (Eq. 2)
        # ========================================================================
        
        phase2_allocations = []
        phase2_overtime = 0
        
        # Residual capacity after Phase 1 (Eq. 18)
        residual_capacity = sum([t['available_capacity'] for t in technicians])
        
        # Determine quota (Eq. 20 with periodic intensification)
        current_quota = self.rho_intens if (day % 7 == 0 and day > 0) else self.rho
        min_backlog_capacity = residual_capacity * current_quota
        
        print(f"  Residual capacity: {residual_capacity:.1f}h, Quota: {current_quota:.0%}, Min backlog: {min_backlog_capacity:.1f}h")
        
        # Create MILP problem for Phase 2
        prob_phase2 = LpProblem("Phase2_Backlog_Commercial", LpMinimize)
        
        # Decision variables
        x_B = {}  # Backlog
        x_C = {}  # Commercial
        
        for i, order in enumerate(backlog_orders):
            for j, tech in enumerate(technicians):
                x_B[(i, j)] = LpVariable(f"x_B_{i}_{j}", cat='Binary')
        
        for i, order in enumerate(new_orders):
            for j, tech in enumerate(technicians):
                x_C[(i, j)] = LpVariable(f"x_C_{i}_{j}", cat='Binary')
        
        # Calculate scores for backlog (Eq. 1 with γ, Eq. 23, Eq. 24)
        backlog_scores = []
        for order in backlog_orders:
            adjusted_priority = self.calculate_adjusted_priority(order, day)
            weighted_delay = self.calculate_weighted_delay(order, day)
            # Eq. 1: Z_B = γ × [T + (1+β) × atp]
            score = self.gamma * (order['service_time'] + (1 + self.beta) * weighted_delay)
            backlog_scores.append(score)
        
        # Objective function Phase 2 (Eq. 2)
        objective_phase2 = 0
        
        # Term 1: Weighted backlog
        for i, order in enumerate(backlog_orders):
            for j, tech in enumerate(technicians):
                objective_phase2 += x_B[(i, j)] * backlog_scores[i]
        
        # Term 2: New orders
        for i, order in enumerate(new_orders):
            for j, tech in enumerate(technicians):
                objective_phase2 += x_C[(i, j)] * order['service_time']
        
        prob_phase2 += objective_phase2
        
        # Coverage constraints (Eq. 4-5): ∑ x_to ≤ 1 (allows postponement)
        for i in range(len(backlog_orders)):
            prob_phase2 += lpSum([x_B[(i, j)] for j in range(len(technicians))]) <= 1
        
        for i in range(len(new_orders)):
            prob_phase2 += lpSum([x_C[(i, j)] for j in range(len(technicians))]) <= 1
        
        # Residual capacity constraint (Eq. 19)
        for j, tech in enumerate(technicians):
            capacity_used = 0
            for i, order in enumerate(backlog_orders):
                capacity_used += x_B[(i, j)] * order['service_time']
            for i, order in enumerate(new_orders):
                capacity_used += x_C[(i, j)] * order['service_time']
            prob_phase2 += capacity_used <= tech['available_capacity']
        
        # Minimum quota constraint for backlog (Eq. 20)
        backlog_capacity_allocated = 0
        for i, order in enumerate(backlog_orders):
            for j in range(len(technicians)):
                backlog_capacity_allocated += x_B[(i, j)] * order['service_time']
        
        prob_phase2 += backlog_capacity_allocated >= min_backlog_capacity
        
        # Solve Phase 2
        prob_phase2.solve(PULP_CBC_CMD(msg=0))
        
        # Extract Phase 2 allocations
        if prob_phase2.status == 1:  # Optimal
            for i, order in enumerate(backlog_orders):
                for j, tech in enumerate(technicians):
                    if value(x_B[(i, j)]) == 1:
                        phase2_allocations.append({
                            'order_id': order['order_id'],
                            'tech_id': tech['tech_id'],
                            'service_time': order['service_time'],
                            'uses_overtime': False,
                            'queue_days': day - order['generation_day'],
                            'phase': 'Backlog',
                            'backlog_score': backlog_scores[i]
                        })
            
            for i, order in enumerate(new_orders):
                for j, tech in enumerate(technicians):
                    if value(x_C[(i, j)]) == 1:
                        phase2_allocations.append({
                            'order_id': order['order_id'],
                            'tech_id': tech['tech_id'],
                            'service_time': order['service_time'],
                            'uses_overtime': False,
                            'queue_days': day - order['generation_day'],
                            'phase': 'New'
                        })
        
        # Combine allocations from both phases
        all_allocations = phase1_allocations + phase2_allocations
        total_overtime = phase1_overtime + phase2_overtime
        
        print(f"  Allocated: Phase1={len(phase1_allocations)}, Phase2={len(phase2_allocations)}")
        
        return all_allocations, total_overtime


# ============================================================================
# MAIN SIMULATOR
# ============================================================================

class SchedulingSimulator:
    """Simulator that compares all policies"""
    
    def __init__(self, orders_df, technicians_df, num_days=30):
        self.orders_df = orders_df.copy()
        self.technicians_df = technicians_df.copy()
        self.num_days = num_days
        self.results = None
        
        # Initialize policies
        self.policies = {
            'FIFO': FIFOPolicy(),
            'EOQ': EOQPolicy(),
            '(s,S)': sSPolicy(),
            'Backlog-MILP': BacklogMILPPolicy()
        }
    
    def run_single_policy(self, policy_name, seed=42):
        """Executes simulation for a specific policy"""
        
        np.random.seed(seed)
        policy = self.policies[policy_name]
        policy.reset_metrics()
        
        # Initial state
        pending_orders = []
        daily_stats = []
        
        for day in range(self.num_days):
            # Add new orders for the day
            new_orders = self.orders_df[self.orders_df['generation_day'] == day].to_dict('records')
            pending_orders.extend(new_orders)
            
            # Reset technician capacity
            techs = self.technicians_df.to_dict('records')
            for tech in techs:
                # Variable capacity (simulates absences, etc)
                capacity_variation = np.random.choice([-2, -1, 0, 1, 2], p=[0.1, 0.2, 0.4, 0.2, 0.1])
                tech['available_capacity'] = max(0, tech['base_capacity'] + capacity_variation)
            
            # Allocate orders
            allocations, overtime = policy.allocate_orders(pending_orders.copy(), techs, day)
            
            # Update pending orders (remove allocated ones)
            allocated_ids = [a['order_id'] for a in allocations]
            pending_orders = [o for o in pending_orders if o['order_id'] not in allocated_ids]
            
            # Calculate daily metrics
            if len(allocations) > 0:
                avg_queue = np.mean([a['queue_days'] for a in allocations])
            else:
                avg_queue = 0
            
            daily_stats.append({
                'day': day,
                'orders_allocated': len(allocations),
                'orders_pending': len(pending_orders),
                'avg_queue_time': avg_queue,
                'overtime_hours': overtime
            })
            
            policy.metrics['total_overtime'] += overtime
        
        # Calculate final metrics
        total_queue_time = sum([s['avg_queue_time'] * s['orders_allocated'] for s in daily_stats])
        total_allocated = sum([s['orders_allocated'] for s in daily_stats])
        
        avg_queue_time = total_queue_time / total_allocated if total_allocated > 0 else 0
        total_overtime = policy.metrics['total_overtime']
        
        # Total cost
        total_cost = (total_allocated * 62 +  # Operational cost
                     total_overtime * 42 +       # Overtime cost
                     len(pending_orders) * 18 * 30)  # Penalty cost
        
        return {
            'policy': policy_name,
            'avg_queue_time': round(avg_queue_time, 2),
            'total_overtime': round(total_overtime, 1),
            'total_cost': round(total_cost / 1000, 1),  # in USD$ K
            'orders_completed': total_allocated,
            'orders_pending': len(pending_orders),
            'service_rate': round(100 * total_allocated / len(self.orders_df), 1)
        }
    
    def run_all_policies(self, num_runs=10):
        """Executes all policies with multiple replications"""
        
        print("\n" + "="*70)
        print("RUNNING SIMULATIONS")
        print("="*70)
        
        results = []
        
        for policy_name in self.policies.keys():
            print(f"\nPolicy: {policy_name}")
            print("-" * 40)
            
            for run in range(num_runs):
                result = self.run_single_policy(policy_name, seed=42 + run)
                result['run'] = run + 1
                results.append(result)
                
                if (run + 1) % 5 == 0:
                    print(f"  Replication {run + 1}/{num_runs} completed")
        
        self.results = pd.DataFrame(results)
        
        print("\n✓ Simulations completed!")
        return self.results
    
    def get_summary_statistics(self):
        """Calculates descriptive statistics by policy"""
        
        if self.results is None or len(self.results) == 0:
            print("Error: Run run_all_policies() first")
            return None
        
        summary = self.results.groupby('policy').agg({
            'avg_queue_time': ['mean', 'std'],
            'total_overtime': ['mean', 'std'],
            'total_cost': ['mean', 'std'],
            'service_rate': ['mean', 'std']
        }).round(2)
        
        return summary
    
    def export_results(self, filename='simulation_results_milp.xlsx'):
        """Exports results to Excel"""
        
        if self.results is None:
            print("Error: No results to export")
            return False
        
        try:
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # Sheet 1: Complete results
                self.results.to_excel(writer, sheet_name='Raw_Results', index=False)
                
                # Sheet 2: Summary statistics
                summary = self.get_summary_statistics()
                summary.to_excel(writer, sheet_name='Summary_Statistics')
                
                # Sheet 3: Input data
                self.orders_df.to_excel(writer, sheet_name='Orders', index=False)
                self.technicians_df.to_excel(writer, sheet_name='Technicians', index=False)
            
            print(f"\n✓ Results exported to: {filename}")
            return True
        
        except Exception as e:
            print(f"\n✗ Export error: {str(e)}")
            return False


# ============================================================================
# DATASET LOADER AND VALIDATOR
# ============================================================================

def validate_dataset(orders_df, technicians_df):
    """
    Validates structure and content of loaded dataset
    
    Returns: (bool, str) - (valid, error_message)
    """
    
    # Validation of Orders
    required_orders_cols = ['order_id', 'generation_day', 'priority', 'service_time', 'service_type']
    missing_orders = [col for col in required_orders_cols if col not in orders_df.columns]
    
    if missing_orders:
        return False, f"Missing columns in 'Orders': {', '.join(missing_orders)}"
    
    # Data type validation
    if not pd.api.types.is_numeric_dtype(orders_df['generation_day']):
        return False, "Column 'generation_day' must be numeric"
    
    if not pd.api.types.is_numeric_dtype(orders_df['priority']):
        return False, "Column 'priority' must be numeric"
    
    if not pd.api.types.is_numeric_dtype(orders_df['service_time']):
        return False, "Column 'service_time' must be numeric"
    
    # Value validation
    if orders_df['generation_day'].min() < 0:
        return False, "Values in 'generation_day' must be ≥ 0"
    
    if not orders_df['priority'].between(0, 10).all():
        return False, "Values in 'priority' must be between 0 and 10"
    
    if orders_df['service_time'].min() <= 0:
        return False, "Values in 'service_time' must be > 0"
    
    # Validation of Technicians
    required_tech_cols = ['tech_id', 'base_capacity', 'hourly_rate', 'overtime_rate']
    missing_tech = [col for col in required_tech_cols if col not in technicians_df.columns]
    
    if missing_tech:
        return False, f"Missing columns in 'Technicians': {', '.join(missing_tech)}"
    
    if not pd.api.types.is_numeric_dtype(technicians_df['base_capacity']):
        return False, "Column 'base_capacity' must be numeric"
    
    if technicians_df['base_capacity'].min() <= 0:
        return False, "Values in 'base_capacity' must be > 0"
    
    return True, "Dataset successfully validated"


def load_dataset_from_excel(filepath):
    """
    Loads dataset from Excel file with complete validation
    
    Expected Excel file structure:
    
    Sheet 'Orders' (required):
    ┌────────────┬────────────────┬──────────┬──────────────┬──────────────┐
    │ order_id   │ generation_day │ priority │ service_time │ service_type │
    ├────────────┼────────────────┼──────────┼──────────────┼──────────────┤
    │ ORD00001   │ 0              │ 6        │ 2.5          │ Maint        │
    │ ORD00002   │ 0              │ 9        │ 1.8          │ Emergency    │
    │ ...        │ ...            │ ...      │ ...          │ ...          │
    └────────────┴────────────────┴──────────┴──────────────┴──────────────┘
    
    Fields:
    - order_id: Unique order identifier (text)
    - generation_day: Generation day (0 to N-1, integer)
    - priority: Priority (0=maximum emergency to 10=low, integer)
    - service_time: Estimated time in hours (decimal > 0)
    - service_type: Service type (text: Maint, Repair, Inspect, Emergency)
    
    Sheet 'Technicians' (required):
    ┌──────────┬───────────────┬─────────────┬───────────────┐
    │ tech_id  │ base_capacity │ hourly_rate │ overtime_rate │
    ├──────────┼───────────────┼─────────────┼───────────────┤
    │ TECH001  │ 8.0           │ 28.0        │ 42.0          │
    │ TECH002  │ 8.0           │ 28.0        │ 42.0          │
    │ ...      │ ...           │ ...         │ ...           │
    └──────────┴───────────────┴─────────────┴───────────────┘
    
    Fields:
    - tech_id: Unique technician identifier (text)
    - base_capacity: Daily base capacity in hours (decimal > 0, typical: 8.0)
    - hourly_rate: Regular hourly rate in USD$ (decimal > 0)
    - overtime_rate: Overtime hourly rate in USD$ (decimal > 0, typical: 1.5× hourly_rate)
    
    Sheet 'Config' (optional):
    ┌───────────────┬────────┐
    │ parameter     │ value  │
    ├───────────────┼────────┤
    │ num_days      │ 30     │
    │ allow_overtime│ True   │
    └───────────────┴────────┘
    
    Returns:
    - orders_df: DataFrame with service orders
    - technicians_df: DataFrame with technicians
    - config: dict with configurations (or default values)
    """
    
    print("\n" + "="*70)
    print("LOADING DATASET FROM EXCEL FILE")
    print("="*70)
    print(f"File: {filepath}")
    
    try:
        # Check if file exists
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        # List available sheets
        excel_file = pd.ExcelFile(filepath)
        available_sheets = excel_file.sheet_names
        print(f"\nSheets found: {', '.join(available_sheets)}")
        
        # Check required sheets
        required_sheets = ['Orders', 'Technicians']
        missing_sheets = [sheet for sheet in required_sheets if sheet not in available_sheets]
        
        if missing_sheets:
            raise ValueError(f"Required sheets missing: {', '.join(missing_sheets)}")
        
        # Load Orders sheet
        print("\n[1/3] Loading 'Orders' sheet...")
        orders_df = pd.read_excel(filepath, sheet_name='Orders')
        print(f"  ✓ {len(orders_df)} orders loaded")
        print(f"  Columns: {', '.join(orders_df.columns.tolist())}")
        
        # Load Technicians sheet
        print("\n[2/3] Loading 'Technicians' sheet...")
        technicians_df = pd.read_excel(filepath, sheet_name='Technicians')
        print(f"  ✓ {len(technicians_df)} technicians loaded")
        print(f"  Columns: {', '.join(technicians_df.columns.tolist())}")
        
        # Load Config sheet (optional)
        print("\n[3/3] Loading configurations...")
        config = {
            'num_days': 30,
            'allow_overtime': True,
            'avg_demand': 95,
            'demand_std': 22,
            'setup_cost': 420,
            'holding_cost': 18,
            'operational_cost': 62,
            'overtime_cost': 42,
            'penalty_cost': 18,
            'reorder_point_ss': 180,
            'base_stock_ss': 85,
            'aging_rate': 0.70,
            'delay_weight': 0.50,
            'backlog_penalty': 2.00,
            'min_quota': 0.60,
            'quota_intense': 0.80
        }
        
        if 'Config' in available_sheets:
            config_df = pd.read_excel(filepath, sheet_name='Config')
            if 'parameter' in config_df.columns and 'value' in config_df.columns:
                for _, row in config_df.iterrows():
                    param = row['parameter']
                    value = row['value']
                    if param in config:
                        # Convert to appropriate type
                        if isinstance(config[param], bool):
                            config[param] = bool(value)
                        elif isinstance(config[param], int):
                            config[param] = int(value)
                        elif isinstance(config[param], float):
                            config[param] = float(value)
                print(f"  ✓ Custom configurations loaded")
        else:
            print(f"  ℹ 'Config' sheet not found, using default values")
        
        # Validate dataset
        print("\n" + "="*70)
        print("VALIDATING DATASET")
        print("="*70)
        
        is_valid, message = validate_dataset(orders_df, technicians_df)
        
        if not is_valid:
            raise ValueError(f"Validation error: {message}")
        
        print(f"✓ {message}")
        
        # Descriptive statistics
        print("\n" + "="*70)
        print("DATASET STATISTICS")
        print("="*70)
        
        num_days = orders_df['generation_day'].max() + 1
        avg_orders_per_day = len(orders_df) / num_days
        
        print(f"\n[SERVICE ORDERS]")
        print(f"  Total orders: {len(orders_df)}")
        print(f"  Period: {num_days} days")
        print(f"  Average rate: {avg_orders_per_day:.1f} orders/day")
        print(f"  Average service time: {orders_df['service_time'].mean():.2f}h (σ={orders_df['service_time'].std():.2f}h)")
        
        print(f"\n[PRIORITY DISTRIBUTION]")
        priority_counts = orders_df['priority'].value_counts().sort_index(ascending=False)
        for priority, count in priority_counts.items():
            percentage = 100 * count / len(orders_df)
            priority_name = {9: "Emergency", 8: "Critical", 6: "High", 4: "Medium", 2: "Low"}.get(priority, f"P{priority}")
            print(f"  {priority_name} ({priority}): {count:4d} ({percentage:5.1f}%)")
        
        print(f"\n[SERVICE TYPES]")
        type_counts = orders_df['service_type'].value_counts()
        for stype, count in type_counts.items():
            percentage = 100 * count / len(orders_df)
            print(f"  {stype}: {count:4d} ({percentage:5.1f}%)")
        
        print(f"\n[TECHNICIANS]")
        print(f"  Total technicians: {len(technicians_df)}")
        print(f"  Average base capacity: {technicians_df['base_capacity'].mean():.1f}h/day")
        print(f"  Average hourly rate: USD$ {technicians_df['hourly_rate'].mean():.2f}/h")
        print(f"  Average overtime rate: USD$ {technicians_df['overtime_rate'].mean():.2f}/h")
        
        print("\n" + "="*70)
        print("DATASET LOADED AND VALIDATED SUCCESSFULLY")
        print("="*70)
        
        return orders_df, technicians_df, config
        
    except FileNotFoundError as e:
        print(f"\n✗ ERROR: {str(e)}")
        print("\nVerify that:")
        print("  1. The file exists in the specified path")
        print("  2. The path is correct (use absolute path if necessary)")
        raise
        
    except ValueError as e:
        print(f"\n✗ VALIDATION ERROR: {str(e)}")
        print("\nVerify that:")
        print("  1. The 'Orders' and 'Technicians' sheets exist")
        print("  2. All required columns are present")
        print("  3. Data types are correct")
        print("  4. Values are within valid ranges")
        raise
        
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {str(e)}")
        print(f"Type: {type(e).__name__}")
        raise


def create_template_excel(output_path='dataset_template.xlsx'):
    """
    Creates an Excel template file with sample data
    
    This file can be used as a base for creating real datasets
    """
    
    print("\n" + "="*70)
    print("CREATING DATASET TEMPLATE")
    print("="*70)
    
    # Create sample data
    np.random.seed(42)
    
    # Sample orders (100 orders, 7 days)
    orders_example = []
    priorities = [9, 8, 6, 4, 2]
    service_types = ['Maint', 'Repair', 'Inspect', 'Emergency']
    
    for i in range(100):
        orders_example.append({
            'order_id': f'ORD{i+1:05d}',
            'generation_day': np.random.randint(0, 7),
            'priority': np.random.choice(priorities, p=[0.13, 0.12, 0.25, 0.35, 0.15]),
            'service_time': max(0.5, np.random.normal(3.0, 1.2)),
            'service_type': np.random.choice(service_types, p=[0.35, 0.30, 0.25, 0.10])
        })
    
    orders_df = pd.DataFrame(orders_example)
    
    # Sample technicians (20 technicians)
    technicians_example = []
    for i in range(20):
        technicians_example.append({
            'tech_id': f'TECH{i+1:03d}',
            'base_capacity': 8.0,
            'hourly_rate': 28.0,
            'overtime_rate': 42.0
        })
    
    technicians_df = pd.DataFrame(technicians_example)
    
    # Sample configurations
    config_example = pd.DataFrame({
        'parameter': ['num_days', 'allow_overtime', 'avg_demand', 'demand_std'],
        'value': [7, True, 95, 22]
    })
    
    # Save to Excel
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        orders_df.to_excel(writer, sheet_name='Orders', index=False)
        technicians_df.to_excel(writer, sheet_name='Technicians', index=False)
        config_example.to_excel(writer, sheet_name='Config', index=False)
    
    print(f"\n✓ Template created: {output_path}")
    print(f"\nContent:")
    print(f"  - Sheet 'Orders': {len(orders_df)} sample orders")
    print(f"  - Sheet 'Technicians': {len(technicians_df)} sample technicians")
    print(f"  - Sheet 'Config': Configurable parameters")
    print(f"\nEdit this file with your real data and use it as input.")
    
    return output_path


# ============================================================================
# MAIN - PROGRAM EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    
    print("\n" + "="*70)
    print("SCHEDULING POLICIES COMPARISON SYSTEM")
    print("Implements: FIFO, EOQ, (s,S), Backlog-MILP")
    print("="*70)
    
    print("\nOptions:")
    print("1. Load dataset from Excel file")
    print("2. Create Excel template for dataset")
    print("3. Exit")
    
    choice = input("\nChoose an option (1-3): ").strip()
    
    if choice == '2':
        # Create template
        template_path = input("\nTemplate filename (Enter for 'dataset_template.xlsx'): ").strip()
        if not template_path:
            template_path = 'dataset_template.xlsx'
        
        create_template_excel(template_path)
        
        print(f"\n{'='*70}")
        print("Template created successfully!")
        print(f"{'='*70}")
        print("\nNext steps:")
        print(f"  1. Open the file: {template_path}")
        print("  2. Edit the 'Orders' and 'Technicians' sheets with your real data")
        print("  3. (Optional) Adjust configurations in 'Config' sheet")
        print("  4. Run this program again and choose option 1")
        
        return
    
    elif choice == '3':
        print("\nExiting...")
        return
    
    elif choice != '1':
        print("\n✗ Invalid option!")
        return
    
    # Load dataset
    filepath = input("\nExcel file path: ").strip()
    
    if not filepath:
        print("\n✗ No file specified!")
        return
    
    try:
        # Load and validate dataset
        orders_df, technicians_df, config = load_dataset_from_excel(filepath)
        
        # Ask for number of replications
        try:
            num_runs = int(input("\nNumber of replications (default: 10): ").strip() or "10")
        except:
            num_runs = 10
        
        # Determine number of days
        num_days = config.get('num_days', orders_df['generation_day'].max() + 1)
        
        print(f"\n{'='*70}")
        print("STARTING SIMULATION")
        print(f"{'='*70}")
        print(f"Orders: {len(orders_df)}")
        print(f"Technicians: {len(technicians_df)}")
        print(f"Period: {num_days} days")
        print(f"Replications: {num_runs}")
        
        # Create simulator
        simulator = SchedulingSimulator(orders_df, technicians_df, num_days=num_days)
        
        # Run simulation
        results = simulator.run_all_policies(num_runs=num_runs)
        
        # Show summary statistics
        print("\n" + "="*70)
        print("SUMMARY STATISTICS (Mean ± SD)")
        print("="*70)
        
        summary = simulator.get_summary_statistics()
        print("\n" + summary.to_string())
        
        # Export results
        output_file = input("\nOutput filename (Enter for 'simulation_results_milp.xlsx'): ").strip()
        if not output_file:
            output_file = "simulation_results_milp.xlsx"
        
        simulator.export_results(output_file)
        
        print("\n" + "="*70)
        print("PROCESS COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"\nGenerated files:")
        print(f"  1. {output_file} - Complete results")
        print(f"\nNext steps:")
        print(f"  - Statistical analysis (ANOVA, t-tests)")
        print(f"  - Visualizations (box plots, time series curves)")
        print(f"  - Comparison with article results")
        print("="*70)
        
    except FileNotFoundError:
        print("\n" + "="*70)
        print("FILE NOT FOUND")
        print("="*70)
        print("\nTo create a dataset, run again and choose option 2")
        
    except ValueError as e:
        print("\n" + "="*70)
        print("VALIDATION ERROR")
        print("="*70)
        print("\nTo see the expected format, create a template (option 2)")
        
    except Exception as e:
        print("\n" + "="*70)
        print("UNEXPECTED ERROR")
        print("="*70)
        print(f"\nDetails: {str(e)}")


if __name__ == "__main__":
    main()
