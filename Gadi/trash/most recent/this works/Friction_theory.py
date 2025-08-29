# Friction_theory.py
import numpy as np
# from frictionweertman import frictionweertman
import matplotlib.pyplot as plt
import numpy as np

class FrictionTheory:
    def __init__(self, config):
        self.config = config
        self.m = getattr(config, 'friction_m', 1.0)
        self.bed_slope = getattr(config, 'base_slope', -0.015)

    def get_basal_velocity(self, md):
        """Return (vx_basal, basal_node_indices)."""
        basal_nodes = np.where(md.mesh.vertexflags(1))[0]
        # Try steady‐state result first
        sol = getattr(md.results, 'StressbalanceSolution', None)
        if sol is not None and hasattr(sol, 'Vx'):
            vx = sol.Vx.flatten()
        else:
            vx = md.initialization.vx

        return np.abs(vx[basal_nodes]), basal_nodes

    def calculate_basal_friction(self, md):
        """Implement Pattyn 2008 style friction."""
        yts = self.config.yts  # seconds per year
        # md.friction = frictionweertman()
        # md.friction.linearize = 0

        ne = md.mesh.numberofelements
        nv = md.mesh.numberofvertices

        md.friction.p = np.ones(ne)
        md.friction.q = np.ones(ne)
        # md.friction.m = np.ones(ne) * self.m
        
        # Get basal nodes
        basal_nodes = np.where(md.mesh.vertexflags(1))[0]
        
        # # Use constant β² like Pattyn (e.g., uniform 1000 Pa·a·m⁻¹)
        # beta_squared_pattyn = 20000.0  # Pa·a·m⁻¹
        
        # C = np.full(nv, beta_squared_pattyn / yts ) # Pa·a·m⁻¹
        C_realistic = 1e5
        md.friction.coefficient = np.full(nv, C_realistic)
        
        # plt.hist(C, bins=50)
        # plt.yscale('log')
        # plt.xlabel('C (Pa·s·m⁻¹)')
        # plt.ylabel('Count')
        # plt.title('Distribution of C (Pa·s·m⁻¹)')
        # plt.show()

        # print(f"β² = {beta_squared_pattyn} Pa·a·m⁻¹")
        print(f"C = {md.friction.coefficient[0]:.2e} Pa·s·m⁻¹")
        # print(f"m = {md.friction.m} ")

        
        return md