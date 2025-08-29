# Budd_theory.py (rewritten for compatibility with ISSM Weertman-style friction)

import numpy as np
import matplotlib.pyplot as plt
from frictionweertman import frictionweertman

class BuddTheory:
    def __init__(self, config):
        self.config = config
        self.m = getattr(config, 'friction_m', 1.0)  # fallback default

    def get_basal_velocity(self, md):

        basal_nodes = np.where(md.mesh.vertexflags(1))[0]

        if basal_nodes.size == 0:
            print("⚠ No basal nodes found for friction coefficient computation.")
            return np.array([]), basal_nodes

        # Try results
        if hasattr(md, "results") and hasattr(md.results, "StressbalanceSolution"):
            if hasattr(md.results.StressbalanceSolution, "Vx"):
                vx = md.results.StressbalanceSolution.Vx
                return np.abs(vx[basal_nodes]), basal_nodes

        # Fall back to initialization
        vx = md.initialization.vx
        return np.abs(vx[basal_nodes]), basal_nodes


    def estimate_velocity_magnitude(self, md):
        vx_basal, _ = self.get_basal_velocity(md)
        if vx_basal.size == 0:
            print("⚠ estimate_velocity_magnitude: no basal velocity data, using fallback.")
            return 1e-2  # Some small fallback value
        velocity_magnitude = np.mean(vx_basal)
        return max(velocity_magnitude, 1e-3)


    def calculate_effective_viscosity(self, md):
        """Return constant effective viscosity for setup."""
        B = np.mean(md.materials.rheology_B)
        n = getattr(self.config, 'rheology_n', 3.0)
        eps_dot = 1e-6  # small strain rate (s⁻¹)
        eta = 0.5 * B * eps_dot**((1 - n) / n)
        eta = np.clip(eta, 1e12, 1e14)  # keep within physical range
        print(f"\nUsing η_eff = {eta:.2e} Pa·s for all vertices")
        eta_eff = np.ones(md.mesh.numberofvertices) * eta
        return np.clip(eta_eff, 1e12, 1e14)


    def calculate_sliding_coefficient(self, md, eta_eff):
        omega = self.config.omega
        beta = self.config.bedrock_params['amplitude']
        basal_nodes = np.where(md.mesh.vertexflags(1))[0]
        x = md.mesh.x[basal_nodes]

        eta_basal = eta_eff[basal_nodes] if len(eta_eff) > len(basal_nodes) else np.full_like(x, np.mean(eta_eff))
        stress = 2 * eta_basal * omega * beta * np.cos(omega * x)

        # safe stress offset independent of velocity
        stress_offset = np.abs(np.min(stress)) * 1.05 if np.min(stress) < 0 else 1e5
        stress_adj = stress + stress_offset
        print(f"  Stress offset applied: {stress_offset:.2e}")

        # Velocity magnitude, safe fallback
        vx_basal, _ = self.get_basal_velocity(md)
        velocity_mag = np.clip(vx_basal, 1e-6, None)  # per-node safe fallback

        print(f"  Velocity magnitude range: {velocity_mag.min():.2e} to {velocity_mag.max():.2e}")
        print(f"  Stress range: {stress.min():.2e} to {stress.max():.2e}")
        print(f"  Adjusted stress range: {stress_adj.min():.2e} to {stress_adj.max():.2e}")

        m = self.m
        shape_factor = 0.8# * config.rho_ice SEE SCREENSHOTS/JR EMAIL????

        coeff = shape_factor * stress_adj / (velocity_mag ** m)

        return coeff, basal_nodes


    def apply_to_model(self, md):
        print("\nApplying Budd-based Weertman friction law...")

        # Set up friction law
        md.friction = frictionweertman()
        md.friction.p = np.ones(md.mesh.numberofelements) * 1.0
        md.friction.q = np.ones(md.mesh.numberofelements) * 1.0
        md.friction.m = np.ones(md.mesh.numberofelements) * self.m

        # Calculate effective viscosity safely
        eta_eff = self.calculate_effective_viscosity(md)

        # Calculate Budd sliding coefficient at basal nodes
        coeff, basal_nodes = self.calculate_sliding_coefficient(md, eta_eff)

        print(f"Raw coeff range: {coeff.min():.2e} to {coeff.max():.2e}")

        # Clip and assign
        coeff_clipped = np.clip(coeff, 1e-3, 5e7)
        md.friction.C = np.ones(md.mesh.numberofvertices) * 1e-3
        md.friction.C[basal_nodes] = coeff_clipped

        print(f"✓ Budd sliding coefficients applied to model.")

        if coeff_clipped.size > 0:
            print(f"  Coefficient range: {np.min(coeff_clipped):.2e} to {np.max(coeff_clipped):.2e}")
        else:
            print("  Coefficient range: [empty] — fallback coefficients used.")

        return md