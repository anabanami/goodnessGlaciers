Debugging Terminus Instability in an long_flowline.py

This document summarizes the process of diagnosing and solving a numerical instability issue in a Flowband transient, Full-Stokes ice flow simulation using ISSM.

# 1. The Initial Problem: Runaway Acceleration

The simulation was failing on the second timestep (t=0.10 years) with a negative jacobian determinant error.

    Symptom: The diagnostic (Stressbalance) solution revealed a non-physical, extreme acceleration of ice at the terminus, with velocities exceeding 8e6 m/yr.

    Root Cause: The model used a stress-free outflow boundary condition (spcvx = NaN) at the terminus of a grounded glacier. This provided no back-pressure to resist the driving stress, leading to a numerical "runaway" effect where the ice accelerated uncontrollably toward the free boundary.

# 2. First Attempted Solution: The Buffer Zone

The first proposed solution was to extend the modeling domain to create a "buffer zone" or "sacrificial zone."
The Theory

The extreme acceleration is a numerical artifact caused by the unrealistic boundary condition. The influence of this artifact is local and decays with distance.

    By extending the domain (e.g., from 160 km to 210 km), the problematic boundary is moved further away from the region of interest (0-160 km).

    The runaway acceleration will now occur at the new, more distant terminus (210 km).

    The 50 km buffer zone absorbs and dampens the "shockwave" of this instability, preventing it from propagating back and corrupting the results in the primary study area.

The Result

This solution alone was not sufficient. The model still crashed at the second timestep, indicating that the instability was propagating upstream almost instantaneously.

# 3. The Deeper Insight: Timestep Instability (CFL Condition)

The key observation was that the crash was not truly instantaneous; it occurred after one full timestep (Δt = 0.05 years). This pointed to a violation of the Courant–Friedrichs–Lewy (CFL) condition, a fundamental concept in numerical modeling.

In simple terms, the distance a point in the model moves in one timestep (Δx = Velocity × Δt) cannot be significantly larger than the size of the mesh elements around it.
The Calculation

Using the values from the simulation:

    Velocity (V): ~8,560,000 m/yr (from the diagnostic run)

    Timestep (Δt): 0.05 yr

Δx = (8,560,000 m/yr) × (0.05 yr) = 428,000 meters

In a single step, the model was trying to stretch the mesh elements at the terminus by 428 km. This violent deformation flipped the elements inside-out, causing the negative jacobian determinant error.


## ~~~~~ interlude ~~~~~

The CFL condition. It's one of the most fundamental and important concepts in any kind of computer simulation that involves things moving or changing over time.

### The Intuitive Explanation: The "Don't Skip a Town" Rule

Imagine you are a superhero who can jump from one town to the next along a road. The towns are your **mesh elements** (your `Δx`). You have a special watch that ticks forward in fixed amounts of time, say, one hour at a time. This is your **timestep** (your `Δt`).

The CFL condition is a simple rule: **On any single jump (one timestep), you are not allowed to jump over an entire town.**

* **If you obey the rule:** You jump from Town A to somewhere inside the limits of Town B. The simulation is **stable**. You have "seen" what happens in every town along the way.
* **If you break the rule:** You are moving so fast that in one jump (one hour), you leap completely over Town B and land in Town C. The simulation is **unstable**. The computer has completely missed all the physics and information that was happening in Town B. It loses track of reality, and the simulation crashes.

In your glacier simulation:
* **Velocity** is how fast your superhero is moving.
* **Timestep (`Δt`)** is how long you wait between jumps.
* **Mesh size (`Δx`)** is the distance between your "towns" (the size of your triangles).

Your crash happened because your velocity was so high and your timestep was so long that your mesh points were trying to jump over hundreds of "towns" in a single leap.

### The Mathematical Relationship

The concept is formalized in a simple equation. For a basic one-dimensional problem, it looks like this:

$C = \frac{u \cdot \Delta t}{\Delta x} \le C_{max}$

Where:
* $C$ is the **Courant number**.
* $u$ is the **velocity** of the flow.
* $\Delta t$ is the **timestep**.
* $\Delta x$ is the **grid spacing** (the size of your mesh elements).
* $C_{max}$ is the maximum allowable Courant number for the simulation to remain stable. For many simple numerical methods, this value is 1.

This equation mathematically states our "don't skip a town" rule. It says that the distance the flow travels in one timestep ($u \cdot \Delta t$) must be less than or equal to the size of one grid cell ($\Delta x$).

If you rearrange the formula to solve for the maximum stable timestep, you get:

$\Delta t \le \frac{\Delta x}{u}$

This is the crucial takeaway: **The maximum stable timestep is directly proportional to the mesh size and inversely proportional to the velocity.**

* If your velocity (`u`) gets very high, your stable timestep (`Δt`) must get very small.
* If your mesh (`Δx`) is very fine (small), your stable timestep (`Δt`) must also be smaller.

### How This Caused Your Specific Crash

In your case, the `negative jacobian determinant` error is the finite-element equivalent of the simulation "losing track of reality." When a mesh point tries to move a distance far greater than the size of the element it belongs to, the element gets stretched and twisted so violently that it turns inside-out. The "Jacobian determinant" is a measure of the element's volume or area; when it becomes negative, it means the element has been inverted, which is physically impossible.

By violating the CFL condition so dramatically, you were telling the solver to perform a physically impossible deformation of the mesh, and it correctly threw an error and stopped.








# 4. The Combined Solution & Main Lessons

The final, successful solution requires combining both strategies.

    Extend the Domain: Create a buffer zone to contain the boundary artifact and reduce the peak velocity at the terminus. For best results, the ice geometry in this zone should be tapered to create a more physically realistic and stable initial state.

    Reduce the Timestep: Drastically reduce Δt to a value small enough to safely resolve the highest velocities present anywhere in the model. The model must take "baby steps" in time to prevent the mesh from deforming too quickly.

# Key Takeaways

    Lesson 1: Isolate Boundary Artifacts. When a boundary condition does not perfectly represent reality, use a buffer zone to move its non-physical effects away from your region of interest.

    Lesson 2: Respect the Timestep. A transient simulation's stability is highly sensitive to the timestep. If a model crashes early, check for extreme velocities in the diagnostic solution and ensure your timestep is small enough to prevent nodes from moving too far in a single step.