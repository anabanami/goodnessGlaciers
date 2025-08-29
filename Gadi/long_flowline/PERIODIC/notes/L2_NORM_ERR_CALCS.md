## Mathematical Foundation

### 1. The L2 Norm (Euclidean Norm)

For a vector **v** = [v₁, v₂, v₃, ..., vₙ], the L2 norm is:

```
||v||₂ = √(v₁² + v₂² + v₃² + ... + vₙ²)
```

This is essentially the "length" or "magnitude" of the vector in n-dimensional space.

### 2. L2 Error Calculation

```python
# Reference solution norm
ref_norm = np.linalg.norm(ref_valid)

# Difference between solutions
diff = ref_valid - comp_valid

# Relative L2 error
l2_error = np.linalg.norm(diff) / ref_norm
```

Mathematically, this is:

```
L2_relative_error = ||u_reference - u_comparison||₂ / ||u_reference||₂
```

## Step-by-Step Breakdown

### Step 1: Calculate the Difference Vector
```python
diff = ref_valid - comp_valid
```

Having velocity data at grid points:
- `ref_valid` = [100.0, 95.0, 80.0, 60.0, 30.0] m/a (reference solution)
- `comp_valid` = [99.5, 94.2, 79.8, 59.5, 29.8] m/a (comparison solution)
- `diff` = [0.5, 0.8, 0.2, 0.5, 0.2] m/a

### Step 2: Calculate L2 Norm of the Difference
```python
error_norm = np.linalg.norm(diff)
```

```
||diff||₂ = √(0.5² + 0.8² + 0.2² + 0.5² + 0.2²)
          = √(0.25 + 0.64 + 0.04 + 0.25 + 0.04)
          = √1.22
          = 1.105 m/a
```

### Step 3: Calculate L2 Norm of Reference Solution
```python
ref_norm = np.linalg.norm(ref_valid)
```

```
||u_ref||₂ = √(100² + 95² + 80² + 60² + 30²)
           = √(10000 + 9025 + 6400 + 3600 + 900)
           = √29925
           = 173.0 m/a
```

### Step 4: Calculate Relative Error
```python
l2_error = error_norm / ref_norm
```

```
L2_relative_error = 1.105 / 173.0 = 0.0064 = 0.64%
```

## Why Use Relative L2 Error?

### 1. **Scale Independence**
- Absolute errors depend on the magnitude of the solution
- If velocities are 1000 m/a, a 10 m/a error might be acceptable
- If velocities are 10 m/a, a 10 m/a error is huge!
- Relative error normalizes this: 10/1000 = 1% vs 10/10 = 100%

### 2. **Global Error Measure**
- L2 norm considers **all** grid points simultaneously
- Unlike maximum error (worst single point), it gives overall solution quality
- Squares the differences, so larger errors are penalized more heavily

### 3. **Convergence Assessment**
- As mesh gets finer, solutions should approach the "true" solution
- Relative L2 error should decrease systematically
- Common threshold: < 1% indicates good convergence

## The Code Implementation

```python
# From your analyse_grid_convergence.py
def calculate_convergence_metrics(self, tolerance=0.01):
    finest_res = min(self.results.keys())  # Reference solution
    ref_result = self.results[finest_res]
    
    for res_factor in sorted(self.results.keys()):
        if res_factor == finest_res:
            continue  # Skip reference
        
        # Get interpolated data on common grid
        ref_data = ref_result['vx_surface']
        comp_data = result['vx_surface_interp']
        
        # Handle near-zero solutions
        ref_norm = np.linalg.norm(ref_data)
        if ref_norm < 1e-10:
            l2_error = np.nan  # Avoid division by zero
        else:
            diff = ref_data - comp_data
            l2_error = np.linalg.norm(diff) / ref_norm  # THIS IS THE KEY CALCULATION
```

## Interpretation of Results

From your convergence report:

| Resolution | Surface vx L2 Error | Interpretation |
|------------|--------------------|--------------| 
| 0.75       | 0.41%              | ✅ Excellent - very close to reference |
| 1.0        | 0.34%              | ✅ Excellent - surprisingly good |
| 1.5        | 1.17%              | ❌ Marginal - noticeable differences |

### What These Numbers Mean:

**0.41% L2 error**: On average across all grid points, the solution differs from the reference by about 0.4% of the reference magnitude.

**Geometric interpretation**: Plotting both solutions, they would be visually almost identical.

## Alternative Error Measures in This Code

### 1. **Maximum Relative Error**
```python
max_error = np.max(np.abs(diff)) / np.max(np.abs(ref_data))
```
- **Purpose**: Worst-case error at any single point
- **Use**: Ensures no individual point has excessive error

### 2. **Root Mean Square Error (RMSE)**
```python
rmse = np.sqrt(np.mean(diff**2))
```
- **Purpose**: Average error magnitude in original physical units
- **Use**: Understand typical error size in m/a

## Why L2 is Preferred for Convergence Studies

1. **Mathematical properties**: Smooth, differentiable, well-behaved
2. **Physical meaning**: Total energy-like measure of the error
3. **Computational efficiency**: Single norm calculation
4. **Standard practice**: Widely used in numerical analysis
5. **Balances all errors**: Not dominated by single outlier points

The L2 relative error gives a robust, scale-independent measure of how much coarser mesh solutions deviate from the finest mesh reference solution.