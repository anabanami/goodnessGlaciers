# Glen's Flow law
This calculations are for a 2D system 
$$\dot{\varepsilon}_{ij} \,=\, A_{n}(T)\,\tau^{n-1}\,\tau_{ij},$$

let $n = 1$:

$$\dot{\varepsilon}_{ij} =  A_{1}(T)\,\tau^{1-1}\,\tau_{ij},$$

$$\Rightarrow\,\dot{\varepsilon}_{ij}\,=\,A_{1}(T)\,\tau_{ij},$$

from the definition of the full stress (i.e. $\sigma_{ij}\,=\,-p\,\delta_{ij}\,+\,\tau_{ij}$ ) , substitute the deviatoric stress tensor $\tau_{ij}$ in the strain rate equation
$$\Rightarrow\,\dot{\varepsilon}_{ij}\,=\,A_{1}(T)\,(\sigma_{ij}\,+\,p\,\delta_{ij}),$$
here the pressure is the negative average of the stress in 2 dimensions $p\,=\,-\frac{1}{2}\,(\Sigma_{i}\,\sigma_{ii})$
$$\Rightarrow\,\dot{\varepsilon}_{ij}\,=\,A_{1}(T)\,(\sigma_{ij}\,-\frac{1}{2}\,(\Sigma_{i}\,\sigma_{ii})\,\delta_{ij}),$$
$$\Rightarrow\,\dot{\varepsilon}_{ij}\,=\,A_{1}(T)\,(\sigma_{ij}\,-\frac{1}{2}\,(\Sigma_{i}\,\sigma_{ii})\,\delta_{ij}),$$$$\Rightarrow\begin{bmatrix}
 \dot{\varepsilon}_{xx} & \dot{\varepsilon}_{xy} \\
\dot{\varepsilon}_{yx} & \dot{\varepsilon}_{yy} \\
\end{bmatrix}\,=\,A_{1}(T)\left (\begin{bmatrix}
 \sigma_{xx} & \sigma_{xy} \\
\sigma_{yx} & \sigma_{yy} \\
\end{bmatrix}\,-\,\frac{1}{2}(\sigma_{xx}\,+\,\sigma_{yy})\begin{bmatrix} 1& 0\\ 0& 1\\ \end{bmatrix}\right ),$$
$$\Rightarrow\begin{bmatrix}
 \dot{\varepsilon}_{xx} & \dot{\varepsilon}_{xy} \\
\dot{\varepsilon}_{yx} & \dot{\varepsilon}_{yy} \\
\end{bmatrix}\,=\,A_{1}(T)\left (\begin{bmatrix}
 \frac{1}{2}(\sigma_{xx}\,-\,\sigma_{yy}) & \sigma_{xy} \\
\sigma_{yx} & \frac{1}{2}(\sigma_{yy}\,-\,\sigma_{xx}) \\
\end{bmatrix}\right ),$$

While letting $n\,=\,3$ in Glen's flow law returns

$$\dot{\varepsilon}_{ij} =  A_{3}(T)\,\tau^{3-1}\,\tau_{ij},$$

$$\Rightarrow\,\dot{\varepsilon}_{ij}\,= \,A_{3}(T)\,\tau^{2}\,\tau_{ij},$$

once again, substitute the deviatoric stress tensor $\tau_{ij}$ in the strain rate equation
$$\Rightarrow\,\dot{\varepsilon}_{ij}\,=\,A_{3}(T)\,\tau^{2}\,(\sigma_{ij}\,+\,p\,\delta_{ij}),$$
here the pressure is the negative average of the stress in 2 dimensions $p\,=\,-\frac{1}{2}\,(\Sigma_{i}\,\sigma_{ii})$
$$\Rightarrow\,\dot{\varepsilon}_{ij}\,=\,A_{3}(T)\,\tau^{2}\,(\sigma_{ij}\,-\frac{1}{2}\,(\Sigma_{i}\,\sigma_{ii})\,\delta_{ij}),$$
$$\Rightarrow\begin{bmatrix}
 \dot{\varepsilon}_{xx} & \dot{\varepsilon}_{xy} \\
\dot{\varepsilon}_{yx} & \dot{\varepsilon}_{yy} \\
\end{bmatrix}\,=\,A_{3}(T)\,\tau^{2}\,\begin{bmatrix}
 \frac{1}{2}(\sigma_{xx}\,-\,\sigma_{yy}) & \sigma_{xy} \\
\sigma_{yx} & \frac{1}{2}(\sigma_{yy}\,-\,\sigma_{xx}) \\
\end{bmatrix},$$
here it is useful to write the effective stress as $\tau\,=\,\sqrt{\frac{1}{2}\tau_{ij}\tau_{ij}}$  $\Rightarrow\,\tau^2\,=\,\frac{1}{2}\tau_{ij}\tau_{ij}$.

The term $\tau_{ij}\tau_{ij}$ represents the sum of the squares of all components of the deviatoric stress tensor matrix (Frobenius norm)
$$\dot{\varepsilon}_{ij}\,=\, A_3(T)\left(  \frac{1}{2}\left( \left( \frac{1}{2}\left(\sigma_{xx}-\sigma_{yy}\right)\right)^2\,+\,\sigma_{xy}^2\,+ \sigma_{yx}^2\,\left( \frac{1}{2}\left(\sigma_{yy}-\sigma_{xx}\right)\right)^2\right)\right) \begin{bmatrix} \frac{1}{2}(\sigma_{xx}-\sigma_{yy}) & \sigma_{xy} \\ \sigma_{yx} & \frac{1}{2}(\sigma_{yy}-\sigma_{xx}) \end{bmatrix} $$
Assuming a symmetric stress tensor, where $\sigma_{xy}\,=\,\sigma_{yx}$,
$$\dot{\varepsilon}_{ij}\,=\, A_3(T)\left(   \frac{1}{2}\left(  \frac{1}{4}\left(\sigma_{xx}-\sigma_{yy}\right)^2\,+\,2\sigma_{xy}^2\,+\,\frac{1}{4}\left(\sigma_{yy}-\sigma_{xx}\right)^2\right)\right) \begin{bmatrix} \frac{1}{2}(\sigma_{xx}-\sigma_{yy}) & \sigma_{xy} \\ \sigma_{yx} & \frac{1}{2}(\sigma_{yy}-\sigma_{xx}) \end{bmatrix}$$
$$\Leftrightarrow\,\begin{bmatrix}
 \dot{\varepsilon}_{xx} & \dot{\varepsilon}_{xy} \\
\dot{\varepsilon}_{yx} & \dot{\varepsilon}_{yy} \\
\end{bmatrix}\,=\,\frac{1}{4}A_{3}(T)\, \left(\left(\sigma_{xx}-\sigma_{yy}\right)^2\,+\,4\sigma_{xy}^2\right) \begin{bmatrix} \frac{1}{2}(\sigma_{xx}-\sigma_{yy}) & \sigma_{xy} \\ \sigma_{yx} & \frac{1}{2}(\sigma_{yy}-\sigma_{xx}) \end{bmatrix}$$

We now match the strain rates for both the linear and non-linear cases:

$$\dot{\varepsilon}_{ij}(n=1)\,=\,\dot{\varepsilon}_{ij}(n=3)$$

let $M = \begin{bmatrix} \frac{1}{2}(\sigma_{xx}-\sigma_{yy}) & \sigma_{xy} \\ \sigma_{yx} & \frac{1}{2}(\sigma_{yy}-\sigma_{xx}) \end{bmatrix}$ and simplify the equation:
$$\Leftrightarrow\,A_{1}(T)M \,=\, \,\frac{1}{4}A_{3}(T)\, \left(\left(\sigma_{xx}-\sigma_{yy}\right)^2\,+\,4\sigma_{xy}^2\right)M$$

$$\therefore\,A_{1}(T) \,=\, \,\frac{1}{4}A_{3}(T)\,\left(\left(\sigma_{xx}-\sigma_{yy}\right)^2\,+\,4\sigma_{xy}^2\right)$$

# IMPLEMENTING THIS IN CODE
the workflow should be a two-step process:

1. **Run the physical, non-linear (`n=3`) simulation first.** This is "reference" or "ground-truth" experiment.
	**`reference_nonlinear_flowline.py`:**
     run the non-linear experiment (`exp = 'S4'`, `rheology_n = 3`).
     It uses the standard `cuffey` function to determine the ice rigidity (`rheology_B`) based on temperature.
        
 Crucially, after the diagnostic solve,  add a new section to save the essential results: `stress_xx`, `stress_yy`, `stress_xy`, and the `rheology_B_n3` field. This perfectly captures the "reference state" needed for the next step.
```python

```

2. **Use the formula to calculate the equivalent, spatially varying `rheology_B` field** for the `n=1` case.
This script is also configured to run the linear experiment (`exp = 'S3'`, `rheology_n = 1`).       
Instead of using a constant value for the ice rigidity. The script then assigns this newly calculated `rheology_B` field to `md.materials.rheology_B` before solving.
```python
```

To verify the success of this implementation, compare the `Vx` output from the diagnostic solves of both models. They should be nearly identical