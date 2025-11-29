import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Sistema
def f(t, z):
    x, y = z
    dx = -2*x + y
    dy = -x - y
    return [dx, dy]

# Matriz para campo de direcciones
xmin, xmax, ymin, ymax = -5, 5, -5, 5
nx, ny = 25, 25
x = np.linspace(xmin, xmax, nx)
y = np.linspace(ymin, ymax, ny)
X, Y = np.meshgrid(x, y)
U = -2*X + Y
V = -X - Y

# Normalizar vectores para streamplot
N = np.sqrt(U**2 + V**2)
U2 = U / (N + 1e-8)
V2 = V / (N + 1e-8)

fig, ax = plt.subplots(figsize=(7,7))
ax.streamplot(X, Y, U2, V2, density=1.2, linewidth=1, arrowsize=1.2, arrowstyle='->', color='gray')

# Nullclinas
xs = np.linspace(xmin, xmax, 200)
ax.plot(xs, 2*xs, 'r--', label=r'$\dot x=0:\; y=2x$')
ax.plot(xs, -xs, 'b--', label=r'$\dot y=0:\; y=-x$')

# Trayectorias desde varias condiciones iniciales
t_span = (0, 10)
t_eval = np.linspace(t_span[0], t_span[1], 400)
inits = [(4,0), (3,3), (0,4), (-4,2), (2,-3), (-3,-3)]
for x0,y0 in inits:
    sol = solve_ivp(f, t_span, [x0, y0], t_eval=t_eval, rtol=1e-8)
    ax.plot(sol.y[0], sol.y[1], label=f'IC ({x0},{y0})')

# Mark origin
ax.plot(0,0,'ko',markersize=6)
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Plano de fase: dx/dt=-2x+y,  dy/dt=-x-y (foco estable)')
ax.legend(loc='upper right', fontsize='small')
ax.set_aspect('equal', 'box')
plt.grid(True)
plt.show()
