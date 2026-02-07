"""
Ramsey Model: Simple Numerical Solver
Solves: k_dot = f(k) - c - (δ+n)k and c_dot/c = σ[f'(k) - δ - ρ]
"""

import numpy as np
import matplotlib.pyplot as plt

### Parameters
A = 1.0
sigma = 0.5
rho = 0.05
delta = 0.05
n = 0.02
dt = 0.1
N = 1000

# Steady-state
k_star = 1.0
c_star = A * k_star - (delta + n) * k_star
k0 = 0.5 * k_star

print(f"k* = {k_star}, c* = {c_star:.4f}, k0 = {k0}")

### Simulate dynamics
def simulate(c0):
    k = np.zeros(N + 1)
    c = np.zeros(N + 1)
    k[0] = k0
    c[0] = c0
    
    for t in range(N):
        f_k = A * k[t]
        k[t + 1] = dt * (f_k - c[t] - (delta + n) * k[t]) + k[t]
        c[t + 1] = dt * sigma * (A - delta - rho) * c[t] + c[t]
    
    return k, c

### Bisection: find optimal c0
c_low, c_high = 0.01, c_star * 0.95
c0_opt = 0.5

for iteration in range(100):
    c0_opt = (c_low + c_high) / 2
    k, c = simulate(c0_opt)
    
    error = np.sqrt(2 * ((c[-1] - c_star) / (c[-1] + c_star))**2 + 
                    ((k[-1] - k_star) / (k[-1] + k_star))**2)
    
    if error < 0.01:
        break
    
    # If final c is above target, we need lower c0
    # If final k is above target, we need lower c0 (less consumption, more investment)
    if c[-1] > c_star or k[-1] > k_star:
        c_high = c0_opt
    else:
        c_low = c0_opt

print(f"\n(a) Optimal c0 = {c0_opt:.6f}")

# Final paths
k_opt, c_opt = simulate(c0_opt)
t = np.arange(N + 1) * dt

### Question (b): Plot k and c with steady-state
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(t, k_opt, 'b-', linewidth=2, label='k(t)')
ax1.axhline(k_star, color='r', linestyle='--', alpha=0.7, label='k*')
ax1.set_xlabel('Time')
ax1.set_ylabel('Capital k')
ax1.set_title('(b) Capital path')
ax1.set_xlim(0, 100)
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(t, c_opt, 'g-', linewidth=2, label='c(t)')
ax2.axhline(c_star, color='r', linestyle='--', alpha=0.7, label='c*')
ax2.set_xlabel('Time')
ax2.set_ylabel('Consumption c')
ax2.set_title('(b) Consumption path')
ax2.set_xlim(0, 100)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ramsey_b.png', dpi=150)
print("(b) Saved: ramsey_b.png")
plt.close()

### Question (c): Compare three scenarios
k_high, c_high = simulate(c0_opt * 1.2)
k_low, c_low = simulate(c0_opt * 0.8)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(t, k_opt, 'b-', linewidth=2, label='Optimal c0')
ax1.plot(t, k_high, 'g--', linewidth=2, label='High c0 (+20%)')
ax1.plot(t, k_low, 'r:', linewidth=2, label='Low c0 (-20%)')
ax1.axhline(k_star, color='gray', linestyle='--', alpha=0.5)
ax1.set_xlabel('Time')
ax1.set_ylabel('Capital k')
ax1.set_title('(c) Capital: three scenarios')
ax1.set_xlim(0, 100)
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(t, c_opt, 'b-', linewidth=2, label='Optimal c0')
ax2.plot(t, c_high, 'g--', linewidth=2, label='High c0 (+20%)')
ax2.plot(t, c_low, 'r:', linewidth=2, label='Low c0 (-20%)')
ax2.axhline(c_star, color='gray', linestyle='--', alpha=0.5)
ax2.set_xlabel('Time')
ax2.set_ylabel('Consumption c')
ax2.set_title('(c) Consumption: three scenarios')
ax2.set_xlim(0, 100)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ramsey_c.png', dpi=150)
print("(c) Saved: ramsey_c.png")
plt.close()

print("\n" + "="*60)
print("EXERCISE ANSWERS")
print("="*60)
print(f"(a) Optimal c0: {c0_opt:.6f}")
print("(b) Capital and consumption paths plotted (ramsey_b.png)")
print("(c) Three scenarios comparison (ramsey_c.png)")
print("="*60)
