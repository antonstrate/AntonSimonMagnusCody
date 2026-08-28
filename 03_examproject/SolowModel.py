from types import SimpleNamespace
from scipy import optimize
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({'axes.grid':True,'grid.color':'black','grid.alpha':'0.25','grid.linestyle':'-'})
plt.rcParams.update({'font.size':14})
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


class SolowModelClass:

    def __init__(self,**kwargs):

        par = self.par = SimpleNamespace()

        # a. technology
        par.alpha = 1/3 # capital share in production
        par.delta = 0.30 # depreciation rate

        # b. the long-run savings rate
        par.s_bar = 0.25

        # c. preferences
        par.beta = 1/1.2 # discount factor

        # d. simulation settings
        par.k0 = 0.10 # initial capital per worker
        par.T = 100 # number of periods

        # e. overwrite with keyword arguments, e.g. SolowModelClass(alpha=0.25)
        for key,value in kwargs.items(): setattr(par,key,value)

        # f. empty container for simulation results
        self.sim = SimpleNamespace()

    def __str__(self):

        par = self.par

        text = 'Solow model with:\n'
        text += f'  alpha = {par.alpha:.4f} (capital share)\n'
        text += f'  delta = {par.delta:.4f} (depreciation rate)\n'
        text += f'  s_bar = {par.s_bar:.4f} (long-run savings rate)\n'
        text += f'  beta  = {par.beta:.4f} (discount factor)\n'
        text += f'  k0    = {par.k0:.4f} (initial capital per worker)\n'
        text += f'  T     = {par.T} (number of periods)'

        return text

    def f(self,k):
        """ output per worker, y = f(k) = k**alpha """

        return k**self.par.alpha

    def k_next(self,k,s):
        """ capital per worker next period, k_next = s*f(k) + (1-delta)*k """

        return s*self.f(k) + (1-self.par.delta)*k

    def steady_state(self,s=None):
        """ analytical steady state for a constant savings rate, returns (k,y,c) """

        par = self.par
        if s is None: s = par.s_bar

        k = (s/par.delta)**(1/(1-par.alpha))
        y = self.f(k)
        c = (1-s)*y

        return k,y,c

    def solve_steady_state(self,s=None):
        """ numerical steady state, the k where k_next(k,s)-k = 0 """

        if s is None: s = self.par.s_bar

        obj = lambda k: self.k_next(k,s)-k # zero in the steady state
        result = optimize.root_scalar(obj,bracket=[1e-8,1e6],method='brentq')

        return result.root

    def simulate(self,s,k0=None):

        par = self.par
        sim = self.sim

        if k0 is None: k0 = par.k0

        # a. savings rate in each period (np.ndim(s) == 0 means s is a single number)
        s_vec = np.full(par.T,s) if np.ndim(s) == 0 else np.asarray(s,dtype=float)
        assert s_vec.size == par.T, f'the savings rate must have {par.T} elements, but has {s_vec.size}'

        # b. allocate memory
        k = np.empty(par.T) # capital per worker
        y = np.empty(par.T) # output per worker
        i = np.empty(par.T) # investment per worker
        c = np.empty(par.T) # consumption per worker

        # c. loop forward in time
        k[0] = k0
        for t in range(par.T):

            y[t] = self.f(k[t]) # production
            i[t] = s_vec[t]*y[t] # investment
            c[t] = y[t]-i[t] # consumption

            if t < par.T-1: k[t+1] = i[t] + (1-par.delta)*k[t] # law of motion

        # d. store the results
        sim.s = s_vec
        sim.k = k
        sim.y = y
        sim.i = i
        sim.c = c

        return sim

    # the savings path s_t = s_bar + (s0-s_bar)*phi**t
    def s_path(self,s0,phi):

        par = self.par
        t = np.arange(par.T)

        return par.s_bar + (s0-par.s_bar)*phi**t

    # the discounted sum of log(c_t)
    def welfare(self,c):
        """ W = sum_{t=0}^{T-1} beta**t * log(c_t), eq. (5) """

        par = self.par
        t = np.arange(par.T)

        return np.sum(par.beta**t*np.log(c))

    # welfare of the savings rule (s0,phi)
    def evaluate(self,s0,phi):
        """ simulate the model under the rule s_path(s0,phi) and return its welfare """

        s = self.s_path(s0,phi)
        sim = self.simulate(s)

        return self.welfare(sim.c)


def copy_sim(sim):

    return SimpleNamespace(s=sim.s.copy(),k=sim.k.copy(),y=sim.y.copy(),i=sim.i.copy(),c=sim.c.copy())


def s_path_general(model,s0,phi,s_target):

    t = np.arange(model.par.T)

    return s_target + (s0-s_target)*phi**t


def s_path_alt(model,s0,p,s_target):

    t = np.arange(model.par.T)

    return s_target + (s0-s_target)/(1+t)**p


def simulate_rule(model,s0,phi,s_target=None):

    if s_target is None: s_target = model.par.s_bar

    return copy_sim(model.simulate(s_path_general(model,s0,phi,s_target)))


def simulate_rule_alt(model,s0,p,s_target):
    """ simulate Question 6's alternative rule and return a safe copy """

    return copy_sim(model.simulate(s_path_alt(model,s0,p,s_target)))


def grid_search(model,s0_grid,phi_grid):

    W_grid = np.empty((len(s0_grid),len(phi_grid)))
    s0_best,phi_best,W_best = s0_grid[0],phi_grid[0],-np.inf

    for i,s0 in enumerate(s0_grid):
        for j,phi in enumerate(phi_grid):

            W = model.evaluate(s0,phi)
            W_grid[i,j] = W

            if W > W_best:
                s0_best,phi_best,W_best = s0,phi,W

    return W_grid,s0_best,phi_best,W_best


def k_last_table(sim_base,sims,names):
    """ small table with k_(T-1) for the baseline and the given rules """

    rows = [{'rule':'baseline','k_(T-1)':sim_base.k[-1]}]
    rows += [{'rule':name,'k_(T-1)':sim.k[-1]} for name,sim in zip(names,sims)]

    return pd.DataFrame(rows).set_index('rule')


def welfare_table(model,sim_base,sims,names):
    """ small table with W and W-W_baseline for the baseline and the given rules """

    W_base = model.welfare(sim_base.c)
    rows = [{'rule':'baseline','W':W_base,'W - W_baseline':0.0}]
    for name,sim in zip(names,sims):
        W = model.welfare(sim.c)
        rows.append({'rule':name,'W':W,'W - W_baseline':W-W_base})

    return pd.DataFrame(rows).set_index('rule')


def plot_three_panels(sim_base,sims=None,names=None,k_star=None,c_star=None,title=''):

    fig = plt.figure(figsize=(13,4.5))
    ax_s = fig.add_subplot(1,3,1)
    ax_k = fig.add_subplot(1,3,2)
    ax_c = fig.add_subplot(1,3,3)

    lw_base = 2.5 if sims else 2
    ax_s.plot(sim_base.s,lw=lw_base,color='black',label='baseline')
    ax_k.plot(sim_base.k,lw=lw_base,color='black',label='baseline')
    ax_c.plot(sim_base.c,lw=lw_base,color='black',label='baseline')

    if k_star is not None: 
        ax_k.axhline(k_star,ls='--',lw=1.5,color='grey',label=r'$k^*$')
        ax_k.set_yticks(np.arange(0,0.9,0.1))
    if c_star is not None:
        ax_c.axhline(c_star,ls='--',lw=1.5,color='grey',label=r'$c^*$')

    if sims is not None:
        for i,(sim,name) in enumerate(zip(sims,names)):
            color = colors[i%len(colors)]
            ax_s.plot(sim.s,lw=1.8,color=color,label=name)
            ax_k.plot(sim.k,lw=1.8,color=color,label=name)
            ax_c.plot(sim.c,lw=1.8,color=color,label=name)

    ax_s.set_title(r'Savings rate, $s_t$')
    ax_k.set_title(r'Capital, $k_t$')
    ax_c.set_title(r'Consumption, $c_t$')

    for ax in fig.axes:
        ax.set_xlabel('period, $t$')
        ax.set_xlim(0,sim_base.k.size-1)
    ax_s.legend(fontsize=9)

    fig.suptitle(title)
    fig.tight_layout()

    return fig


def plot_welfare_grid(s0_grid,phi_grid,W_grid,s0_best,phi_best):
    """ contourf of W over the (s0,phi) grid, with the best point marked """

    fig = plt.figure(figsize=(7,5.5))
    ax = fig.add_subplot(1,1,1)
    cs = ax.contourf(s0_grid,phi_grid,W_grid.T,levels=40,cmap='viridis')
    fig.colorbar(cs,ax=ax,label='W')
    ax.plot(s0_best,phi_best,'o',ms=10,color='red',label='best grid point')
    ax.set_xlabel(r'$s_0$'); ax.set_ylabel(r'$\varphi$')
    ax.set_title('W on a grid over $(s_0,\\varphi)$')
    ax.legend()
    fig.tight_layout()

    return fig
