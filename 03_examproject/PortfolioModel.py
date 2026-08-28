from types import SimpleNamespace
import numpy as np

class PortfolioModelClass:
    """ a portfolio of a risky and a safe asset with a rebalancing rule """

    def __init__(self,**kwargs):
        """ set the default parameters, then overwrite with any keyword arguments """

        par = self.par = SimpleNamespace()

        # a. returns
        par.mu = 0.05 # mean log return on the risky asset
        par.sigma = 0.20 # standard deviation of the log return on the risky asset
        par.r = 0.01 # log return on the safe asset

        # b. the rebalancing rule
        par.theta_star = 0.50 # target share of wealth in the risky asset
        par.Delta = 0.10 # width of the no-trade band
        par.tau = 0.01 # proportional transaction cost

        # c. preferences
        par.gamma = 3.0 # relative risk aversion

        # d. simulation settings
        par.W0 = 1.0 # initial wealth
        par.T = 40 # number of periods
        par.N = 50_000 # number of simulated portfolios
        par.seed = 2026 # seed for the random number generator

        # e. overwrite with keyword arguments, e.g. PortfolioModelClass(Delta=0.0)
        for key,value in kwargs.items(): setattr(par,key,value)

        # f. empty container for simulation results
        self.sim = SimpleNamespace()

    def __str__(self):
        """ called when using print """

        par = self.par

        text = 'Portfolio model with:\n'
        text += f'  mu    = {par.mu:.4f}, sigma = {par.sigma:.4f}, r = {par.r:.4f}\n'
        text += f'  theta_star = {par.theta_star:.4f}, Delta = {par.Delta:.4f}, tau = {par.tau:.4f}\n'
        text += f'  gamma = {par.gamma:.4f} (relative risk aversion)\n'
        text += f'  W0 = {par.W0:.2f}, T = {par.T}, N = {par.N:,}, seed = {par.seed}'

        return text

    def draw_returns(self):
        """ draw the gross return on the risky asset in all periods and all portfolios

        Returns:

            (ndarray): gross returns with shape (N,T)

        """

        par = self.par

        rng = np.random.default_rng(par.seed)
        eps = rng.normal(size=(par.N,par.T))

        return np.exp(par.mu + par.sigma*eps)

    def u(self,W):
        """ CRRA utility of wealth """

        par = self.par

        return W**(1-par.gamma)/(1-par.gamma)

    # the share of wealth in the risky asset after trading, and the amount traded
    def trade(self,theta):

        par = self.par

        # a. is the portfolio too far from the target?
        distance = np.abs(theta-par.theta_star)
        outside_band = distance > par.Delta

        # b. trade back to the target if outside the band, else keep theta
        theta_post = np.where(outside_band,par.theta_star,theta)

        # c. how much was traded
        amount_traded = np.abs(theta_post-theta)

        return theta_post,amount_traded

    # simulate all N portfolios forward T periods
    def simulate(self,R=None):

        par = self.par

        # a. the returns. If R is not given, draw a new set. If it is given,
        #    several calls to .simulate() can use the same returns.
        if R is None: R = self.draw_returns()
        Rf = np.exp(par.r) # the safe return is the same every period

        T,N = par.T,par.N

        # b. allocate arrays. theta and W have T+1 rows (state at the start
        #    of each period, plus the final state); traded/dist have T rows.
        theta = np.empty((T+1,N)) # share in the risky asset
        W = np.empty((T+1,N)) # wealth
        traded = np.empty((T,N),dtype=bool) # was period t traded?
        dist = np.empty((T,N)) # |theta_t-theta_star| before trading

        theta[0,:] = par.theta_star # start at the target
        W[0,:] = par.W0

        # c. loop forward in time
        for t in range(T):

            # i. trade or not
            dist[t,:] = np.abs(theta[t,:]-par.theta_star)
            theta_post,amount_traded = self.trade(theta[t,:])
            traded[t,:] = amount_traded > 0

            # ii. pay the trading cost
            W_post = W[t,:]*(1-par.tau*amount_traded)

            # iii. realize the return and update wealth and theta
            W[t+1,:] = theta_post*W_post*R[:,t] + (1-theta_post)*W_post*Rf
            theta[t+1,:] = theta_post*W_post*R[:,t]/W[t+1,:]

        # d. save the results
        sim = self.sim
        sim.R = R
        sim.Rf = Rf
        sim.theta = theta
        sim.W = W
        sim.traded = traded
        sim.dist = dist

        return sim

    # the numbers to report for a rule, including expected utility
    def summary(self):

        sim = self.sim

        WT = sim.W[-1,:] # terminal wealth, i.e. wealth in the last period

        return {
            # average number of trades per portfolio
            'n_trades': sim.traded.sum(axis=0).mean(),

            # average distance to target, over time and portfolios
            'avg_dist': sim.dist.mean(),

            'mean_WT': WT.mean(),
            'median_WT': np.median(WT),
            'p10_WT': np.percentile(WT,10),
            'E[u(WT)]': self.u(WT).mean(),
        }
