import numpy as np


class IncomeModel:
    """Life-cycle model for the income distribution."""

    def __init__(self, seed=123, **kwargs):
        """Set parameters and simulation settings."""

        # a. settings
        self.seed = seed
        self.N = 50_000
        self.age_min = 18
        self.age_max = 65
        self.ages = np.arange(self.age_min, self.age_max + 1)

        # b. education
        self.p_e = np.array([0.40, 0.35, 0.25])
        self.S_e = np.array([1, 3, 5])
        self.h_e0 = np.array([1.00, 1.20, 1.55])
        self.delta_e = np.array([0.010, 0.020, 0.030])

        # c. labor market
        self.delta = 0.06
        self.sigma_psi = 0.10
        self.lambda_ = 0.60
        self.sigma = 0.05

        # d. income
        self.y_SU = 0.45
        self.rho = 0.60
        self.y_floor = 0.35

        # e. health-risk extension
        self.p_health = 0.02
        self.health_loss = 0.30
        self.sigma_health = 0.15

        # f. overwrite parameters when needed
        for key, value in kwargs.items():
            setattr(self, key, value)

    def simulate(self, health_risk=False, no_unemployment=False):
        """Simulate the life-cycle model.

        Args:
            health_risk (bool): Add permanent health risk if True.
            no_unemployment (bool): Remove unemployment if True.
        """

        # a. random number generators
        rng = np.random.default_rng(self.seed)
        health_rng = np.random.default_rng(self.seed + 1)

        # b. setup
        N = self.N
        T = len(self.ages)

        education = rng.choice(3, size=N, p=self.p_e)
        education_years = self.S_e[education]
        h0 = self.h_e0[education]
        growth = self.delta_e[education]

        employed = np.zeros((N, T), dtype=bool)
        h = np.zeros((N, T))
        income = np.zeros((N, T))

        last_job_income = np.zeros(N)
        ever_employed = np.zeros(N, dtype=bool)
        sick = np.zeros(N, dtype=bool)

        # c. initial human capital
        h[:, 0] = h0

        # d. simulate over the life cycle
        for t, age in enumerate(self.ages):

            studying = age < self.age_min + education_years
            just_finished = age == self.age_min + education_years
            previously_active = age > self.age_min + education_years
            active = ~studying

            # i. health risk
            if health_risk:
                new_sick = (
                    active
                    & ~sick
                    & (health_rng.random(N) < self.p_health)
                )
                sick[new_sick] = True
            else:
                new_sick = np.zeros(N, dtype=bool)

            # ii. labor market
            if no_unemployment:
                employed[active, t] = True
            elif t > 0:
                draw_job = rng.random(N)
                was_employed = employed[:, t - 1]

                separation = np.full(N, self.sigma)
                if health_risk:
                    separation[sick] = self.sigma_health

                stays_employed = (
                    previously_active
                    & was_employed
                    & (draw_job >= separation)
                )
                finds_job = (
                    previously_active
                    & ~was_employed
                    & (draw_job < self.lambda_)
                )

                employed[stays_employed | finds_job, t] = True
                employed[just_finished, t] = False

            employed[studying, t] = False

            # iii. human capital
            if t > 0:
                h[:, t] = h[:, t - 1]

                psi = rng.lognormal(
                    -0.5 * self.sigma_psi**2,
                    self.sigma_psi,
                    size=N,
                )

                working = previously_active & employed[:, t]
                unemployed = previously_active & ~employed[:, t]

                if no_unemployment:
                    working = previously_active & employed[:, t]
                    unemployed = np.zeros(N, dtype=bool)

                h[working, t] = (
                    h[working, t - 1]
                    * (1 + growth[working])
                    * psi[working]
                )

                h[unemployed, t] = (
                    h[unemployed, t - 1]
                    * (1 - self.delta)
                    * psi[unemployed]
                )

                h[new_sick, t] = (
                    h[new_sick, t] * (1 - self.health_loss)
                )

            # iv. income
            working = active & employed[:, t]
            unemployed = active & ~employed[:, t]

            income[studying, t] = self.y_SU
            income[working, t] = h[working, t]

            last_job_income[working] = income[working, t]
            ever_employed[working] = True

            income[unemployed & ever_employed, t] = (
                self.rho * last_job_income[unemployed & ever_employed]
            )
            income[unemployed & ~ever_employed, t] = self.y_floor

        # e. save results
        self.education = education
        self.education_years = education_years
        self.employed = employed
        self.h = h
        self.income = income
        self.sick = sick


def gini(x):
    """Compute the Gini coefficient of a vector of incomes."""

    x = np.asarray(x).flatten()
    x = np.sort(x)

    n = len(x)
    index = np.arange(1, n + 1)

    return (
        2 * np.sum(index * x) / (n * np.sum(x))
        - (n + 1) / n
    )
