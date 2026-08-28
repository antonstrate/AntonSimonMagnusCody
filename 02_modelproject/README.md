# ModelProject

Model analysis project: a consumer with nested CES preferences over food, bus
trips and train trips, solved numerically and used to study relative-price
effects and tax policy.

## Files

- `Consumer.py` — `ConsumerClass`: the nested CES consumer (utility, budget
  shares, quantities), plus two solution methods (`solve_grid()` for a
  2D grid search, `solve()` for L-BFGS-B).
- `Government.py` — `GovernmentClass(ConsumerClass)`: adds a lump-sum tax and
  product taxes on top of the consumer, tax revenue, and root-finding for a
  target revenue.
- `Results.ipynb` — the single self-contained notebook with all results,
  figures and answers.

## Running

Open `Results.ipynb` in the same folder as `Consumer.py` and `Government.py`,
and run all cells from the top ("Restart Kernel and Run All"). No other setup
is required beyond `numpy`, `scipy`, `pandas` and `matplotlib`.

## Contributions

The work on the model project was mainly divided as follows:

- **Anton Strate:** Tasks 1 and 3
- **Cody Brinch:** Task 2
- **Simon Fangel and Magnus Green:** Tasks 4 and 5

In addition, all group members contributed to and discussed the different tasks throughout the project.