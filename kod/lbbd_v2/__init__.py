"""lbbd_v2 — clean-room exact LBBD for the wildfire suppression problem.

Reference model = the CORRECTED canonical model (three fixes over fire_1404_baseline.py):
  1. per-cell water Big-M  M^s_i = a_i·burn_i·margin + Δ_wat   (never a scalar constant)
  2. (26'): only BURNING neighbours constrain t^{s,min}_i
  3. (28'): the t^{s,min} selector must reference a burning neighbour

Priority order: CORRECTNESS > EXACTNESS > REPRODUCIBILITY > PERFORMANCE.
All results are written under Logicbbd/result_lbbd/.
The legacy files in the repo root are historical reference and are NOT imported here.
"""
