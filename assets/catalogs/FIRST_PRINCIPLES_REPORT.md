# first-principles build report

PMLB `first_principles_*` (MIT) + refit reference laws. Deterministic rebuild (`tools/build_first_principles.py`); no RNG.

| eq_id | n | d | law | constants | FVU(ref) | R2 | gt_kind |
|---|---|---|---|---|---|---|---|
| hubble | 32 | 1 | `{c1}*v1` | ['475.3'] | 2.752e-01 | 0.72479 | reference |
| kepler | 6 | 1 | `{c1}*((v1)**(1.5))` | ['366.651'] | 3.111e-06 | 1.00000 | reference |
| newton | 30 | 3 | `log({c1}*v2*v3/((v1)**(2)))` | ['6.54662e-11'] | 9.585e-05 | 0.99990 | reference |
| ideal_gas | 30 | 3 | `log({c1}*v1*v2/v3)` | ['8.00321'] | 3.872e-03 | 0.99613 | reference |
| planck | 100 | 2 | `log({c1})+3*log(v1)-{c2}*v1/v2-log(1-exp(-{c2}*v1/v2))` | ['1.44845e-50', '4.79883e-11'] | 2.443e-07 | 1.00000 | reference |
| rydberg | 50 | 2 | `-log({c1}*(1/((v1)**(2))-1/((v2)**(2))))` | ['1.09634e+07'] | 3.483e-05 | 0.99997 | reference |
| leavitt | 26 | 1 | `{c1}*v1+{c2}` | ['-2.00315', '15.558'] | 6.547e-02 | 0.93453 | reference |
| schechter | 27 | 1 | `{c1}+{c2}*log(v1)+{c3}*v1` | ['-2.19506', '-1.20967', '-4.03879e-09'] | 1.478e-03 | 0.99852 | reference |
| bode | 8 | 1 | `{c1}+{c2}*exp({c3}*v1)` | ['0.429197', '0.139529', '0.700103'] | 1.861e-04 | 0.99981 | reference |
| tully_fisher | 18 | 1 | `{c1}*log(v1)+{c2}` | ['-2.64968', '-3.86615'] | 5.369e-02 | 0.94631 | reference |
| absorption | 14 | 1 | `log(1/({c1}+exp(-{c2}*v1)))` | ['0.0335752', '0.232973'] | 2.741e-03 | 0.99726 | reference |
| supernovae_zg | 243 | 1 | `{c1}/({c2}*exp({c3}*v1)+exp(-{c4}*v1))` | ['6.73704', '5.30797', '0.0722546', '0.239405'] | 6.336e-03 | 0.99366 | reference |
| supernovae_zr | 236 | 1 | `{c1}/({c2}*exp({c3}*v1)+exp(-{c4}*v1))` | ['147.05', '148.893', '0.0422184', '0.48034'] | 2.238e-02 | 0.97762 | reference |

13 problems -> `first-principles.npz`.

Split policy: all points support, empty validation. gt_kind rule: metadata-synthetic AND FVU<=1e-12 -> exact, else reference.
