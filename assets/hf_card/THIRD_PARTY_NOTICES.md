# Third-party notices (per-catalog upstream sources and licenses)

Full license texts and vendored upstream files live in the source repo
(github.com/psaegert/symbolic-data, `assets/upstream/*/NOTICE` + `LICENSE.*` files).

| Catalog(s) | Upstream source | License |
|---|---|---|
| fastsrb (@1, @2) | FastSRB (viktmar/FastSRB; SRSD-lineage Feynman renderings, Matsubara et al. 2024 ranges) | MIT |
| feynman, feynman-bonus | Feynman Symbolic Regression Database (Udrescu & Tegmark 2020) — specs regenerated from the equation database (formulas + ranges as cited facts; original data files unlicensed, not redistributed); bonus csv via WassimTenachi/PhySO (MIT) | see note |
| nguyen, constant, grammarvae, jin, keijzer, korns, koza, livermore, livermore2, meier, neat, nonic, pagie, poly, r-rationals, sine, vladislavleva | dso-org/deep-symbolic-optimization benchmarks.csv (Petersen et al.) | BSD-3-Clause |
| srsd-dummy | SRSD recipe (Matsubara et al. 2024) applied to fastsrb entries | MIT |
| first-principles | PMLB first_principles_* (EpistasisLab/pmlb; EmpiricalBench/Cranmer 2023 + MvSR/Russeil et al. 2024) | MIT |
| cp3-cosmo, cp3-blackbox | CP3-Origins/Things-to-bench (Thing & Koksbang, arXiv:2406.15531) | MIT |
| ai-descartes | IBM/AI-Descartes (Cornelio et al. 2023) | MIT |
| physo-streams, physo-astro, physo-class | WassimTenachi/PhySO (Tenachi et al.); physo-astro specs authored from arXiv:2303.03192 App. A | MIT |
| soose-nc, soose-wc, soose-fc | SymposiumOrganization/NeuralSymbolicRegressionThatScales (Biggio et al. 2021) | MIT |
| erbench-syneq, erbench-phybench, erbench-densities | ERBench (Kahlmeyer et al., arXiv:2606.09276; HF EquationDiscovery/Equation_Recovery_Benchmark) — SynEq author-generated MIT; PHYBench MIT (paper table + upstream Eureka-Lab/PHYBench); Densities BSD-3 (SciPy-derived) | MIT / BSD-3 |
| srbench2-blackbox | PMLB data (EpistasisLab/pmlb, MIT); selection list from SRBench 2.0 (Aldeia et al., facts only — no GPL code/data) | MIT |
| v23-val, lample-charton-v23 | self-generated (flash-ansr project artifacts) | project license |

CC BY-SA 4.0 catalogs (erbench-oeis, erbench-eponymous — OEIS-/Wikipedia-derived) are in the
separate psaegert/symbolic-data-assets-sa repo with their own LICENSE and attribution.
