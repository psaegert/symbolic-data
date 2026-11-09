# Third-party notices (per-catalog upstream sources and licenses)

No single license covers this repo: every catalog stays under the license of the source it took content from.
The vendored upstream files and their NOTICEs live in the source repo (github.com/psaegert/symbolic-data,
`assets/upstream/*/NOTICE` + `LICENSE.*` files).

## License texts

- MIT, BSD-3-Clause and Apache-2.0: `LICENSES/MIT.txt`, `LICENSES/BSD-3-Clause.txt`, `LICENSES/Apache-2.0.txt`; each
  source's copyright notice is in the table "Copyright notices" at the end.
- CC BY 4.0: https://creativecommons.org/licenses/by/4.0/legalcode
- CC BY 3.0: https://creativecommons.org/licenses/by/3.0/legalcode
- CC BY 2.5: https://creativecommons.org/licenses/by/2.5/legalcode
- CC0 1.0: https://creativecommons.org/publicdomain/zero/1.0/legalcode
- Etalab Open Licence 2.0: https://www.etalab.gouv.fr/wp-content/uploads/2018/11/open-licence.pdf

What srbf changed: every catalog of srbf's 2026-11 release states it in its metadata (`srbf_changes`); a data
catalog stores the rows its authors published as arrays.

## Datasets under other terms in catalogs published before 2026-11

`srbench2-blackbox@1` and `first-principles@1` (published 2026-07) include datasets whose sources set
non-commercial terms or state no license; the license check of 2026-11-09 found them. Their versions stay
published so that pinned references keep resolving. Newer catalogs keep such data in
psaegert/symbolic-data-assets-restricted.

| Catalog | Dataset | Terms at the source |
|---|---|---|
| srbench2-blackbox@1 | 192_vineyard | non-commercial use only (Simonoff smoothmeth); PMLB's copy (MIT) |
| srbench2-blackbox@1 | 557_analcatdata_apnea1 | non-commercial use only (analcatdata); PMLB's copy (MIT) |
| srbench2-blackbox@1 | 1028_SWD, 1089_USCrime, 210_cloud, 522_pm10, 678_visualizing_environmental | none stated at the source; PMLB's copy (MIT) |
| first-principles@1 | absorption, supernovae_zg, supernovae_zr | none stated: erusseil/MvSR-analysis real_data (Russeil et al. 2024), unlicensed; PMLB's copy (MIT) |

srbf republishes these data unchanged so that every run uses exactly the same bytes, pinned by sha256, and
does not depend on a third-party host that may change, move or disappear. If you hold rights to these data
and object to their republication, open an issue at https://github.com/psaegert/symbolic-data and they will
be removed.

## Permission notes

Data that may be redistributed only with a note attached carry it here and in the problem's metadata
(`permission_note`).

505_tecator (srbench-2021-blackbox), from the Tecator data set at StatLib (OpenML 505):

```
1. Statement of permission from Tecator (the original data source)

These data are recorded on a Tecator Infratec Food and Feed Analyzer
working in the wavelength range 850 - 1050 nm by the Near Infrared
Transmission (NIT) principle. Each sample contains finely chopped pure
meat with different moisture, fat and protein contents.

If results from these data are used in a publication we want you to
mention the instrument and company name (Tecator) in the publication.
In addition, please send a preprint of your article to

Karin Thente, Tecator AB,
Box 70, S-263 21 Hoganas, Sweden

The data are available in the public domain with no responsability from
the original data source. The data can be redistributed as long as this
permission note is attached.
For more information about the instrument - call Perstorp Analytical's
representative in your area.
```

## Catalogs published before 2026-11

| Catalog(s) | Upstream source | License |
|---|---|---|
| fastsrb (@1, @2, @3) | FastSRB (viktmar/FastSRB; SRSD-lineage Feynman renderings, Matsubara et al. 2024 ranges) | MIT |
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
| v23-val, lample-charton-v23, lample-charton-v24, lample-charton-v24-bench | self-generated (flash-ansr project artifacts) | project license |
| bench-nv25 | self-generated expression corpora (drawn from flash-ansr's v25 training catalog, masked and unmasked) + corpus_external_soose (the expressions of soose-fc, soose-nc, soose-wc; NeuralSymbolicRegressionThatScales, MIT) | project license + MIT |

Share-alike catalogs (erbench-oeis, erbench-eponymous — OEIS-/Wikipedia-derived, CC BY-SA 4.0;
scilaws-real-share-alike — SciLaws-Bench tasks under CC BY-SA 3.0 and 4.0) are in the separate
psaegert/symbolic-data-assets-sa repo with their own LICENSE and attribution.

## srbf's 2026-11 benchmark catalogs

The license pass over every catalog of srbf's 2026-11 release: one row per catalog published here, with the source
its content was taken from and that source's license. Each catalog also carries its license, its citations (paper
and data) and what srbf changed (`srbf_changes`) in its metadata. A data catalog holding several datasets keeps only
the members whose license allows redistribution; the license column names theirs.

<!-- generated below: the publish tool's --notices, from tools/catalog_build/tags/licenses.json and copyrights.json -->

| Catalog | Upstream source | License |
|---|---|---|
| ai-descartes-real-data | IBM/AI-Descartes data/ (https://github.com/IBM/AI-Descartes @88bda46) for 5 tables; Cornelio et al. 2023 Nat. Commun. 14:1777 SI Supplementary Table 2 (https://doi.org/10.1038/s41467-023-37236-y) for the exoplanet table | MIT (IBM/AI-Descartes) + CC BY 4.0 (Cornelio et al. 2023, Nat. Commun. 14:1777, SI Table 2 exoplanet columns) |
| ai-feynman-2-0-modular-equations | Udrescu, Tan, Feng, Neto, Wu & Tegmark 2020, 'AI Feynman 2.0', NeurIPS 2020 / arXiv:2006.10782, Table 4 rows 5-16 | facts only: formulas cited from Udrescu et al. 2020 (AI Feynman 2.0, NeurIPS 33, arXiv:2006.10782), Table 4; no upstream license |
| ai-feynman-database-100-main-equations | Feynman Symbolic Regression Database (Udrescu & Tegmark 2020): FeynmanEquations.csv + Feynman_without_units files (space.mit.edu), units table | facts only: formulas, ranges and units cited from the Feynman Symbolic Regression Database (Udrescu & Tegmark 2020, Sci. Adv. 6:eaay2631); no upstream license |
| ai-hilbert-problem-set | IBM/AI-Hilbert data_generation/data_gen_mix.py and data_gen_kepler.py (@621b08b; Zenodo 10.5281/zenodo.11453179); formulas as printed in Cory-Wright et al. 2024, Nat. Commun. 15:5922 | MIT (IBM/AI-Hilbert) |
| ai-hilbert-real-data | IBM/AI-Hilbert main_problems notebook rows (https://github.com/IBM/AI-Hilbert @621b08b); Zenodo 10.5281/zenodo.11453179 | MIT (IBM/AI-Hilbert) |
| autora-synthetic-behavioural-science-models-points | AutoResearch/autora-synthetic v2.2.0 factory defaults (np.linspace grids; @d04b439) | MIT (AutoResearch/autora-synthetic) |
| black-hole-spiral-plane | Davis & Jin 2023, ApJL 956 L22, Table 1 machine-readable ASCII (https://doi.org/10.3847/2041-8213/acfa98) | CC BY 4.0 (Davis & Jin 2023, ApJL 956 L22, Table 1; doi:10.3847/2041-8213/acfa98) |
| bms-synthetic-prior-expression-and-bessel-functions | Guimera et al. 2020, Sci. Adv. 6:eaav6971 (main text + supplement aav6971_SM.pdf via Europe PMC PMC6994216), arXiv:2004.12157 | facts only: formulas cited from Guimera et al. 2020 (Bayesian machine scientist, Sci. Adv. 6:eaav6971, CC BY-NC 4.0); no text or tables reproduced |
| bms-synthetic-prior-expression-and-bessel-functions-points | Guimera et al. 2020, Sci. Adv. 6:eaav6971 (Bessel J_nu on [0,4]; srbf draws 100 seeded uniform points per order) | facts only: Bessel-function test points generated by srbf (Guimera et al. 2020, Sci. Adv. 6:eaav6971, CC BY-NC 4.0); no upstream data |
| breiman-i | Breiman, Friedman, Olshen & Stone 1984, CART, Ch. 8 (definition via Sauve & Tuleau-Malot 2011, Torgo's 2dplanes page, HeuristicLab BreimanOne.cs) | facts only: formula and input distribution cited from Breiman et al. 1984 (CART, Ch. 8); no upstream license |
| chen-luo-jiang-block-building-test-cases | Chen, Luo & Jiang, arXiv:1706.02281v3 Appendices A-B (Expert Syst. Appl. 2018) | facts only: formulas and ranges cited from Chen, Luo & Jiang 2018 (Expert Syst. Appl., arXiv:1706.02281); no upstream license |
| coevolved-fitness-predictors-test-problems-points | Schmidt & Lipson, GPTP IV 2006 Table 4 grids (F1-F3) and TEC 2008 Sec. V-C (100 evenly spaced points on [-2,2]) | facts only: functions and grids cited from Schmidt & Lipson 2006 (GPTP IV) and 2008 (IEEE TEVC 12:736); no upstream license |
| constant | dso-org/deep-symbolic-optimization benchmarks.csv (Petersen et al.) | BSD-3-Clause (dso-org/deep-symbolic-optimization benchmarks.csv); formulas and ranges as in each benchmark's first publication (facts) |
| cp3-bench | CP3-Origins/Things-to-bench cosmo_data/C1a-C6c.csv (https://github.com/CP3-Origins/Things-to-bench @69a45dd, v1.1.0) | MIT (CP3-Origins/Things-to-bench) |
| cp3-bench-standard-tuning-datasets-points | Thing & Koksbang 2025 (JCAP 01:040, arXiv:2406.15531) App. A Table 6 equidistant grids; Things-to-bench datasets/F*.csv | MIT (CP3-Origins/Things-to-bench; Thing & Koksbang 2025, arXiv:2406.15531) |
| cvgp-trigonometric-control-variable-datasets | Jiang & Xue, ECML-PKDD 2023 (arXiv:2306.08057) CVGP expression files; identical 1050 expressions in jiangnanhugo/egg-sr datasets/scibench (MIT) | MIT (jiangnanhugo/egg-sr scibench copy of the CVGP expressions; Jiang & Xue 2023, arXiv:2306.08057) |
| ddrsr-simple-test-and-complex-test | Deng et al., 'Deep Divide-and-Reduce in Symbolic Regression', arXiv:2608.02628v3, App. F Table 5 | facts only: formulas and ranges cited from Deng et al. 2026 (DDRSR, arXiv:2608.02628), Table 5; no upstream license |
| decomposable-nsr-synthetic-e1-e13 | NISL-MSU/MultiSetSR GenerateDatasets.py (@d5ba2ec); Morales & Sheppard, TMLR 2026 (arXiv:2511.04124) Table 2 | MIT (NISL-MSU/MultiSetSR; Morales & Sheppard 2026, arXiv:2511.04124) |
| dgsr-additional-feynman | Holt, Qian & van der Schaar, ICLR 2023 (arXiv:2401.00282) Table 35 + libs/sd3/dso/.../benchmarks.csv (@5dbf04d); base laws AI Feynman | CC BY 4.0 (Holt et al. 2023, DGSR, arXiv:2401.00282, Table 35) + BSD-3-Clause (benchmarks.csv, DSO fork in samholt/DeepGenerativeSymbolicRegression) |
| dgsr-feynman-d-2-d-5-subsets | Holt et al. 2023 (arXiv:2401.00282) Table 2 subsets + libs/sd3/dso/.../benchmarks.csv (@5dbf04d); base laws AI Feynman FSReD | CC BY 4.0 (Holt et al. 2023, DGSR, arXiv:2401.00282) + BSD-3-Clause (benchmarks.csv, DSO fork in samholt/DeepGenerativeSymbolicRegression) |
| dgsr-synthetic-d-12 | samholt/DeepGenerativeSymbolicRegression libs/sd3/dso/dso/task/regression/benchmarks.csv rows l_cd_12_* (@5dbf04d) | BSD-3-Clause (benchmarks.csv, DSO fork in samholt/DeepGenerativeSymbolicRegression; Holt et al. 2023, arXiv:2401.00282) |
| dso-benchmark-registry | dso-org/deep-symbolic-optimization dso/dso/task/regression/benchmarks.csv (@2069d4e, a49c228) | BSD-3-Clause (dso-org/deep-symbolic-optimization) |
| dso-benchmark-registry-points | dso-org/deep-symbolic-optimization benchmarks.csv Neat-6 (fit on integers 1..50, test 1..120) | BSD-3-Clause (dso-org/deep-symbolic-optimization) |
| e2e-in-domain-synthetic-validation-set | facebookresearch/symbolicregression generator (@c4144c9), frozen 10,000-function draw with the authors' seed | Apache-2.0 (facebookresearch/symbolicregression; Kamienny et al. 2022, arXiv:2204.10532) |
| eql-cart-pendulum | Sahoo, Lampert & Martius, ICML 2018 (arXiv:1806.07259) Eq. (12); input ranges from the authors' generator martius-lab/EQL EQL-DIV-ICML (@28521ae) | facts only: equations and input ranges cited from Sahoo, Lampert & Martius 2018 (ICML, arXiv:1806.07259) and their EQL generator; no upstream license |
| eql-formulas-f-1-f-4-and-division-formula | Martius & Lampert 2016 (arXiv:1610.02995) Eqs. (10)-(12); Sahoo et al. 2018 (arXiv:1806.07259) Eq. (11) | facts only: formulas and ranges cited from Martius & Lampert 2016 (arXiv:1610.02995) and Sahoo et al. 2018 (arXiv:1806.07259); no upstream license |
| eql-pendulum-vector-field | Martius & Lampert 2016, arXiv:1610.02995v1 (ICLR 2017 submission), pendulum task | facts only: formulas and ranges cited from Martius & Lampert 2016 (arXiv:1610.02995); no upstream license |
| eql-robotic-arm-forward-kinematics | Martius & Lampert 2016, arXiv:1610.02995 (kinematics tasks; formulas derived from the stated geometry) | facts only: task geometry and ranges cited from Martius & Lampert 2016 (arXiv:1610.02995); formulas derived by srbf; no upstream license |
| erbench-public-legacy-formulas | 18 legacy-family members (Korns, Nguyen, Neat, AI Feynman ...) taken from each family's own authors (Korns 2011, Uy et al. 2011, DSO benchmarks.csv, AI Feynman), mapped to ERBench ids | facts only: formulas and ranges cited from the original family authors (Korns 2011; Uy et al. 2011; Udrescu et al. 2020; DSO benchmarks.csv, BSD-3-Clause); ERBench ids for mapping only |
| errless-synthetic-posterior-set | ERRLESS, arXiv:2608.09617v2 App. B.1 Table 4 (code jaggbow/ERRLESS @b7ae784) | CC BY 4.0 (ERRLESS, arXiv:2608.09617, Table 4) |
| esr-cosmology | Cosmic-chronometer H(z) table, Moresco et al. 2022, Living Rev. Relativ. 25:6, Table 1 (arXiv:2201.07241), as esr/data/CC_Hubble.dat in DeaglanBartlett/ESR @23c3907 | CC BY 4.0 (Moresco et al. 2022, Living Rev. Relativ. 25:6, Table 1; arXiv:2201.07241) via DeaglanBartlett/ESR (MIT) |
| esr-cosmology-datasets | DeaglanBartlett/ESR examples/mock_cc.py (LCDM mock, @23c3907); Bartlett et al. 2024, IEEE TEVC 28:950 (arXiv:2211.11461) | MIT (DeaglanBartlett/ESR; Bartlett, Desmond & Ferreira 2024, IEEE TEVC 28:950, arXiv:2211.11461) |
| evogp-uci | UCI 409 Daily Demand Forecasting Orders (doi:10.24432/C5BC8T); UCI 9 Auto MPG (doi:10.24432/C5859H); California housing (Pace & Barry 1997) via the scikit-learn figshare mirror (https://figshare.com/articles/dataset/cal_housing_tgz/3829992) | CC BY 4.0 (UCI ML Repository: Daily Demand Forecasting Orders, doi:10.24432/C5BC8T; Auto MPG, doi:10.24432/C5859H) + CC BY 4.0 (figshare 3829992, California housing, Pace & Barry 1997) |
| evomo-sr-random-expression-datasets-p1-p4 | Rossetti et al., EvoMO-SR, arXiv:2609.36187v1 Table 3; anonymised code release random_functions/manifest.csv | CC BY 4.0 (Rossetti et al. 2026, EvoMO-SR, arXiv:2609.36187, Table 3) |
| falling-balls | isds-neu/SymbolicPhysicsLearner physics_task/data/*_train/_test.csv (@c0c18c9), from de Silva et al. 2020 Ball_drops_data.xls (briandesilva/discovery-of-physics-from-data @b79c343) | MIT (isds-neu/SymbolicPhysicsLearner; raw drops: briandesilva/discovery-of-physics-from-data, MIT) |
| feynman | Feynman Symbolic Regression Database (Udrescu & Tegmark 2020) | facts only: formulas and ranges from the Feynman Symbolic Regression Database (Udrescu & Tegmark 2020); its data files are not redistributed |
| friedman-functions | Friedman 1991, 'Multivariate Adaptive Regression Splines', Ann. Stat. 19:1 (revised tech report Aug 1990), Sec. 4.2-4.4 | facts only: test functions and input distributions cited from Friedman 1991 (Ann. Stat. 19:1-67); no upstream license |
| friedman-random-function-generator-instances | HeuristicLab FriedmanRandomFunction(.cs)/InstanceProvider (v3.3.16 @ecf4b5c), 12 instances regenerated from the provider seed; family defined by Friedman 2001 | facts only: 12 random-function instances generated with HeuristicLab's FriedmanRandomFunction provider (GPL-3.0 code, not copied; Friedman 2001, Ann. Stat. 29:1189) |
| gecco-2013-2017-real | Oliveira et al. 2018 compilation (GECCO 2018 Companion, doi:10.1145/3205651.3208293): 11 UCI tables (ids 1, 29, 162, 165, 186 x2, 242 x2, 243, 291, 294), StatLib Boston, mlbench Ozone (via laic-ufmg/gp-benchmarks), Dow Chemical + Tower (Dow), QSAR %F/LD50/%PPB via DALabNOVA/slim | CC BY 4.0 (UCI Machine Learning Repository members) and MIT (the QSAR members, DALabNOVA/slim) |
| gecco-2022-covid-ny | Zenodo 10.5281/zenodo.6842176 'Results files from the 2022 SRBench Competition' (W. La Cava 2022), srbench_competition_results.zip, members zenodo/data/stage2/data/135_NY_value_{cases,deaths,hosp}_14_14_{train,test}.csv | CC BY 4.0 (Zenodo 10.5281/zenodo.6842176, SRBench 2022 competition, La Cava et al.) |
| gecco-2022-exact-rediscovery | Zenodo 10.5281/zenodo.6842176 'Results files from the 2022 SRBench Competition' (W. La Cava 2022), srbench_competition_results.zip, members zenodo/data/stage1/data/*_exact_formula_*_data{,_train,_test}.csv (the source of record; the files are downloaded from their byte-identical copies in the GPL-3.0 cavalab/srbench srcomp branch, sha256 equal, checked 2026-11-09) | CC BY 4.0 (Zenodo 10.5281/zenodo.6842176, SRBench 2022 competition, La Cava et al.) |
| gecco-2022-extrapolation | Zenodo 10.5281/zenodo.6842176 'Results files from the 2022 SRBench Competition' (W. La Cava 2022), srbench_competition_results.zip, members zenodo/data/stage1/data/*_extrapolation_*_data*.csv | CC BY 4.0 (Zenodo 10.5281/zenodo.6842176, SRBench 2022 competition, La Cava et al.) |
| gecco-2022-feature-selection | Zenodo 10.5281/zenodo.6842176 'Results files from the 2022 SRBench Competition' (W. La Cava 2022), srbench_competition_results.zip, members zenodo/data/stage1/data/*_featureselection_*_data*.csv (the source of record; the files are downloaded from their byte-identical copies in the GPL-3.0 cavalab/srbench srcomp branch, sha256 equal, checked 2026-11-09) | CC BY 4.0 (Zenodo 10.5281/zenodo.6842176, SRBench 2022 competition, La Cava et al.) |
| gecco-2022-local-optima | Zenodo 10.5281/zenodo.6842176 'Results files from the 2022 SRBench Competition' (W. La Cava 2022), srbench_competition_results.zip, members zenodo/data/stage1/data/*_localopt_*_data*.csv | CC BY 4.0 (Zenodo 10.5281/zenodo.6842176, SRBench 2022 competition, La Cava et al.) |
| gecco-2022-noise | Zenodo 10.5281/zenodo.6842176 'Results files from the 2022 SRBench Competition' (W. La Cava 2022), srbench_competition_results.zip, members zenodo/data/stage1/data/*_noise_*_data*.csv | CC BY 4.0 (Zenodo 10.5281/zenodo.6842176, SRBench 2022 competition, La Cava et al.) |
| gecco-2023-performance | Zenodo 10.5281/zenodo.8283005 'SRBench Competition 2023 - Datasets, Scripts, and Results' (de Franca, Virgolin, Kamienny, Bomarito 2023), srbench-competition-2023.zip datasets/dataset_{1,2,3}{,_test}.csv | CC BY 4.0 (Zenodo 10.5281/zenodo.8283005, SRBench 2023 competition, de Franca et al.) |
| geobench | Xu, Ruan, Liu & Sun, 'GeoBench: A new benchmark on Symbolic Regression with Geometric Expressions', ICLR 2025 submission (OpenReview TqzNI4v9DT) | facts only: geometric formulas and ranges cited from Xu et al. 2024 (GeoBench, OpenReview TqzNI4v9DT); no upstream license |
| gfn-sr-noisy-near-twin-set | Li, Marinescu & Musslick, arXiv:2312.00396v1 Sec. 3.2 and Table 2 (NeurIPS 2023 workshop) | CC BY 4.0 (Li, Marinescu & Musslick 2023, GFN-SR, arXiv:2312.00396) |
| gp-benchmarks-sr-table | McDermott et al. 2012, 'Genetic Programming Needs Better Benchmarks', GECCO 2012, Table 3 (corrected v3 on gpbenchmarks.org); members from Koza, Nguyen, Korns, Keijzer, Vladislavleva, Pagie | facts only: formulas and ranges cited from McDermott et al. 2012 (GECCO, Table 3) and the original family papers; no upstream license |
| gp-benchmarks-sr-table-points | McDermott et al. 2012 (GECCO) Table 3 grids for Keijzer-6, Vladislavleva-3, Pagie-2 | facts only: grids cited from McDermott et al. 2012 (GECCO, Table 3) and the original family papers; no upstream license |
| gp-gomea-real-world | GP-GOMEA authors' preprocessed copies on Google Drive (Virgolin et al. 2021, EC 29(2), arXiv:1904.02050) of UCI Airfoil, Concrete, Energy (x2), Wine (x2), Yacht; Boston housing; Dow Chemical; Tower | CC BY 4.0 (UCI Machine Learning Repository) |
| gplearn-documentation-examples | gplearn documentation, Symbolic Regressor example (doc/examples.rst @64517a8) | BSD-3-Clause (trevorstephens/gplearn documentation example) |
| gpu-lm-inner-constant-recovery-problems | TensorConv/CuSR experiments/e3_admit_criterion/out/corpus_manifest.json (@414b905), 18 single-inner ADMIT records (arXiv:2609.03352 Sec. IV-E) | facts only: formulas and ranges cited from the CuSR e3 corpus manifest (TensorConv/CuSR; arXiv:2609.03352); no upstream license |
| grammarvae-points | dso-org/deep-symbolic-optimization benchmarks.csv (Petersen et al.) | BSD-3-Clause (dso-org/deep-symbolic-optimization benchmarks.csv); formulas and ranges as in each benchmark's first publication (facts) |
| gsgp-real-world | UCI Airfoil (291), Concrete (165), Parkinsons Telemonitoring (189), Concrete Slump (182); QSAR %F/%PPB/LD50 via DALabNOVA/slim (@aca7357) | CC BY 4.0 (UCI ML Repository: Airfoil Self-Noise, Concrete Compressive Strength, Parkinsons Telemonitoring, Concrete Slump Test; per-dataset DOIs) + MIT (DALabNOVA/slim QSAR tables; Archetti et al. 2007) |
| gsr-symset | Tohme, Liu & Youcef-Toumi, 'GSR', TMLR 2023 (arXiv:2205.15569v2), Table 20 (SymSet) | facts only: formulas and ranges cited from Tohme et al. 2023 (GSR, TMLR, arXiv:2205.15569), Table 20; no upstream license |
| hep-llm-sr-angular-distribution-problems | Morales-Alvarado, arXiv:2510.03397 (LO QED angular law, windows); alpha from arXiv:2508.00989 | CC BY 4.0 (Morales-Alvarado 2025, arXiv:2510.03397) |
| heuristiclab-physics-instances | HeuristicLab 'Physics Benchmark Problems' provider and descriptors (@c57538b): RocketFuelFlow, AircraftLift, FluidDynamics, AircraftMaximumLift (from Chen et al. 2018), SheetBendingProcess | facts only: formulas and ranges cited from the HeuristicLab Physics instance provider (heal-research/HeuristicLab, GPL-3.0 code, not copied) and Chen et al. 2018 (arXiv:1706.02281) |
| heuristiclab-regression-instance-providers | HeuristicLab regression instance providers (@c57538b): 16 fixed formulas taken from their defining authors (Keijzer 2003, Korns 2011, Vladislavleva 2009, Breiman 1984, Friedman 1991, Chen 2018) + 88 generator instances (Friedman random, variable networks, feature selection) | facts only: formulas cited from the defining authors and instances generated with HeuristicLab's providers (heal-research/HeuristicLab, GPL-3.0 code, not copied) |
| heuristiclab-regression-instance-providers-points | Keijzer-6 integer grid as defined by Keijzer 2003 / HeuristicLab provider | facts only: Keijzer-6 grid cited from Keijzer 2003 (EuroGP) as provided by HeuristicLab; no upstream license |
| hierbosss-single-atom | tsenftle/Metal-Oxide-LASSO-lo Step_1/Binding_energyV4.xlsx (@8501e29), data of O'Connor et al. 2018, Nat. Catal. 1:531-539 | MIT (tsenftle/Metal-Oxide-LASSO-lo; O'Connor, Jonayat, Janik & Senftle 2018, Nat. Catal. 1:531) |
| hierbosss-single-atom-catalysis-and-synthetic-polynomial | HierBOSSS, arXiv:2509.19710v1 eq. 5.2 / Sec. 5.1 and the archived HierBOSSS_simulated_example.R (MIT repo, deleted; Wayback) | CC BY 4.0 (Roy et al. 2025, HierBOSSS, arXiv:2509.19710v1) + MIT (Roy-SR-007/HierBOSSS) |
| jin | dso-org/deep-symbolic-optimization benchmarks.csv (Petersen et al.) | BSD-3-Clause (dso-org/deep-symbolic-optimization benchmarks.csv); formulas and ranges as in each benchmark's first publication (facts) |
| kan-feynman-no-units-subset | Liu et al., 'KAN: Kolmogorov-Arnold Networks', arXiv:2404.19756, Table 2 (27 dimensionless Feynman formulas) | CC BY 4.0 (Liu et al. 2024, KAN, arXiv:2404.19756, Table 2) |
| kan-toy-functions | Liu et al., KAN, arXiv:2404.19756 Sec. 3.1 (toy functions); input range from pykan create_dataset defaults | CC BY 4.0 (Liu et al. 2024, KAN, arXiv:2404.19756) + MIT (KindXiaoming/pykan) |
| keijzer | dso-org/deep-symbolic-optimization benchmarks.csv (Petersen et al.) | BSD-3-Clause (dso-org/deep-symbolic-optimization benchmarks.csv); formulas and ranges as in each benchmark's first publication (facts) |
| keijzer-points | Keijzer 2003 grids via the released keijzer catalog (formulas + ranges from DSO benchmarks.csv) | BSD-3-Clause (dso-org/deep-symbolic-optimization) |
| kim-et-al-eql-benchmark-functions | Kim et al. 2020, IEEE TNNLS (arXiv:1912.04825) benchmark table + samuelkim314/DeepSymReg benchmark.py (@57bbe2e) | MIT (samuelkim314/DeepSymReg) + CC BY 4.0 (Kim et al. 2020, IEEE TNNLS) |
| knowledge-integration-newtonian-cases | bilgesi/SR-LLM-Integration src/run_main_claim.py generator (@1508a37); Taskin, Xie & Lazebnik 2026, Sci. Rep. 16:1614 Eqs. (2)-(4) | Apache-2.0 (bilgesi/SR-LLM-Integration) + CC BY 4.0 (Taskin et al. 2026, Sci. Rep. 16:1614) |
| korns | dso-org/deep-symbolic-optimization benchmarks.csv (Petersen et al.) | BSD-3-Clause (dso-org/deep-symbolic-optimization benchmarks.csv); formulas and ranges as in each benchmark's first publication (facts) |
| koza | dso-org/deep-symbolic-optimization benchmarks.csv (Petersen et al.) | BSD-3-Clause (dso-org/deep-symbolic-optimization benchmarks.csv); formulas and ranges as in each benchmark's first publication (facts) |
| langlaw-perovskites | NAKONAKO4/langlaw data/{bulk_modulus,bandgap,oer_activity}.csv (@a69229c); Roy-SR-007/HierBOSSS perovskites_data/processed_data.csv (@0c6b856) | MIT (NAKONAKO4/langlaw, README licence statement; Roy-SR-007/HierBOSSS) ; underlying data CC BY 4.0 (Weng et al. 2020, Nat. Commun. 11:3513; NOMAD) |
| lasr-bigbench | trishullab/LibraryAugmentedSymbolicRegression.jl bigbench/csvs.zip (@de6b22d, lasr-experiments branch), derived from google/BIG-bench results | Apache-2.0 (trishullab/LibraryAugmentedSymbolicRegression.jl; derived from google/BIG-bench, Apache-2.0) |
| lasr-noisy-feynman-with-distractors | trishullab/LibraryAugmentedSymbolicRegression.jl data/FeynmanEquations.csv + experiments/*.py dummy recipe (@de6b22d); base laws = FSReD | Apache-2.0 (trishullab/LibraryAugmentedSymbolicRegression.jl; Grayeli et al. 2024, arXiv:2409.09359) |
| lasr-synthetic-equations | trishullab/LibraryAugmentedSymbolicRegression.jl experiments/dataset.py synthetic_equations() (@de6b22d) | Apache-2.0 (trishullab/LibraryAugmentedSymbolicRegression.jl; Grayeli et al. 2024, arXiv:2409.09359) |
| livermore | dso-org/deep-symbolic-optimization benchmarks.csv (Petersen et al.) | BSD-3-Clause (dso-org/deep-symbolic-optimization benchmarks.csv); formulas and ranges as in each benchmark's first publication (facts) |
| livermore2 | dso-org/deep-symbolic-optimization benchmarks.csv (Petersen et al.) | BSD-3-Clause (dso-org/deep-symbolic-optimization benchmarks.csv); formulas and ranges as in each benchmark's first publication (facts) |
| llm-lex-random-mathematica-functions | Harvey, Ruehle, Fraser-Taliente & Halverson, arXiv:2505.07956v1 Table 1 + Sec. 3.2 (MLST 7:045040) | CC BY 4.0 (Harvey et al. 2025, arXiv:2505.07956; Mach. Learn.: Sci. Technol. 7:045040) |
| llm-lex-random-mathematica-functions-points | Harvey et al. 2025 (arXiv:2505.07956): 100 evenly spaced points on [0.1, 5] | CC BY 4.0 (Harvey et al. 2025, arXiv:2505.07956) |
| llm-pysr-battery | XieZikai/LLM-PySR data/battery_cycle_life_log_zscore (@dce2fae), features derived from Severson et al. 2019 (Nat. Energy 4:383) raw cycling data on data.matr.io | MIT (XieZikai/LLM-PySR; features from Severson et al. 2019, Nat. Energy 4:383, data.matr.io) |
| llm-sr-bactgrow | deep-symbolic-mathematics/LLM-SR data/bactgrow/{train,test_id,test_ood}.csv (@41c2123) | MIT (deep-symbolic-mathematics/LLM-SR; Shojaee et al. 2025, ICLR) |
| llm-sr-stress-strain | deep-symbolic-mathematics/LLM-SR data/stressstrain (@41c2123), derived from Mendeley Data 10.17632/rd6jm9tyb6.1 (Aakash, Connors & Shields 2019, Al 6061-T651 stress-strain) | CC BY 4.0 (Mendeley Data 10.17632/rd6jm9tyb6.1, Aakash et al. 2019) via deep-symbolic-mathematics/LLM-SR (MIT) |
| mcdermott-predictive-modelling | QSAR %F/LD50/%PPB tables via DALabNOVA/slim (@aca7357); Mackey-Glass series mg.dat of R. Jang (Wayback copy of neural.cs.nthu.edu.tw/jang/dataset/mg/mg.dat); SILSO yearly sunspot number V1.0 (https://www.sidc.be/SILSO/DATA/ARCHIVE/V1.0/yearssn.dat) | MIT (DALabNOVA/slim QSAR tables) |
| mcdermott-table-1-predictive-modelling-sets-points | DyFor simulated series (piecewise law on x = 1..60), McDermott et al. 2012 Table 1 / Wagner et al. 2007 (IEEE TEVC) | facts only: piecewise law and integer grid cited from the DyFor benchmark (Wagner, Michalewicz, Khouja & McGregor 2007, IEEE TEVC 11:433); no upstream license |
| meier | dso-org/deep-symbolic-optimization benchmarks.csv (Petersen et al.) | BSD-3-Clause (dso-org/deep-symbolic-optimization benchmarks.csv); formulas and ranges as in each benchmark's first publication (facts) |
| multi-view-sr-synthetic-benchmarks | Russeil et al., 'Multi-View Symbolic Regression', GECCO 2024 (arXiv:2402.04298v4) Sec. 3.2 + erusseil/MvSR-analysis generate_toy.py (@ec5abd6) | CC BY 4.0 (Russeil et al. 2024, arXiv:2402.04298) |
| multi-view-sr-synthetic-benchmarks-points | Russeil et al. 2024 (arXiv:2402.04298) 20-point grids (np.arange in generate_toy.py) | CC BY 4.0 (Russeil et al. 2024, arXiv:2402.04298) |
| neat | dso-org/deep-symbolic-optimization benchmarks.csv (Petersen et al.) | BSD-3-Clause (dso-org/deep-symbolic-optimization benchmarks.csv); formulas and ranges as in each benchmark's first publication (facts) |
| neat-points | Neat-6 integer grid via the released neat catalog (DSO benchmarks.csv) | BSD-3-Clause (dso-org/deep-symbolic-optimization) |
| newtonbench | HKUST-KnowComp/NewtonBench modules/*/LAW_REGISTRY (@912a4ba); identical to arXiv:2510.07172 Table 5 | MIT (HKUST-KnowComp/NewtonBench) |
| nguyen | dso-org/deep-symbolic-optimization benchmarks.csv (Petersen et al.) | BSD-3-Clause (dso-org/deep-symbolic-optimization benchmarks.csv); formulas and ranges as in each benchmark's first publication (facts) |
| nguyen-c-points | Nguyen-2c of SPL: formula, arity and range in isds-neu/SymbolicPhysicsLearner regression_task/task_info.py (@c0c18c9); test grid E[0,100] | MIT (isds-neu/SymbolicPhysicsLearner) |
| nikuradse | PMLB nikuradse_1, nikuradse_2 (EpistasisLab/pmlb @7c1f4bd), Nikuradse 1933 rough-pipe data | MIT (PMLB) |
| nonic-points | Nonic via the released nonic catalog (DSO benchmarks.csv); seeded pool of 10,000 random points | BSD-3-Clause (dso-org/deep-symbolic-optimization) |
| nsrwh-only-five-variables-nc | SymposiumOrganization/ControllableNeuralSymbolicRegression test_set/only_five_variables_nc.csv (@3060cb8) | MIT (SymposiumOrganization/ControllableNeuralSymbolicRegression) |
| nsrwh-only-five-variables-wc | SymposiumOrganization/ControllableNeuralSymbolicRegression test_set/only_five_variables_wc.csv (@3060cb8) | MIT (SymposiumOrganization/ControllableNeuralSymbolicRegression) |
| nsrwh-train-nc | SymposiumOrganization/ControllableNeuralSymbolicRegression test_set/train_nc.csv (@3060cb8) | MIT (SymposiumOrganization/ControllableNeuralSymbolicRegression) |
| nsrwh-train-wc | SymposiumOrganization/ControllableNeuralSymbolicRegression test_set/train_wc.csv (@3060cb8) | MIT (SymposiumOrganization/ControllableNeuralSymbolicRegression) |
| nuclear-misr | munozariasjm/nuclear-misr Data/Experimental/{be,rc}_{train,test}.csv (@791e28c); values: AME2020 binding energies (Wang et al. 2021, Chin. Phys. C 45:030003) and IAEA charge radii (Angeli & Marinova 2013) | CC BY 3.0 (AME2020, Chin. Phys. C 45, 030003) |
| occamnet-analytic-and-non-analytic-suites | Dugan et al., 'OccamNet', arXiv:2007.10784v3 Tables 1, 2, 5 (cross-checked with OccamNet_Public analytic-benchmarks/Benchmarks.py) | facts only: target functions and ranges cited from Dugan et al. 2023 (OccamNet, arXiv:2007.10784); no upstream license |
| pagie-points | dso-org/deep-symbolic-optimization benchmarks.csv (Petersen et al.) | BSD-3-Clause (dso-org/deep-symbolic-optimization benchmarks.csv); formulas and ranges as in each benchmark's first publication (facts) |
| phye2e-llm-generated-synthetic-formulas | figshare PhyE2E_datas v1 (doi:10.6084/m9.figshare.29615831.v1), exprs_test_ranked.json (600 formulas) | Apache-2.0 (figshare 10.6084/m9.figshare.29615831, PhyE2E_datas; Ying et al. 2025, Nat. Mach. Intell. 7:1726) |
| phye2e-space-physics | figshare 'PhyE2E_datas' v1 (doi:10.6084/m9.figshare.29615831.v1; Ying et al. 2025, Nat. Mach. Intell. 7:1726): SN_m_tot_V2.0.txt (SILSO), 12 THEMIS/Geotail plasma quartile tables, solarDifferentialRotation.csv, 2 Fe X line tables (CHIANTI) | Apache-2.0 (the authors' figshare record) |
| physo-astrophysical-case-studies | Tenachi, Ibata & Diakogiannis 2023, ApJ 959:99 (arXiv:2303.03192) Appendices A-B; PhySO demos | CC BY 4.0 (Tenachi et al. 2023, ApJ 959:99, arXiv:2303.03192, Appendices A-B) + MIT (WassimTenachi/PhySO) |
| physo-class | WassimTenachi/PhySO physo/benchmark/ClassDataset/ClassEquations.csv (v1.1.0, Zenodo 10.5281/zenodo.11663147) | MIT (WassimTenachi/PhySO) |
| physo-milky-way-streams | WassimTenachi/PhySO demos/class_sr/demo_milky_way_streams/streams.csv (@bfbfa88) | MIT (WassimTenachi/PhySO) |
| poly | dso-org/deep-symbolic-optimization benchmarks.csv (Petersen et al.) | BSD-3-Clause (dso-org/deep-symbolic-optimization benchmarks.csv); formulas and ranges as in each benchmark's first publication (facts) |
| pse-high-dimensional-distractor-set | intell-sci-comput/PSE data/many_input/generate.py and synthetic_50d_datasets_with_gt (@7105cab); Ruan et al. 2026, Nat. Comput. Sci. 6:53 Supplementary Note 7 | MIT (intell-sci-comput/PSE; Ruan et al. 2026, arXiv:2407.04405) |
| pysr-empiricalbench | MilesCranmer/pysr_paper src/data/empirical_bench/{hubble,leavitt}.csv + inline tables in equations.yml (@b306874) | MIT (MilesCranmer/pysr_paper) |
| pysr-empiricalbench-formulas | MilesCranmer/pysr_paper src/data/empirical_bench/equations.yml (@b306874) | MIT (MilesCranmer/pysr_paper) |
| r-rationals-points | R1-R3, R1a-R3a via the released r-rationals catalog (DSO benchmarks.csv); seeded points | BSD-3-Clause (dso-org/deep-symbolic-optimization) |
| reproduction-bias-not-included-test-sets | Shun-0922/Mem-Bias-NSR test_datasets/original_datasets/{not_included,baseline}.json (@7a153ec) | MIT (Shun-0922/Mem-Bias-NSR; Sato & Sato 2025, arXiv:2505.22081) |
| rils-rols-random-instances | kartelj/rils-rols paper_resources/random_12345_truth/all.f (@54861c1) | MIT (kartelj/rils-rols; Kartelj & Djukanovic 2023, J. Big Data 10:71) |
| rsrm-constant-optimization-benchmark | Xu, Liu & Sun, RSRM, ICLR 2024 proceedings PDF, Appendix C.6 Tables S8-S9 (not in arXiv:2305.14656v1; not in the MIT repo) | facts only: formulas and ranges cited from Xu, Liu & Sun 2024 (RSRM, ICLR 2024, Appendix C.6, Table S8); no upstream license |
| rsrm-gaussian-cdf-extrapolation-test-points | Xu, Liu & Sun 2024 (RSRM, ICLR; arXiv:2305.14656) Gaussian-CDF curves on the 201-point grid E[-100,100,1] | facts only: curves and grid cited from Xu, Liu & Sun 2024 (RSRM, ICLR 2024, arXiv:2305.14656); no upstream license |
| scilaws-real | Hugging Face RealSR/SciLaws-Bench @15b9325, tasks/typeI/* (66 Type I tasks), per-task licences in dataset/LICENSES.md (github.com/yiyihum/SciLaws-Bench, MIT harness) | per problem (SciLaws-Bench dataset/LICENSES.md, RealSR/SciLaws-Bench): CC BY 2.5/3.0/4.0, CC0 1.0, public domain, MIT, etalab-2.0 |
| scimed-settling-sphere-configurations | LironSimon/SciMED data_generators/*.py and utills/consts.py (@2779587); Keren, Liberzon & Lazebnik 2023, Sci. Rep. 13:1249 Table 3 | MIT (LironSimon/SciMED) + CC BY 4.0 (Keren et al. 2023, Sci. Rep. 13:1249) |
| scrbenchmark | prescriptiveanalytics/SCR-Benchmarks (@2869bd5) SCRBenchmark/SRSDFeynman/feynman.py, Data/feynman_srsd_info.py, suite.py | MIT (prescriptiveanalytics/SCR-Benchmarks; SRSD laws: omron-sinicx/srsd-benchmark, MIT) |
| shape-constrained-real-world | folivetti/ITEA datasets/knowledge/{F05128,FlowStress,Cars} (@8669fd1, first commit; removed from the tree 2020-09-19), data of Kronberger et al. 2022 (Evol. Comput. 30(1):75) | CC BY 4.0 (UCI Auto MPG) |
| shape-constrained-sr-synthetic | Kronberger et al. 2022, Evol. Comput. 30(1):75 (arXiv:2103.15624v2) Tables 1 and 6: 3 FDE problems (Chen et al. 2018) and 4 FEY variants (FSReD) | facts only: formulas and input spaces cited from Kronberger et al. 2022 (Evol. Comput. 30(1):75, arXiv:2103.15624); no upstream license |
| sine-points | Sine on x = 0.0..6.2 step 0.1 via the released sine catalog (DSO benchmarks.csv) | BSD-3-Clause (dso-org/deep-symbolic-optimization) |
| sisso-octet-binaries | Ghiringhelli et al. 2015, PRL 114:105503, Supplemental Material Critical_Role_SI.pdf (82 octet binaries, RS vs ZB energy differences + atomic features) | CC BY 3.0 (Ghiringhelli et al. 2015, Phys. Rev. Lett. 114:105503, Supplemental Material) |
| spl-noise-data-scarcity-grid | Sun, Liu, Wang & Sun, 'Symbolic Physics Learner', ICLR 2023, arXiv:2205.13134v2 Sec. 4.1 (f(x) = 0.3x^3 + 0.5x^2 + 2x, x ~ U[-10, 10]) | facts only: one formula and its range from Sun et al. 2023 (Symbolic Physics Learner, ICLR, arXiv:2205.13134v2, a CC BY-SA 4.0 paper) |
| srbench-2.0-blackbox | PMLB (EpistasisLab/pmlb) datasets selected by SRBench 2.0 (Aldeia et al. 2025) | per dataset: CC BY 4.0 (UCI), generated data, OpenML BNG (public domain), StatLib permissions (503_wind; 505_tecator with its note), CC0 (titanic); PMLB's copies (MIT) |
| srbench-2.0-first-principles | PMLB first_principles_* datasets (EpistasisLab/pmlb), selected by SRBench 2.0 | per dataset: MIT (MilesCranmer/pysr_paper, EmpiricalBench); two from the unlicensed MvSR-analysis repo |
| srbench-2021-blackbox | PMLB regression datasets used by SRBench 2021 (La Cava et al. 2021), EpistasisLab/pmlb | per dataset: CC BY 4.0 (UCI), generated data, OpenML BNG (public domain), StatLib permissions (503_wind; 505_tecator with its note), CC0 (titanic); PMLB's copies (MIT) |
| srbench-2021-strogatz | PMLB strogatz_* datasets (EpistasisLab/pmlb), the ODE-Strogatz track of SRBench 2021 | MIT (PMLB) |
| superconductor-tc | henniggroup/symbolic-regression-utilities datasets/AllenDynes and examples/data_AD.txt (@11cf1f3), Allen-Dynes 1975 table (Xie et al. 2019, PRB 100:174513) | Apache-2.0 (henniggroup/symbolic-regression-utilities; Xie et al. 2019, PRB 100:174513; values from Allen & Dynes 1975, PRB 12:905) |
| surfacebench-explicit-surfaces | Kabra et al., SurfaceBench, TMLR 2026, arXiv:2511.10833v2 App. A.7 Table 7 | CC BY 4.0 (Kabra et al. 2026, SurfaceBench, arXiv:2511.10833, Table 7) |
| sym-q-ai-feynman-with-drift-terms | Sym-Q (Tian et al. 2025, Nat. Commun. 16:3930; arXiv:2402.05306) drift-term setup; EPFL-IMOS/Sym-Q utils.py BENCHMARK (@104a486); base AI-Feynman laws | MIT (EPFL-IMOS/Sym-Q) + CC0 1.0 (Tian et al., arXiv:2402.05306) |
| symantic-synthetic-set | Muthyala, Sorourifar, Peng & Paulson, SyMANTIC, arXiv:2502.03367v1 Table 1 + SI S1.1-S1.10 | facts only: formulas and ranges cited from Muthyala et al. 2025 (SyMANTIC, Ind. Eng. Chem. Res. 64:3354; arXiv:2502.03367, CC BY-NC-ND 4.0); no text or tables reproduced |
| symbolfit-lhc-spectra | hftsoi/symbolfit examples/datasets/{dijet,diphoton,trijet,dijet_pair,dimuon} (@3c81433), spectra from CMS/ATLAS HEPData records | MIT (hftsoi/symbolfit; spectra from HEPData records, CC0) |
| symbolicgpt-bivariate-set | SymbolicGPT 2-variable dataset (mojivalipour/symbolicgpt, HF DataAnalyticsLab/SymbolicGPT) with the Symbolic-Diffusion 7000-sample evaluation split (Tymkow et al. 2025, arXiv:2510.07570) | MIT (mojivalipour/symbolicgpt; HF DataAnalyticsLab/SymbolicGPT) |
| symbolicgpt-test-sets | SymbolicGPT test sets (Valipour et al. 2021): the authors' published instance files (repo @6aef07c, Dropbox, HF DataAnalyticsLab/SymbolicGPT) | MIT (mojivalipour/symbolicgpt; HF DataAnalyticsLab/SymbolicGPT) |
| symbolicregression-com-toy-test-suite-1 | E. Vladislavleva's symbolicregression.com 'Test Suite 1' page (Wayback 2013-12-08), formulas in Mathematica syntax; Vladislavleva 2008 thesis | facts only: formulas cited from Vladislavleva's symbolicregression.com Toy Test Suite 1 (2008-2013) and Vladislavleva et al. 2009 (IEEE TEVC); no upstream license |
| symformer-in-domain-synthetic-evaluation | SymFormer generator (vastlik/symformer @35bc93c), a frozen draw with the authors' configs; Vastl et al. 2024, IEEE Access 12:37840 | CC BY 4.0 (Vastl et al. 2024, SymFormer, IEEE Access 12:37840); formulas drawn by srbf from the authors' generator |
| symtree-benchmark-functions | de Franca 2018, Information Sciences 442:18 (arXiv:1801.01807) Table 1 + Sec. 5; author's notebook in folivetti/ITSR (GPL-3.0) | facts only: functions and ranges cited from de Franca 2018 (Inf. Sci. 442:18, arXiv:1801.01807) and the original Kattan et al. papers; no upstream license |
| tisr-air-density | Frotscher et al. 2023, Int. J. Thermophys. 44:105 (doi:10.1007/s10765-023-03197-z), ESM1 Tables 4-5 (MS grid, MS densities at experimental state points, experimental densities); experimental values originate in von Preetzmann et al. 2021, Int. J. Thermophys. 42:127 (doi:10.1007/s10765-021-02871-4), Table 2; TU Berlin DepositOnce copy doi:10.14279/depositonce-18496 | CC BY 4.0 (Frotscher et al. 2023, Int. J. Thermophys. 44:105, ESM1; experimental densities: von Preetzmann et al. 2021, Int. J. Thermophys. 42:127, Table 2) |
| torchsisso-specific-energy | PaulsonLab/TorchSISSO examples/Specific_Energy/{train.dat,redox_test_pade.csv} (pinned @acbf6ee, which carries the MIT LICENSE; blobs identical to f869f95, the last commit touching the folder) | MIT (PaulsonLab/TorchSISSO) |
| torchsisso-synthetic-expressions | Muthyala, Sorourifar & Paulson, TorchSISSO, arXiv:2410.01752 Table 1 + Sec. 4.1 (Digit. Chem. Eng. 13:100198) | facts only: formulas and ranges cited from Muthyala et al. 2024 (TorchSISSO, Digit. Chem. Eng. 13:100198; arXiv:2410.01752, CC BY-NC-ND 4.0); no text or tables reproduced |
| treloar-rubber | Linka & Kuhl 2023, Comput. Methods Appl. Mech. Eng. 403:115731, Table 2 (arXiv:2210.02202v2), tabulating Treloar 1944 (Trans. Faraday Soc. 40:59) at 20 C and 50 C | CC BY 4.0 (Linka & Kuhl 2023, Comput. Methods Appl. Mech. Eng. 403:115731, Table 2; data: Treloar 1944, Trans. Faraday Soc. 40:59) |
| vladislavleva | dso-org/deep-symbolic-optimization benchmarks.csv (Petersen et al.) | BSD-3-Clause (dso-org/deep-symbolic-optimization benchmarks.csv); formulas and ranges as in each benchmark's first publication (facts) |
| vladislavleva-points | Vladislavleva-2/-3 meshes (Vladislavleva et al. 2009, IEEE TEVC, Table I) via the released vladislavleva catalog | BSD-3-Clause (dso-org/deep-symbolic-optimization) |
| white-et-al-2013-proposed-sr-set-points | Keijzer-6 grid as specified by White et al. 2013 (GPEM 14:3) Table 5 | facts only: Keijzer-6 grid cited from White et al. 2013 (GPEM 14:3) and Keijzer 2003; no upstream license |

## Copyright notices

The copyright notice of every MIT, BSD-3-Clause and Apache-2.0 source the catalogs published here took content from, as the source's license file states it. The license texts are in LICENSES/.

| Source | License | Copyright notice | Catalogs |
|---|---|---|---|
| [AutoResearch/autora-synthetic](https://github.com/AutoResearch/autora-synthetic/blob/main/LICENSE) | MIT | Copyright (c) 2023 Autonomous Empirical Research Initiative | autora-synthetic-behavioural-science-models-points |
| [bilgesi/SR-LLM-Integration](https://github.com/bilgesi/SR-LLM-Integration/blob/main/LICENSE) | Apache-2.0 | none stated (the license file carries no copyright line; no NOTICE file) | knowledge-integration-newtonian-cases |
| [briandesilva/discovery-of-physics-from-data](https://github.com/briandesilva/discovery-of-physics-from-data/blob/master/LICENSE) | MIT | Copyright (c) 2020 Brian de SIlva | falling-balls |
| [CP3-Origins/Things-to-bench](https://github.com/CP3-Origins/Things-to-bench/blob/master/LICENSE) | MIT | Copyright (c) 2023 CP3-Origins | cp3-bench, cp3-bench-standard-tuning-datasets-points, cp3-blackbox, cp3-cosmo |
| [DALabNOVA/slim](https://github.com/DALabNOVA/slim/blob/main/LICENSE) | MIT | Copyright (c) 2024 DALabNOVA | gecco-2013-2017-real, gsgp-real-world, mcdermott-predictive-modelling |
| [DeaglanBartlett/ESR](https://github.com/DeaglanBartlett/ESR) | MIT | Copyright 2022 Deaglan J. Bartlett (no license file; the README states the MIT licence and this copyright line) | esr-cosmology, esr-cosmology-datasets |
| [deep-symbolic-mathematics/LLM-SR](https://github.com/deep-symbolic-mathematics/LLM-SR/blob/main/LICENSE) | MIT | Copyright (c) 2024 [Your Name or Organization] (as the license file states it: the project left the template's holder unfilled) | llm-sr-bactgrow, llm-sr-stress-strain |
| [dso-org/deep-symbolic-optimization](https://github.com/dso-org/deep-symbolic-optimization/blob/master/LICENSE) | BSD-3-Clause | Copyright (c) 2018, Lawrence Livermore National Security, LLC | constant, dgsr-additional-feynman, dgsr-feynman-d-2-d-5-subsets, dgsr-synthetic-d-12, dso-benchmark-registry, dso-benchmark-registry-points, grammarvae, grammarvae-points, jin, keijzer, keijzer-points, korns, koza, livermore, livermore2, meier, neat, neat-points, nguyen, nonic, nonic-points, pagie, pagie-points, poly, r-rationals, r-rationals-points, sine, sine-points, vladislavleva, vladislavleva-points |
| [EPFL-IMOS/Sym-Q](https://github.com/EPFL-IMOS/Sym-Q/blob/main/LICENSE) | MIT | Copyright (c) 2023 Wenqi Zhou | sym-q-ai-feynman-with-drift-terms |
| [EpistasisLab/pmlb](https://github.com/EpistasisLab/pmlb/blob/master/LICENSE) | MIT | Copyright (c) 2016 Epistasis Lab at UPenn | first-principles, nikuradse, srbench-2.0-blackbox, srbench-2021-blackbox, srbench-2021-strogatz, srbench2-blackbox |
| [EquationDiscovery/Equation_Recovery_Benchmark](https://huggingface.co/datasets/EquationDiscovery/Equation_Recovery_Benchmark) | MIT | none stated (ERBench (Kahlmeyer, Voigt, Habeck & Giesen) states MIT for SynEq and PHYBench without a copyright line) | erbench-phybench, erbench-syneq |
| [facebookresearch/symbolicregression](https://github.com/facebookresearch/symbolicregression/blob/main/LICENSE) | Apache-2.0 | Copyright 2020 - present, Facebook, Inc | e2e-in-domain-synthetic-validation-set |
| [figshare 10.6084/m9.figshare.29615831 (PhyE2E_datas, Ying et al. 2025)](https://doi.org/10.6084/m9.figshare.29615831.v1) | Apache-2.0 | none stated (Apache 2.0 per the figshare record (code: Jie0618/PhysicsRegression); no copyright line or NOTICE file) | phye2e-llm-generated-synthetic-formulas, phye2e-space-physics |
| [google/BIG-bench](https://github.com/google/BIG-bench/blob/main/LICENSE) | Apache-2.0 | none stated (the license file carries no copyright line; no NOTICE file) | lasr-bigbench |
| [henniggroup/symbolic-regression-utilities](https://github.com/henniggroup/symbolic-regression-utilities/blob/master/LICENSE) | Apache-2.0 | none stated (the license file carries no copyright line; no NOTICE file) | superconductor-tc |
| [hftsoi/symbolfit](https://github.com/hftsoi/symbolfit/blob/main/LICENSE) | MIT | Copyright (c) 2024 Ho Fung Tsoi | symbolfit-lhc-spectra |
| [HKUST-KnowComp/NewtonBench](https://github.com/HKUST-KnowComp/NewtonBench/blob/main/LICENSE) | MIT | Copyright (c) 2025 HKUST-KnowComp | newtonbench |
| [IBM/AI-Descartes](https://github.com/IBM/AI-Descartes/blob/main/LICENSE) | MIT | Copyright (c) 2022 International Business Machines | ai-descartes, ai-descartes-real-data |
| [IBM/AI-Hilbert](https://github.com/IBM/AI-Hilbert/blob/main/LICENSE) | MIT | Copyright (c) 2024 International Business Machines | ai-hilbert-problem-set, ai-hilbert-real-data |
| [intell-sci-comput/PSE](https://github.com/intell-sci-comput/PSE/blob/main/LICENSE) | MIT | Copyright (c) 2024 intell-sci-comput | pse-high-dimensional-distractor-set |
| [isds-neu/SymbolicPhysicsLearner](https://github.com/isds-neu/SymbolicPhysicsLearner/blob/main/LICENSE) | MIT | Copyright (c) 2023 isds-neu | falling-balls, nguyen-c-points |
| [jiangnanhugo/egg-sr](https://github.com/jiangnanhugo/egg-sr/blob/main/LICENSE) | MIT | Copyright (c) 2014 | cvgp-trigonometric-control-variable-datasets |
| [kartelj/rils-rols](https://github.com/kartelj/rils-rols/blob/master/LICENSE) | MIT | Copyright (c) 2016 Epistasis Lab at UPenn | rils-rols-random-instances |
| [KindXiaoming/pykan](https://github.com/KindXiaoming/pykan/blob/master/LICENSE) | MIT | Copyright (c) 2024 Ziming Liu | kan-toy-functions |
| [LironSimon/SciMED](https://github.com/LironSimon/SciMED/blob/master/LICENSE) | MIT | Copyright (c) 2022 Teddy Lazebnik | scimed-settling-sphere-configurations |
| [MilesCranmer/pysr_paper](https://github.com/MilesCranmer/pysr_paper/blob/main/LICENSE) | MIT | Copyright (c) 2020-2023 @MilesCranmer | pysr-empiricalbench, pysr-empiricalbench-formulas, srbench-2.0-first-principles |
| [mojivalipour/symbolicgpt](https://github.com/mojivalipour/symbolicgpt/blob/master/LICENSE.txt) | MIT | Copyright (c) 2020 Andrej Karpathy (minGPT), 2021 Mojtaba Valipour (symblicGPT) | symbolicgpt-bivariate-set, symbolicgpt-test-sets |
| [NISL-MSU/MultiSetSR](https://github.com/NISL-MSU/MultiSetSR/blob/master/LICENSE) | MIT | Copyright (c) 2024 Montana State University (the license file states YEAR: 2024, COPYRIGHT HOLDER: Montana State University) | decomposable-nsr-synthetic-e1-e13 |
| [omron-sinicx/srsd-benchmark](https://github.com/omron-sinicx/srsd-benchmark/blob/main/LICENSE) | MIT | Copyright (c) 2022 OMRON SINIC X Corporation | scrbenchmark, srsd-dummy |
| [PaulsonLab/TorchSISSO](https://github.com/PaulsonLab/TorchSISSO/blob/main/LICENSE) | MIT | Copyright (c) 2025 Madhav Reddy Muthyala | torchsisso-specific-energy |
| [prescriptiveanalytics/SCR-Benchmarks](https://github.com/prescriptiveanalytics/SCR-Benchmarks/blob/main/LICENSE) | MIT | none stated (the license file carries no copyright line) | scrbenchmark |
| [psaegert/symbolic-data](https://github.com/psaegert/symbolic-data/blob/main/LICENSE) | MIT | Copyright (c) 2024 Paul Saegert | bench-nv25, lample-charton-v23, lample-charton-v24, lample-charton-v24-bench, v23-val |
| [RealSR/SciLaws-Bench](https://huggingface.co/datasets/RealSR/SciLaws-Bench) | MIT | none stated (MIT for the tasks magnet_3c90_core_loss__log_P_vol and ocean_mesoscale_eddy_closure__Sx per SciLaws-Bench's dataset/LICENSES.md, which names no copyright holder) | scilaws-real |
| [Roy-SR-007/HierBOSSS](https://github.com/Roy-SR-007/HierBOSSS/blob/main/LICENSE) | MIT | Copyright (c) 2026 Authors of HierBOSSS [Somjit Roy, Pritam Dey, Bani K. Mallick, and Debdeep Pati] (langlaw's README states the MIT License without a copyright line; its data come from HierBOSSS) | hierbosss-single-atom-catalysis-and-synthetic-polynomial, langlaw-perovskites |
| [samuelkim314/DeepSymReg](https://github.com/samuelkim314/DeepSymReg) | MIT | none stated (no license file; the README states: All content in this repository is licensed under the MIT license.) | kim-et-al-eql-benchmark-functions |
| [scipy/scipy](https://github.com/scipy/scipy/blob/main/LICENSE.txt) | BSD-3-Clause | Copyright (c) 2001-2002 Enthought, Inc. 2003, SciPy Developers. | erbench-densities |
| [Shun-0922/Mem-Bias-NSR](https://github.com/Shun-0922/Mem-Bias-NSR/blob/main/LICENSE) | MIT | Copyright (c) 2021 SymposiumOrganization | reproduction-bias-not-included-test-sets |
| [SymposiumOrganization/ControllableNeuralSymbolicRegression](https://github.com/SymposiumOrganization/ControllableNeuralSymbolicRegression/blob/main/LICENSE) | MIT | Copyright (c) 2021 SymposiumOrganization | nsrwh-only-five-variables-nc, nsrwh-only-five-variables-wc, nsrwh-train-nc, nsrwh-train-wc |
| [SymposiumOrganization/NeuralSymbolicRegressionThatScales](https://github.com/SymposiumOrganization/NeuralSymbolicRegressionThatScales/blob/main/LICENSE) | MIT | Copyright (c) 2021 SymposiumOrganization | bench-nv25, soose-fc, soose-nc, soose-wc |
| [trevorstephens/gplearn](https://github.com/trevorstephens/gplearn/blob/main/LICENSE) | BSD-3-Clause | Copyright (c) 2015-2026, Trevor Stephens | gplearn-documentation-examples |
| [trishullab/LibraryAugmentedSymbolicRegression.jl](https://github.com/trishullab/LibraryAugmentedSymbolicRegression.jl/blob/master/LICENSE) | Apache-2.0 | Copyright 2020 Miles Cranmer | lasr-bigbench, lasr-noisy-feynman-with-distractors, lasr-synthetic-equations |
| [tsenftle/Metal-Oxide-LASSO-lo](https://github.com/tsenftle/Metal-Oxide-LASSO-lo/blob/master/LICENSE) | MIT | Copyright (c) 2018 tsenftle | hierbosss-single-atom |
| [viktmar/FastSRB](https://github.com/viktmar/FastSRB/blob/main/LICENSE) | MIT | Copyright (c) 2025 Viktor Martinek | fastsrb, srsd-dummy |
| [WassimTenachi/PhySO](https://github.com/WassimTenachi/PhySO/blob/main/LICENSE) | MIT | Copyright (c) 2023 Wassim Tenachi | feynman-bonus, physo-astro, physo-astrophysical-case-studies, physo-class, physo-milky-way-streams, physo-streams |
| [XieZikai/LLM-PySR](https://github.com/XieZikai/LLM-PySR/blob/main/LICENSE) | MIT | Copyright (c) 2026 Z.X. | llm-pysr-battery |

### NOTICE of dso-org/deep-symbolic-optimization

```
This work was produced under the auspices of the U.S. Department of
Energy by Lawrence Livermore National Laboratory under Contract
DE-AC52-07NA27344.

This work was prepared as an account of work sponsored by an agency of
the United States Government. Neither the United States Government nor
Lawrence Livermore National Security, LLC, nor any of their employees
makes any warranty, expressed or implied, or assumes any legal liability
or responsibility for the accuracy, completeness, or usefulness of any
information, apparatus, product, or process disclosed, or represents that
its use would not infringe privately owned rights.

Reference herein to any specific commercial product, process, or service
by trade name, trademark, manufacturer, or otherwise does not necessarily
constitute or imply its endorsement, recommendation, or favoring by the
United States Government or Lawrence Livermore National Security, LLC.

The views and opinions of authors expressed herein do not necessarily
state or reflect those of the United States Government or Lawrence
Livermore National Security, LLC, and shall not be used for advertising
or product endorsement purposes.
```
