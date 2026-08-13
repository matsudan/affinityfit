# Changelog

## [0.4.0](https://github.com/matsudan/affinityfit/compare/v0.3.0...v0.4.0) (2026-08-13)


### ⚠ BREAKING CHANGES

* `ci="bootstrap"` now returns an undetermined interval (`bounded=False`) for a rank-deficient fit instead of a narrow one.
* `aicc` now returns a smaller (more negative, or less positive) value for the same fit, and requires one more data point than before to be finite rather than infinite. `DiagnosticCode.ONE_POINT_NEAR_KD` is removed; a single measurement near *K*<sub>d</sub> now reports `NO_POINTS_NEAR_KD` instead.
* `DiagnosticCode.POOR_FIT` is removed. `DiagnosticCode.NO_FIT` now fires based on the F-test above rather than `R² < 0.5`. `diagnose()` no longer takes an `r_squared` argument.
* `load_csv` is removed. Read the CSV with `numpy.loadtxt`, `pandas.read_csv`, or equivalent, and pass the resulting arrays to `fit()` or `Dataset` directly.

### Features

* replace R^2 thresholds with an F-test for NO_FIT ([#25](https://github.com/matsudan/affinityfit/issues/25)) ([169280b](https://github.com/matsudan/affinityfit/commit/169280b1f11f1cfe84c80fa06a6eed6d7bcffa79))


### Bug Fixes

* correct AICc small-sample correction and diagnostic role checks ([#27](https://github.com/matsudan/affinityfit/issues/27)) ([d8e105b](https://github.com/matsudan/affinityfit/commit/d8e105be2e5def388c1f25334daba4c431a5f0bc))
* expand the range of the concentration axis on the example plot ([#24](https://github.com/matsudan/affinityfit/issues/24)) ([1911469](https://github.com/matsudan/affinityfit/commit/191146912c0c43a1c1353a993871733d6e6c47e0))
* report bootstrap intervals as undetermined for a rank-deficient fit ([#28](https://github.com/matsudan/affinityfit/issues/28)) ([71a263a](https://github.com/matsudan/affinityfit/commit/71a263ac3630bf1ad3ebd6697e7a514d3dcb8535))


### Code Refactoring

* remove load_csv ([#22](https://github.com/matsudan/affinityfit/issues/22)) ([96c27c5](https://github.com/matsudan/affinityfit/commit/96c27c5764ff647c9f677e56338794795aef8437))

## [0.3.0](https://github.com/matsudan/affinityfit/compare/v0.2.0...v0.3.0) (2026-08-12)


### ⚠ BREAKING CHANGES

* a `UserWarning` is emitted when `receptor_conc` is unavailable, since the bias of the standard form then cannot be assessed. Pass `receptor_conc=` to silence it and to get the exact correction.

### Features

* add the exact IC50-to-Ki correction for a depleting receptor ([#20](https://github.com/matsudan/affinityfit/issues/20)) ([e2c1d66](https://github.com/matsudan/affinityfit/commit/e2c1d661dd336f4d95626c0bc543b9fd097fa07e))


### Bug Fixes

* write the Cheng-Prusoff slope warning in English ([#15](https://github.com/matsudan/affinityfit/issues/15)) ([c21b3ec](https://github.com/matsudan/affinityfit/commit/c21b3ec501d5d83d5bcd060b6c8d0a539993fcd9))


### Documentation

* apply typographic conventions to the README ([#18](https://github.com/matsudan/affinityfit/issues/18)) ([8456167](https://github.com/matsudan/affinityfit/commit/84561670f352abc24eb36b29c49ca1f9756be441))
* correct README statements that disagree with the implementation ([#14](https://github.com/matsudan/affinityfit/issues/14)) ([530957c](https://github.com/matsudan/affinityfit/commit/530957c9b9385482e1b52514a7aab04a8650fba0))
* tighten README prose and give quantitative claims their conditions ([#17](https://github.com/matsudan/affinityfit/issues/17)) ([bb7bceb](https://github.com/matsudan/affinityfit/commit/bb7bceb66b7dd8116bfa12a508390b902b1f91dd))
* use a synthetic value in the rounding example ([#19](https://github.com/matsudan/affinityfit/issues/19)) ([c7e65de](https://github.com/matsudan/affinityfit/commit/c7e65de8ba37f78db71871f90320da2f5813ebe4))

## [0.2.0](https://github.com/matsudan/affinityfit/compare/v0.1.0...v0.2.0) (2026-08-11)


### Features

* add DiagnosticCode enum for discoverable diagnostic codes ([#12](https://github.com/matsudan/affinityfit/issues/12)) ([b63caa0](https://github.com/matsudan/affinityfit/commit/b63caa091f5b792aa45cf5179c869ce9a789853e))
* add ic50 and tight_binding models with depletion-aware diagnostics ([#9](https://github.com/matsudan/affinityfit/issues/9)) ([88a88ec](https://github.com/matsudan/affinityfit/commit/88a88ecda00215e247e6f873a72980551721e66b))
* add structured diagnostics API ([#11](https://github.com/matsudan/affinityfit/issues/11)) ([135e288](https://github.com/matsudan/affinityfit/commit/135e288f16eccb04a71bdfca92bd67c74d425022))


### Bug Fixes

* make profile confidence intervals scale-invariant ([#10](https://github.com/matsudan/affinityfit/issues/10)) ([f8ac624](https://github.com/matsudan/affinityfit/commit/f8ac6247df1c02982e3bbcc344053d8cd2ee2a71))


### Documentation

* translate README ([#7](https://github.com/matsudan/affinityfit/issues/7)) ([f15ac15](https://github.com/matsudan/affinityfit/commit/f15ac15c6c23ef502970e9f437f5246f9f0223df))

## 0.1.0 (2026-08-10)


### ⚠ BREAKING CHANGES

* rename package to affinityfit ([#2](https://github.com/matsudan/affinityfit/issues/2))

### Features

* add core fitting library ([12b5e08](https://github.com/matsudan/affinityfit/commit/12b5e080bfb85af79a78fb2b75731d5dba1dd4bb))
* expose raw statistics ([092d301](https://github.com/matsudan/affinityfit/commit/092d301a2605895603c1233e870578fbdc53ec29))


### Bug Fixes

* replace fitting terminology ([31583cb](https://github.com/matsudan/affinityfit/commit/31583cbd88702a552fa38ed9207dd605a577d554))
* reset release-please manifest to no prior release ([#5](https://github.com/matsudan/affinityfit/issues/5)) ([7ca66fa](https://github.com/matsudan/affinityfit/commit/7ca66fa00fca0859a1ff2c68ee319f2ccb0abae4))


### Documentation

* add plotting sample and matplotlib-free smoke script ([779b6e6](https://github.com/matsudan/affinityfit/commit/779b6e6bffd89ddf3762456a030e4d85d110e318))
* add README ([234cc4e](https://github.com/matsudan/affinityfit/commit/234cc4e1d7c0d0168c52fb1184e2aaee616329d0))
* fix example script comments ([dcabdad](https://github.com/matsudan/affinityfit/commit/dcabdadae8aeab6d470585c03cf504e8f5431060))
* update homepage URL after the GitHub repository rename ([#3](https://github.com/matsudan/affinityfit/issues/3)) ([2586fd9](https://github.com/matsudan/affinityfit/commit/2586fd9c2cf251a234a361d96c0e3b70ddfa3cbf))
* update README with refined scope and roadmap ([bed2716](https://github.com/matsudan/affinityfit/commit/bed2716f55a7255ae603bfe7eeef639b7a37a060))


### Miscellaneous Chores

* rename package to affinityfit ([#2](https://github.com/matsudan/affinityfit/issues/2)) ([5a0846e](https://github.com/matsudan/affinityfit/commit/5a0846e98fa16c22fd57b5c7f777b8481720c3eb))
