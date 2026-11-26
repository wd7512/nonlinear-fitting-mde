# Gate baseline, 2026-11-26

Rust policies on the frozen gate tiers, before any change to `auto`. Apple Silicon, one thread.
Tier files are not committed (they embed third-party data); `symreg-rs gate build` recreates
them from local runs. References depend on those runs, so check the hashes before comparing.

```
986be03c8989bdd1dcaafcb0f60c83b565fc84fab239f9ed3eb01529da08a29d  dev.json
da15dd32e3af0c41478a97d56c5d1c16b04c2cf19e16d5e62dd4d1d207996832  quick.json
10a8226aa65f276fed95f28af3d0694ade777861991b14ffa7ce74a173167b4c  rank.json
ed878ce5103d6bfdac280d91cd5c6d20a92802e0d5aa90da74e38ff27ba05cbf  test.json
```

## quick

success (mean over suites; per-suite columns at the main checkpoint)

| solver          | 1 ms   | 5 ms   | 10 ms   | 25 ms   | 5 ms nist_strd   | 5 ms mgh   | 5 ms synthetic_scale   |
|:----------------|:-------|:-------|:--------|:--------|:-----------------|:-----------|:-----------------------|
| rs_auto         | 85.4%  | 89.6%  | 91.7%   | 91.7%   | 100.0%           | 75.0%      | 93.8%                  |
| rs_gn_lm        | 75.0%  | 87.5%  | 89.6%   | 89.6%   | 93.8%            | 75.0%      | 93.8%                  |
| rs_lm           | 77.1%  | 87.5%  | 87.5%   | 89.6%   | 93.8%            | 75.0%      | 93.8%                  |
| rs_varpro_gn_lm | 83.3%  | 87.5%  | 89.6%   | 89.6%   | 93.8%            | 75.0%      | 93.8%                  |

change vs rs_auto (percentage points)

| solver          |   1 ms |   5 ms |   10 ms |   25 ms |   5 ms nist_strd |   5 ms mgh |   5 ms synthetic_scale |
|:----------------|-------:|-------:|--------:|--------:|-----------------:|-----------:|-----------------------:|
| rs_gn_lm        |  -10.4 |   -2.1 |    -2.1 |    -2.1 |             -6.2 |          0 |                      0 |
| rs_lm           |   -8.3 |   -2.1 |    -4.2 |    -2.1 |             -6.2 |          0 |                      0 |
| rs_varpro_gn_lm |   -2.1 |   -2.1 |    -2.1 |    -2.1 |             -6.2 |          0 |                      0 |

speed and failures

| solver          |   median fit ms |   median ms to target |   failures |   overruns (>10%) |   fits |
|:----------------|----------------:|----------------------:|-----------:|------------------:|-------:|
| rs_auto         |           0.073 |                 0.028 |          0 |                 0 |     48 |
| rs_gn_lm        |           0.095 |                 0.037 |          0 |                 0 |     48 |
| rs_lm           |           0.098 |                 0.042 |          0 |                 0 |     48 |
| rs_varpro_gn_lm |           0.067 |                 0.021 |          0 |                 0 |     48 |

## dev

success (mean over suites; per-suite columns at the main checkpoint)

| solver          | 1 ms   | 5 ms   | 10 ms   | 25 ms   | 5 ms nist_strd   | 5 ms mgh   | 5 ms fitbenchmarking   | 5 ms sagefit_spb   | 5 ms synthetic_scale   | 5 ms stringer2019_gratings   |
|:----------------|:-------|:-------|:--------|:--------|:-----------------|:-----------|:-----------------------|:-------------------|:-----------------------|:-----------------------------|
| rs_auto         | 72.4%  | 86.0%  | 87.0%   | 87.7%   | 100.0%           | 89.5%      | 100.0%                 | 52.4%              | 95.3%                  | 78.7%                        |
| rs_gn_lm        | 64.3%  | 83.2%  | 85.9%   | 86.7%   | 96.2%            | 89.5%      | 100.0%                 | 44.4%              | 88.0%                  | 81.3%                        |
| rs_lm           | 61.6%  | 81.3%  | 84.6%   | 85.6%   | 96.2%            | 89.5%      | 100.0%                 | 31.7%              | 89.3%                  | 81.3%                        |
| rs_varpro_gn_lm | 70.8%  | 85.2%  | 86.0%   | 86.7%   | 96.2%            | 89.5%      | 100.0%                 | 50.8%              | 94.7%                  | 80.0%                        |

change vs rs_auto (percentage points)

| solver          |   1 ms |   5 ms |   10 ms |   25 ms |   5 ms nist_strd |   5 ms mgh |   5 ms fitbenchmarking |   5 ms sagefit_spb |   5 ms synthetic_scale |   5 ms stringer2019_gratings |
|:----------------|-------:|-------:|--------:|--------:|-----------------:|-----------:|-----------------------:|-------------------:|-----------------------:|-----------------------------:|
| rs_gn_lm        |   -8.1 |   -2.7 |    -1.1 |    -1.1 |             -3.8 |          0 |                      0 |               -7.9 |                   -7.3 |                          2.7 |
| rs_lm           |  -10.9 |   -4.6 |    -2.4 |    -2.1 |             -3.8 |          0 |                      0 |              -20.6 |                   -6   |                          2.7 |
| rs_varpro_gn_lm |   -1.6 |   -0.8 |    -1   |    -1   |             -3.8 |          0 |                      0 |               -1.6 |                   -0.7 |                          1.3 |

speed and failures

| solver          |   median fit ms |   median ms to target |   failures |   overruns (>10%) |   fits |
|:----------------|----------------:|----------------------:|-----------:|------------------:|-------:|
| rs_auto         |           2.556 |                 0.232 |          0 |                 0 |    552 |
| rs_gn_lm        |           2.56  |                 0.518 |          2 |                 0 |    552 |
| rs_lm           |           3.59  |                 0.608 |          3 |                 0 |    552 |
| rs_varpro_gn_lm |           1.92  |                 0.224 |          0 |                 0 |    552 |

## test

success (mean over suites; per-suite columns at the main checkpoint)

| solver          | 1 ms   | 5 ms   | 10 ms   | 25 ms   | 5 ms nist_strd   | 5 ms mgh   | 5 ms fitbenchmarking   | 5 ms sagefit_spb   | 5 ms synthetic_scale   | 5 ms stringer2019_gratings   |
|:----------------|:-------|:-------|:--------|:--------|:-----------------|:-----------|:-----------------------|:-------------------|:-----------------------|:-----------------------------|
| rs_auto         | 73.8%  | 84.1%  | 86.0%   | 86.5%   | 100.0%           | 93.0%      | 92.3%                  | 37.1%              | 100.0%                 | 82.0%                        |
| rs_gn_lm        | 64.8%  | 84.4%  | 86.3%   | 86.8%   | 92.9%            | 90.7%      | 92.3%                  | 46.0%              | 100.0%                 | 84.7%                        |
| rs_lm           | 62.1%  | 82.4%  | 83.9%   | 85.0%   | 92.9%            | 90.7%      | 92.3%                  | 32.3%              | 100.0%                 | 86.0%                        |
| rs_varpro_gn_lm | 73.3%  | 83.8%  | 85.1%   | 85.5%   | 100.0%           | 93.0%      | 92.3%                  | 36.3%              | 100.0%                 | 81.3%                        |

change vs rs_auto (percentage points)

| solver          |   1 ms |   5 ms |   10 ms |   25 ms |   5 ms nist_strd |   5 ms mgh |   5 ms fitbenchmarking |   5 ms sagefit_spb |   5 ms synthetic_scale |   5 ms stringer2019_gratings |
|:----------------|-------:|-------:|--------:|--------:|-----------------:|-----------:|-----------------------:|-------------------:|-----------------------:|-----------------------------:|
| rs_gn_lm        |   -9   |    0.3 |     0.4 |     0.4 |             -7.1 |       -2.3 |                      0 |                8.9 |                      0 |                          2.7 |
| rs_lm           |  -11.7 |   -1.7 |    -2.1 |    -1.4 |             -7.1 |       -2.3 |                      0 |               -4.8 |                      0 |                          4   |
| rs_varpro_gn_lm |   -0.5 |   -0.2 |    -0.8 |    -1   |              0   |        0   |                      0 |               -0.8 |                      0 |                         -0.7 |

speed and failures

| solver          |   median fit ms |   median ms to target |   failures |   overruns (>10%) |   fits |
|:----------------|----------------:|----------------------:|-----------:|------------------:|-------:|
| rs_auto         |           5.029 |                 0.18  |          1 |                 0 |    521 |
| rs_gn_lm        |           8.866 |                 0.69  |          2 |                 0 |    521 |
| rs_lm           |          10.755 |                 0.739 |          3 |                 0 |    521 |
| rs_varpro_gn_lm |           4.119 |                 0.17  |          2 |                 0 |    521 |

## rank

sagefit_spb: 18 curves

|                 |   tau @ 10 ms |   top1 @ 10 ms |   lost @ 10 ms |   tau @ 100 ms |   top1 @ 100 ms |   lost @ 100 ms |
|:----------------|--------------:|---------------:|---------------:|---------------:|----------------:|----------------:|
| rs_auto         |         0.735 |          0.667 |          0.408 |          0.768 |           0.722 |           0.384 |
| rs_gn_lm        |         0.447 |          0.556 |          0.368 |          0.550 |           0.667 |           0.320 |
| rs_lm           |         0.544 |          0.389 |          0.584 |          0.677 |           0.611 |           0.472 |
| rs_varpro_gn_lm |         0.676 |          0.556 |          0.456 |          0.755 |           0.667 |           0.408 |

stringer2019_gratings: 100 curves

|                 |   tau @ 10 ms |   top1 @ 10 ms |   lost @ 10 ms |   tau @ 100 ms |   top1 @ 100 ms |   lost @ 100 ms |
|:----------------|--------------:|---------------:|---------------:|---------------:|----------------:|----------------:|
| rs_auto         |         0.652 |          0.680 |          0.000 |          0.674 |           0.700 |           0.000 |
| rs_gn_lm        |         0.672 |          0.610 |          0.000 |          0.696 |           0.640 |           0.000 |
| rs_lm           |         0.684 |          0.610 |          0.000 |          0.709 |           0.640 |           0.000 |
| rs_varpro_gn_lm |         0.636 |          0.660 |          0.000 |          0.664 |           0.680 |           0.000 |

