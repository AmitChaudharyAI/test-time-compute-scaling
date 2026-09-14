# Controller error analysis

Categories are computed exhaustively from all 500 outer-test predictions, not
manually selected. They are non-exclusive: for example, a regression avoided
is also a correct early stop, and invalid-answer uncertainty can accompany any
outcome. Representative IDs are the ten numerically smallest IDs in each
category, a deterministic rule rather than cherry-picking.

| Category | Logistic count | Logistic representative IDs | Tree count | Tree representative IDs |
|---|---:|---|---:|---|
| Premature-stop missed recovery | 14 | 180, 189, 234, 266, 279, 298, 302, 357, 380, 385 | 17 | 64, 180, 189, 234, 266, 279, 298, 302, 332, 357 |
| Unnecessary continuation in hindsight | 123 | 4, 8, 11, 14, 16, 24, 25, 26, 28, 42 | 130 | 4, 8, 11, 14, 16, 24, 26, 28, 41, 42 |
| Correct early stop | 384 | 0, 1, 2, 3, 4, 5, 6, 7, 8, 10 | 381 | 0, 1, 2, 3, 4, 5, 6, 7, 8, 10 |
| Regression avoided | 8 | 17, 18, 123, 190, 309, 423, 460, 494 | 7 | 17, 18, 41, 190, 423, 460, 494 |
| Stable wrong despite maximum compute | 18 | 9, 11, 33, 62, 103, 109, 152, 236, 239, 248 | 14 | 9, 11, 33, 62, 103, 109, 236, 239, 248, 285 |
| Invalid-answer driven uncertainty | 81 | 9, 11, 21, 25, 46, 60, 64, 88, 94, 96 | 76 | 9, 11, 21, 25, 46, 60, 64, 88, 94, 96 |

Definitions are operational. A missed recovery is an early wrong stop whose
N=16 aggregate is correct. Unnecessary continuation means the t=2 vote and
correctness already match the returned state but the policy spends beyond t=2;
it is knowable only retrospectively. Stable wrong means the policy reaches
N=16 and remains wrong. Invalid-answer uncertainty means at least one observed
prefix through the stopping point has a nonzero invalid rate.

The dominant scientific failure is not inability to discriminate recovery rows
in aggregate; it is converting rare-event probabilities into stable sequential
decisions across questions. The primary logistic model misses 14 N=16
recoveries while avoiding eight N=16 regressions and making 94 early wrong
stops overall. The full case-level table is `results/error_cases.csv`.
