# Model Card: Nazarian Advising Consortium — Federated Course Failure Model
## Intended use and prohibited uses
This model estimates the risk of course failure or late withdrawal for a student-course enrollment. The target, 'failed_course', is 1 if the student earned below a C or withdrew after the drop deadline.
Academic advisors can use the model to prioritize advising outreach and tutoring referrals during the first month. Under the team's agent policy, individual risk ranking is limited to Accounting, the designated home department.
Prohibited uses include admissions, financial aid and grading decisions, sharing individual records across departments, and labeling students to instructors.

## Members and data
The instructor generated the synthetic Campus dataset, which is stored in 'data/consortia/campus/'. Each row represents one student-course enrollment. Training uses 12 features, standardized with the reference statistics specified for the consortium. The neutral evaluation set contains 1,600 records, with 400 records from each department.
| Member | Rows | Base rate |Department finding |
| --- | ---: | ---: | --- |
| Accounting | 2,200 | 29% | Largest training dataset. |
| Business Analytics | 900 | 30% | Fewer than half as many training records as Accounting, with a similar failure rate. |
| Marketing | 1,300 | 31% | Highest observed failure rate. |
| Real Estate | 140 | 22% | Smallest training dataset and lowest observed failure rate. |

Provenance: The instructor generated the synthetic Campus dataset in `data/consortia/campus/`. The consortium schema provides the reference statistics used for feature standardization.

## Training
Baseline run '9404116a' used charter v1.0 and trained a federated model to predict 'failed_course' across four departments. Training ran for 15 rounds with at most 5 local epochs, mean aggregation, clipping at 2.0, secure aggregation enabled, and a DP noise multiplier of 0.3.

## Performance
| Member | Local only | FedAvg | Centralized | Gain |
| --- | ---: | ---: | ---: | ---: |
|Accounting|0.748|0.738|0.750|-0.010|
|Business Analytics|0.748|0.760|0.752|+0.012|
|Marketing|0.743|0.748|0.752|+0.005|
|Real Estate|0.745|0.750|0.782|+0.005|
|overall|-|0.749|0.759|-|

FedAvg achieved 0.7488 overall accuracy and an AUC of 0.755. The federation gate passed under charter v1.0. Business Analytics had the largest gain over its local model (+1.2 percentage points), while Accounting was the only department whose accuracy declined (-1.0 percentage points).

### Member Accuracy Floor Test
In run `f6780136`, we raised the minimum member accuracy from 0.62 to 0.74. Overall accuracy remained at 0.7488 and met the overall threshold of 0.68. Accounting scored 0.7375, below the new member threshold of 0.74, so the federation gate returned `BLOCKED`. This test confirmed that the gate enforces the member accuracy floor even when the model meets the overall accuracy requirement.

### Privacy Noise Comparison

We tested three DP noise multipliers with secure aggregation enabled and the minimum member accuracy fixed at 0.62.

| DP noise | Run ID | Overall | Accounting | Business Analytics | Marketing | Real Estate | Gate |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 0.0 | `5f0cc606` | 76.06% | 75.00% | 75.75% | 75.75% | 77.75% | PASS |
| 0.3 | `9404116a` | 74.88% | 73.75% | 76.00% | 74.75% | 75.00% | PASS |
| 1.0 | `009d2831` | 68.06% | 65.75% | 68.25% | 70.50% | 67.75% | PASS |

Increasing the noise multiplier from 0.0 to 0.3 reduced overall accuracy by 1.19 percentage points. Real Estate had the largest decline over that interval, at 2.75 percentage points. At 1.0, overall accuracy narrowly cleared the 68% threshold, and all four departments performed worse than their local-only models.

### Real Estate Driver Analysis
The 'driver_signs_all_members' tool found weak positive correlations between weekly work hours and course failure in Accounting (+0.080), Business Analytics (+0.096), and Marketing (+0.085). Real Estate was the only department with a negative correlation (-0.117). These findings suggest that the relationship between work hours and course failure varies by department. For Real Estate, FedAvg achieved 75.0% accuracy on the neutral evaluation set, compared with 74.5% for the local-only model. Further feature-level analysis would be needed to determine whether the shared model captures the distinct relationship between weekly work hours and the outcome in Real Estate.

## Limitations and fairness
These results come from an instructor-generated synthetic dataset and reflect performance on the neutral evaluation set. The reported overall accuracy weights each department equally because the evaluation set contains 400 records per department. The comparisons are at the department level. Performance across student subgroups and on real enrollment data remains unverified.