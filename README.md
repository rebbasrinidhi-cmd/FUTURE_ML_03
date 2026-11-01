# FUTURE_ML_03: Resume / Candidate Screening System

Future Interns · Machine Learning Track · Task 3

An ML-assisted screening tool that parses resumes, reads a job description, ranks candidates by role fit
with an explainable score, and reports skill gaps for each candidate and for the whole applicant pool.

## The business problem
A role attracts hundreds of resumes. Reading them all is slow and inconsistent, and strong candidates get
buried. This project sorts the pile so recruiters spend their time on the best matches, and shows *why* each
candidate scored what they did and what skills they are missing.

## What is in this repo
| File | Purpose |
|---|---|
| `Resume_Screening_System.ipynb` | Full workflow: parsing, skill extraction, job description parsing, ranking, evaluation, skill gaps |
| `images/` | Charts used in the notebook |
| `outputs/ranked_candidates_data_scientist.csv` | Every candidate ranked for the target role, with matched and missing skills |
| `outputs/skill_gap_report_data_scientist.csv` | Top-10 shortlist with a one-line gap summary each |
| `outputs/ranking_method_comparison.csv` | Evaluation of four ranking methods |
| `outputs/parsed_resumes.csv` | Structured fields extracted from each resume |
| `requirements.txt` | Python dependencies |

## Approach
1. **Parsing and cleaning**: contact details removed from scoring text, name kept for display only (blind scoring), years of experience, education level and skills extracted with rules.
2. **Skill extraction**: a 65-skill taxonomy with aliases (`sklearn`, `k8s`, `ReactJS`), matched case-insensitively with word boundaries.
3. **Job description parsing**: required vs nice-to-have skills, minimum years and education are read from the text. Any new job description can be pasted in.
4. **Ranking model (hybrid, 0-100)**: TF-IDF text similarity (30%), weighted skill coverage (35%), role-fit probability from a logistic-regression classifier (20%), experience (7.5%) and education (7.5%). The classifier's probabilities are out-of-fold, so no resume is scored by a model trained on it.
5. **Evaluation**: keyword matching, text similarity, role classifier and the hybrid are compared with Precision@10, NDCG@10 and MAP, plus how well each orders candidates within a role.
6. **Skill gaps**: per-candidate heatmap, a gap sentence for each shortlisted candidate, and a pool-level chart of the hardest required skills to find.

## Results (simulated data)
- Role classifier (5-fold, out-of-fold): **98.9% accuracy**.
- Ranking metrics are near the ceiling for most methods because the simulated roles use fairly distinct vocabularies. Keyword matching alone is weakest at telling neighbouring roles apart (MAP 0.88 vs 0.99+ for the other methods), since skills like Python and SQL appear in several roles.
- Inside a role, skill coverage is the strongest single signal for ordering candidates. The hybrid keeps role awareness and still orders candidates well.
- Example for the Data Scientist role: 34 Shortlist, 34 Review and 292 Not a fit out of 360 resumes.

## Important notes
- **Simulated data.** The notebook runs on 360 simulated resumes of fictional candidates, with no real personal data. Results show the pipeline works on controlled data. Real resumes are messier, so accuracy will be lower and the weights should be re-tuned.
- **A decision aid, not a decision maker.** Keep a human in the loop and never auto-reject on score alone. Audit shortlists for bias before using any screening tool on real candidates.
- **Literal skill matching.** Skills outside the taxonomy are invisible. Extend `SKILLS` for new roles.

## Using a real resume dataset
Set `DATA_PATH` in the configuration cell to a CSV with a `Category` and a `Resume` column (for example the Kaggle
*Resume Dataset*). The parser, ranker and gap analysis run unchanged. Rename the dataset's categories to match the
role names in `JOB_DESCRIPTIONS`, or add job descriptions and skills for the dataset's own roles.

## Run it
```bash
pip install -r requirements.txt
jupyter notebook Resume_Screening_System.ipynb
```
