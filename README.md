
# When to Curtail
### Hydropower and Firm Supply Security

University of Iceland – Operations Research II

## Project overview

This project develops a stochastic dynamic programming (SDP) model for hydropower production at Fljótsdalsstöð, using the historical Alcoa Fjarðaál supply contract.

The objective is to maximize expected delivery of 54 MW curtailable electricity while maintaining at least 95% annual reliability for 483 MW firm supply.

The model uses historical inflow data, monthly Markov transition matrices, Bellman backward induction, and reservoir storage constraints.

## Repository files

| File | Description |
|---|---|
| `Untitled0.ipynb` | Main Colab notebook containing the group report, calculations, figures and appendices |
| `hluti_2.py` | Monthly inflow classification and transition matrices |
| `curtail_sdp.py` | SDP model, Bellman solution, reliability evaluation and visualizations |
| `einstaklingsverkefni_3.py` | Individual assignment 3: independent reliability verification |
| `einstaklingsverkefni_4.py` | Individual assignment 4: policy iteration |
| `Guðný-Einstaklingshluti_2.ipynb` | Individual assignment 2 |
| `hydro_group_project_data.zip` | Original hydropower dataset and documentation |

## Requirements

- Python 3
- Google Colab or Jupyter Notebook
- numpy
- pandas
- matplotlib
- scipy
- openpyxl

## Running the project in Google Colab

1. Download or clone this repository.
2. Extract `hydro_group_project_data.zip`.
3. Upload the project files and extracted data to:

   `/content/drive/MyDrive/When_to_Curtail/`

4. Open `Untitled0.ipynb` in Google Colab.
5. Mount Google Drive when prompted.
6. Run the notebook cells in order, from top to bottom.

The notebook contains the data preparation, stochastic optimization, evaluation, and figures.

Some sensitivity analyses and individual assignments re-solve the SDP model repeatedly and may take longer than the main calculations.

## Model assumptions

- 12 model months of 730 hours each.
- Reservoir capacity: 100–2,270 Mm³.
- Firm supply: 483 MW.
- Curtailable supply: up to 54 MW.
- Maximum power production: 575 MW.
- Training data: 2015–2023.
- Independent test year: 2024.
- Initial reservoir volumes: 1,600, 1,800 and 1,950 Mm³.

Annual reliability is measured as the probability of meeting firm supply in all 12 months.

## Reproducibility

The Python code and notebook are provided to allow the numerical results to be reproduced.

The individual assignments build on the group SDP model. Some individual analyses use a coarser discretization to reduce computational runtime; these differences are documented in the notebook.

## Data source

The original hydropower dataset and assignment description were provided through the University of Iceland course materials.

## Limitations

This is an educational model based on pooled reservoir storage and monthly decisions. It does not represent all operational restrictions of the actual Icelandic electricity system.

Monthly supply adequacy does not guarantee uninterrupted hourly electricity delivery.
