# Public Excel Model-Based RL + Online Optimization Imputer

This Streamlit application allows any user with access to the public app to:

1. Upload an Excel file.
2. Automatically detect the country column and year columns.
3. Train a Random Forest temporal world model.
4. Impute missing annual values using model-based prediction plus iterative online optimization.
5. View before/after results and validation metrics.
6. Download the completed Excel workbook.

## Expected input

Wide-format Excel data:

`geoUnit | 2015 | 2016 | ... | 2025`

The country column can also be named `country`, `Country`, `country_name`, `Country Name`, or `Entity`.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Make it public

Create a public GitHub repository containing:

- `app.py`
- `requirements.txt`
- `README.md`

Then deploy it using Streamlit Community Cloud. The application has no login/authentication layer of its own, so users can upload their own Excel file and download their result.

Do not put confidential or personally identifiable data into a public application unless appropriate safeguards are added.

## Output

The downloaded workbook contains:

- `Original_Data`
- `Imputed_Data`
- `Summary`
