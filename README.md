# AirVision — fixed VS Code starter

## 1. Set up the API key
1. Open the project folder in VS Code.
2. Copy `.env.example` to a new file named `.env`.
3. In `.env`, replace the placeholder with your real OpenAQ key:
   `OPENAQ_API_KEY=your_actual_key_here`
4. Never post or share the key.

Get/manage the key at https://explore.openaq.org/account.

## 2. Install packages
Activate your existing `.venv`, then run:
```powershell
python -m pip install -r requirements.txt
```

## 3. Run the pipeline
```powershell
python run_pipeline.py
```

## 4. Open the dashboard
```powershell
python -m streamlit run app.py
```

## Important notes
- OpenAQ's point/radius query supports a maximum radius of 25,000 m; this project uses 25 km around Vijayawada.
- The pipeline uses real OpenAQ readings and reports API errors rather than making up data.
- The current baseline model uses only latitude/longitude. It is not a satellite/weather ML model. RMSE is reported only if enough distinct stations are found.
- Sparse local station coverage may mean there is not enough data to train a model. That is a data limitation, not a code failure.
