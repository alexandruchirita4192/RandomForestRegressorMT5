# Strategie ML pentru MetaTrader 5: Python -> ONNX -> Expert Advisor MQL5

Acest pachet contine un exemplu complet pentru fluxul cel mai practic cand vrei:

1. sa antrenezi modelul in Python,
2. sa il exporti in ONNX,
3. sa il rulezi in Expert Advisor-ul MT5,
4. sa il testezi direct in **MetaTrader 5 Strategy Tester**.

## Ce contine

- `train_mt5_ml_strategy.py` - scriptul Python pentru training si export ONNX
- `MT5_ML_ONNX_Strategy.mq5` - Expert Advisor-ul pentru MT5
- `README.md` - pasii de lucru

## Ideea strategiei

Modelul invata sa estimeze **randamentul urmatorului bar inchis** (`next closed-bar return`) pe baza a 10 features simple:

- `ret_1`, `ret_3`, `ret_5`, `ret_10`
- `vol_10`, `vol_20`
- `dist_sma_10`, `dist_sma_20`
- `zscore_20`
- `atr_14`

Semnalul de tranzactionare este apoi:

- `BUY` daca predictia > `+EntryThreshold`
- `SELL` daca predictia < `-EntryThreshold`
- `FLAT` altfel

EA-ul deschide cel mult o pozitie pe simbol si foloseste optional SL/TP pe baza ATR.

---

## De ce fluxul este facut asa

**MetaTrader 5 Strategy Tester testeaza programe MQL5, nu scripturi Python.**

De aceea, pentru a putea face backtesting chiar in MT5, fluxul corect este:

- training in Python,
- export model in ONNX,
- rulare model in MQL5,
- backtest in Strategy Tester.

---

## 1. Instalare dependinte Python

Exemplu cu venv:

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate   # Windows

pip install --upgrade pip
pip install MetaTrader5 pandas numpy scikit-learn skl2onnx onnx
```

Daca nu vrei sa citesti datele direct din terminalul MT5, poti folosi si un CSV exportat separat din MT5.

---

## 2. Antrenare model din Python

### Varianta A - citire directa din terminalul MT5

Terminalul MT5 trebuie sa fie pornit si conectat.

```bash
python train_mt5_ml_strategy.py --symbol XAGUSD --timeframe M15 --bars 15000 --output-dir output
```

### Varianta B - citire din CSV

CSV-ul trebuie sa aiba macar coloanele:

- `time`
- `open`
- `high`
- `low`
- `close`

Optional:

- `volume`

Exemplu:

```bash
python train_mt5_ml_strategy.py --symbol XAGUSD --timeframe M15 --bars 15000 --output-dir output
```

### Ce obtii dupa rulare

In directorul `output/`:

- `ml_strategy_model.onnx`
- `model_metadata.json`
- `training_rates_snapshot.csv`
- `training_features_snapshot.csv`

`model_metadata.json` contine si un prag initial sugerat pentru intrare.

---

## 3. Pregatire pentru MT5

1. Copiaza `MT5_ML_ONNX_Strategy.mq5` in:
   - `MQL5/Experts/` sau intr-un subfolder al lui.
2. Copiaza `output/ml_strategy_model.onnx` in **acelasi folder** cu fisierul `.mq5`.
3. Deschide `MetaEditor`.
4. Compileaza EA-ul.

**Important:** ONNX-ul este inclus ca resource la compilare. Daca schimbi modelul, trebuie sa recompilezi EA-ul.

---

## 4. Cum faci backtesting in MT5

In Strategy Tester:

1. Selecteaza `MT5_ML_ONNX_Strategy`
2. Alege acelasi simbol si acelasi timeframe folosite la training
3. Alege perioada de test
4. Pentru ca EA-ul lucreaza doar la aparitia unei bare noi, poti folosi o modelare compatibila cu backtest-ul pe bare; totusi testeaza si in modul cel mai realist disponibil pentru brokerul tau
5. Porneste testul

### Parametri importanti ai EA-ului

- `InpEntryThreshold`
  - pragul minim absolut al predictiei pentru a intra in piata
  - incepe cu valoarea sugerata in `model_metadata.json`
- `InpUseAtrStops`
  - activeaza SL/TP pe baza ATR
- `InpStopAtrMultiple`, `InpTakeAtrMultiple`
  - multipli ATR pentru SL/TP
- `InpMaxBarsInTrade`
  - inchidere fortata dupa un anumit numar de bare
- `InpCloseOnOppositeSignal`
  - inchide pozitia cand vine semnalul opus

---

## 5. Ce trebuie sa verifici ca sa nu te pacalesti in backtest

### A. Acelasi simbol si acelasi timeframe
Modelul trebuie testat pe acelasi tip de date pe care a fost antrenat.

### B. Doar bare inchise
EA-ul calculeaza features pe **bara inchisa**, nu pe bara in curs, ca sa evite o parte din lookahead bias.

### C. Costuri de tranzactionare
Verifica in tester:

- spread
- comisioane
- swap
- slippage (cat poate simula brokerul / testerul)

### D. Nu optimiza excesiv
Nu transforma testerul intr-un generator de overfitting.

Bun de facut:

- training pe o perioada,
- test pe alta perioada,
- eventual optimizare pe o bucata mica,
- apoi validare out-of-sample pe o bucata diferita.

### E. Walk-forward real
Fluxul serios este:

- train pe trecut,
- validare pe viitor nefolosit,
- eventual retraining periodic.

Pachetul de fata iti da baza tehnica, nu garantia unei strategii profitabile.

---

## 6. Cum inveti din el

### Pasul 1
Ruleaza exact exemplul fara sa schimbi nimic.

### Pasul 2
Uita-te la:

- `training_features_snapshot.csv`
- jurnalul din MT5
- raportul de backtest

### Pasul 3
Schimba o singura chestie odata:

- pragul de intrare
- multiplii ATR
- numarul maxim de bare in trade
- simbolul
- timeframe-ul

### Pasul 4
Abia dupa aceea schimba modelul.

Exemple de extensii:

- RandomForestClassifier sau LogisticRegression
- mai multe features tehnice
- filtre pe sesiune
- filtru de trend pe timeframe mai mare
- model separat pentru BUY si SELL

---

## 7. Observatii importante

- Scriptul Python foloseste momentan `RandomForestRegressor` pentru a exporta usor un model ONNX cu o singura iesire numerica.
- Daca vrei, il poti schimba ulterior in alt model suportat de scikit-learn / ONNX.
- Pentru productivitate, incepe simplu si complica abia dupa ce poti reproduce aceleasi rezultate de la cap la coada.

---

## 8. Flux recomandat de lucru

1. antrenezi modelul pe istoricul vechi
2. exporti `ml_strategy_model.onnx`
3. compilezi EA-ul cu modelul inclus
4. rulezi backtest
5. ajustezi numai dupa ce ai vazut:
   - numar tranzactii
   - profit factor
   - drawdown
   - expectancy
   - distributia tranzactiilor
6. refaci training-ul doar cand ai o ipoteza clara

---

## 9. Ce as face mai departe dupa ce ruleaza

Dupa ce confirmi ca fluxul functioneaza cap-coada:

- adauga **walk-forward retraining** in Python
- compara `RandomForestRegressor` cu `LogisticRegression`, `Ridge`, `XGBoost` sau `LightGBM`
- salveaza pragul si alti parametri intr-un fisier JSON/CSV si citeste-i din MQL5
- fa o versiune multi-simbol

## 10. Invatare pe 70% si testare pe 30%

Pregatire date de output full in `output_full`:
```bash
python train_mt5_ml_strategy.py --symbol XAGUSD --timeframe M15 --bars 15000 --output-dir output_full
```

Invatare dupa 70% din date:
```bash
python -c "import pandas as pd; df=pd.read_csv('output_full/training_rates_snapshot.csv', parse_dates=['time']); split=int(len(df)*0.7); train=df.iloc[:split].copy(); test=df.iloc[split:].copy(); train.to_csv('output_full/train_rates.csv', index=False); test.to_csv('output_full/test_rates.csv', index=False); print('ROWS_TOTAL=', len(df)); print('ROWS_TRAIN=', len(train)); print('ROWS_TEST=', len(test)); print('TRAIN_START=', train['time'].iloc[0]); print('TRAIN_END=', train['time'].iloc[-1]); print('TEST_START=', test['time'].iloc[0]); print('TEST_END=', test['time'].iloc[-1])"

python train_mt5_ml_strategy.py --csv output_full\train_rates.csv --symbol XAGUSD --timeframe M15 --output-dir output_train70
```

Compilare cu onnx din output_train70. Testare pe 30% din date pe baza TEST_START si TEST_END din terminal.

## 11. Invatare pe 50% si testare pe 50%

Invatare dupa 50% din date:
```bash
python -c "import pandas as pd; df=pd.read_csv('output_full/training_rates_snapshot.csv', parse_dates=['time']); split=int(len(df)*0.5); train=df.iloc[:split].copy(); test=df.iloc[split:].copy(); train.to_csv('output_full/train_rates_50.csv', index=False); test.to_csv('output_full/test_rates_50.csv', index=False); print('ROWS_TOTAL=', len(df)); print('ROWS_TRAIN=', len(train)); print('ROWS_TEST=', len(test)); print('TRAIN_START=', train['time'].iloc[0]); print('TRAIN_END=', train['time'].iloc[-1]); print('TEST_START=', test['time'].iloc[0]); print('TEST_END=', test['time'].iloc[-1])"

python train_mt5_ml_strategy.py --csv output_full\train_rates_50.csv --symbol XAGUSD --timeframe M15 --output-dir output_train50
```

Compilare cu onnx din output_train50. Testare pe 50% din date pe baza TEST_START si TEST_END din terminal.

## 12. Verificare shuffled data

Pregatire shuffled data:
```bash
python -c "import pandas as pd, numpy as np; df=pd.read_csv('output_full/train_rates.csv', parse_dates=['time']); shuffled=df[['open','high','low','close','volume']].sample(frac=1, random_state=42).reset_index(drop=True); out=pd.DataFrame({'time': df['time'].reset_index(drop=True), 'open': shuffled['open'], 'high': shuffled['high'], 'low': shuffled['low'], 'close': shuffled['close'], 'volume': shuffled['volume']}); out.to_csv('output_full/train_rates_shuffled.csv', index=False); print('Saved output_full/train_rates_shuffled.csv with', len(out), 'rows')"

python train_mt5_ml_strategy.py --csv output_full\train_rates_shuffled.csv --symbol XAGUSD --timeframe M15 --output-dir output_train_shuffled
```

Compilare cu onnx din output_train_shuffled. Testare pe 100% din date.
