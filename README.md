# ML Strategy for MetaTrader 5: Python -> ONNX -> MQL5 Expert Advisor

This package contains a complete example for the most practical flow when you want:

1. to train the model in Python,
2. to export it to ONNX,
3. to run it in the MT5 Expert Advisor,
4. to test it directly in **MetaTrader 5 Strategy Tester**.

## What it contains

- `train_random_forrest_regressor.py` - Python script for training and ONNX export
- `MT5_RF_Regressor_ONNX_Strategy.mq5` - Expert Advisor for MT5
- `README.md` - work steps

## Strategy idea

The model learns to estimate the **return of the next closed bar** (`next closed-bar return`) based on 10 simple features:

- `ret_1`, `ret_3`, `ret_5`, `ret_10`
- `vol_10`, `vol_20`
- `dist_sma_10`, `dist_sma_20`
- `zscore_20`
- `atr_14`

The trading signal is then:

- `BUY` if prediction > `+EntryThreshold`
- `SELL` if prediction < `-EntryThreshold`
- `FLAT` otherwise

The EA opens at most one position per symbol and optionally uses SL/TP based on ATR.

---

## Why the flow is made this way

**MetaTrader 5 Strategy Tester tests MQL5 programs, not Python scripts.**

Therefore, to be able to do backtesting right in MT5, the correct flow is:

- training in Python,
- export model to ONNX,
- run model in MQL5,
- backtest in Strategy Tester.

---

## 1. Install Python dependencies

Example with venv:

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate   # Windows

pip install --upgrade pip
pip install MetaTrader5 pandas numpy scikit-learn skl2onnx onnx
```

If you don't want to read data directly from the MT5 terminal, you can also use a CSV exported separately from MT5.

---

## 2. Train model from Python

### Variant A - read directly from MT5 terminal

The MT5 terminal must be started and connected.

```bash
python train_random_forrest_regressor.py --symbol XAGUSD --timeframe M15 --bars 15000 --output-dir output
```

### Variant B - read from CSV

The CSV must have at least the columns:

- `time`
- `open`
- `high`
- `low`
- `close`

Optional:

- `volume`

Example:

```bash
python train_random_forrest_regressor.py --symbol XAGUSD --timeframe M15 --bars 15000 --output-dir output
```

### What you get after running

In the `output/` directory:

- `ml_strategy_model.onnx`
- `model_metadata.json`
- `training_rates_snapshot.csv`
- `training_features_snapshot.csv`

`model_metadata.json` also contains an initial suggested threshold for entry.

---

## 3. Preparation for MT5

1. Copy `MT5_RF_Regressor_ONNX_Strategy.mq5` to:
   - `MQL5/Experts/` or a subfolder of it.
2. Copy `output/ml_strategy_model.onnx` to the **same folder** as the `.mq5` file.
3. Open `MetaEditor`.
4. Compile the EA.

**Important:** The ONNX is included as a resource at compilation. If you change the model, you must recompile the EA.

---

## 4. How to do backtesting in MT5

In Strategy Tester:

1. Select `MT5_RF_Regressor_ONNX_Strategy`
2. Choose the same symbol and same timeframe used in training
3. Choose the test period
4. Since the EA works only on the appearance of a new bar, you can use a modeling compatible with bar backtest; however, also test in the most realistic mode available for your broker
5. Start the test

### Important EA parameters

- `InpEntryThreshold`
  - the minimum absolute threshold of the prediction to enter the market
  - start with the value suggested in `model_metadata.json`
- `InpUseAtrStops`
  - activate SL/TP based on ATR
- `InpStopAtrMultiple`, `InpTakeAtrMultiple`
  - ATR multiples for SL/TP
- `InpMaxBarsInTrade`
  - forced closure after a certain number of bars
- `InpCloseOnOppositeSignal`
  - close the position when the opposite signal comes

---

## 5. What you need to check so as not to fool yourself in backtest

### A. The same symbol and same timeframe
The model must be tested on the same type of data it was trained on.

### B. Only closed bars
The EA calculates features on the **closed bar**, not on the current bar, to avoid part of the lookahead bias.

### C. Transaction costs
Check in tester:

- spread
- commissions
- swap
- slippage (how much the broker / tester can simulate)

### D. Do not over-optimize
Do not turn the tester into an overfitting generator.

Good to do:

- training on one period,
- test on another period,
- possibly optimize on a small piece,
- then out-of-sample validation on a different piece.

### E. Real walk-forward
The serious flow is:

- train on the past,
- validate on unused future,
- possibly periodic retraining.

The current package gives you the technical basis, not the guarantee of a profitable strategy.

---

## 6. How to learn from it

### Step 1
Run the example exactly without changing anything.

### Step 2
Look at:

- `training_features_snapshot.csv`
- the MT5 journal
- the backtest report

### Step 3
Change only one thing at a time:

- entry threshold
- ATR multiples
- maximum number of bars in trade
- symbol
- timeframe

### Step 4
Only after that change the model.

Exemple de extensii:

- RandomForestClassifier sau LogisticRegression
- mai multe features tehnice
- filtre pe sesiune
- filtru de trend pe timeframe mai mare
- model separat pentru BUY si SELL

---

## 7. Important observations

- The Python script currently uses `RandomForestRegressor` to easily export an ONNX model with a single numeric output.
- If you want, you can later change it to another model supported by scikit-learn / ONNX.
- For productivity, start simple and complicate only after you can reproduce the same results from end to end.

---

## 8. Recommended workflow

1. train the model on old history
2. export `ml_strategy_model.onnx`
3. compile the EA with the included model
4. run backtest
5. adjust only after you have seen:
   - number of transactions
   - profit factor
   - drawdown
   - expectancy
   - distribution of transactions
6. redo the training only when you have a clear hypothesis

---

## 9. What I would do next after it runs

After you confirm that the flow works end-to-end:

- add **walk-forward retraining** in Python
- compare `RandomForestRegressor` with `LogisticRegression`, `Ridge`, `XGBoost` or `LightGBM`
- save the threshold and other parameters in a JSON/CSV file and read them from MQL5
- make a multi-symbol version

## 10. Learning on 70% and testing on 30%

Preparation of full output data in `output_full`:
```bash
python train_random_forrest_regressor.py --symbol XAGUSD --timeframe M15 --bars 15000 --output-dir output_full
```

Learning after 70% of data:
```bash
python -c "import pandas as pd; df=pd.read_csv('output_full/training_rates_snapshot.csv', parse_dates=['time']); split=int(len(df)*0.7); train=df.iloc[:split].copy(); test=df.iloc[split:].copy(); train.to_csv('output_full/train_rates.csv', index=False); test.to_csv('output_full/test_rates.csv', index=False); print('ROWS_TOTAL=', len(df)); print('ROWS_TRAIN=', len(train)); print('ROWS_TEST=', len(test)); print('TRAIN_START=', train['time'].iloc[0]); print('TRAIN_END=', train['time'].iloc[-1]); print('TEST_START=', test['time'].iloc[0]); print('TEST_END=', test['time'].iloc[-1])"

python train_random_forrest_regressor.py --csv output_full\train_rates.csv --symbol XAGUSD --timeframe M15 --output-dir output_train70
```

Compile with onnx from output_train70. Test on 30% of data based on TEST_START and TEST_END from terminal.

## 11. Learning on 50% and testing on 50%

Learning after 50% of data:
```bash
python -c "import pandas as pd; df=pd.read_csv('output_full/training_rates_snapshot.csv', parse_dates=['time']); split=int(len(df)*0.5); train=df.iloc[:split].copy(); test=df.iloc[split:].copy(); train.to_csv('output_full/train_rates_50.csv', index=False); test.to_csv('output_full/test_rates_50.csv', index=False); print('ROWS_TOTAL=', len(df)); print('ROWS_TRAIN=', len(train)); print('ROWS_TEST=', len(test)); print('TRAIN_START=', train['time'].iloc[0]); print('TRAIN_END=', train['time'].iloc[-1]); print('TEST_START=', test['time'].iloc[0]); print('TEST_END=', test['time'].iloc[-1])"

python train_random_forrest_regressor.py --csv output_full\train_rates_50.csv --symbol XAGUSD --timeframe M15 --output-dir output_train50
```

Compile with onnx from output_train50. Test on 50% of data based on TEST_START and TEST_END from terminal.

## 12. Verification shuffled data

Prepare shuffled data:
```bash
python -c "import pandas as pd, numpy as np; df=pd.read_csv('output_full/train_rates.csv', parse_dates=['time']); shuffled=df[['open','high','low','close','volume']].sample(frac=1, random_state=42).reset_index(drop=True); out=pd.DataFrame({'time': df['time'].reset_index(drop=True), 'open': shuffled['open'], 'high': shuffled['high'], 'low': shuffled['low'], 'close': shuffled['close'], 'volume': shuffled['volume']}); out.to_csv('output_full/train_rates_shuffled.csv', index=False); print('Saved output_full/train_rates_shuffled.csv with', len(out), 'rows')"

python train_random_forrest_regressor.py --csv output_full\train_rates_shuffled.csv --symbol XAGUSD --timeframe M15 --output-dir output_train_shuffled
```

Compile with onnx from output_train_shuffled. Test on 100% of data.
