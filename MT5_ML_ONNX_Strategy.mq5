#property strict
#property version   "1.00"
#property description "EA de exemplu: model ML antrenat in Python, exportat ONNX, rulat in MT5 Strategy Tester"

#include <Trade/Trade.mqh>

// IMPORTANT:
// Inainte de compilare, copiaza fisierul ml_strategy_model.onnx in acelasi folder cu acest .mq5.
#resource "ml_strategy_model.onnx" as uchar ExtModel[]

input double InpLots                  = 0.10;      // Lot fix
input double InpEntryThreshold        = 0.00060;   // Semnal minim absolut pe randamentul prezis
input bool   InpUseAtrStops           = true;      // Foloseste SL/TP pe baza ATR
input double InpStopAtrMultiple       = 1.50;      // SL = ATR * multiplicator
input double InpTakeAtrMultiple       = 2.50;      // TP = ATR * multiplicator
input int    InpMaxBarsInTrade        = 12;        // Iesire fortata dupa N bare inchise
input bool   InpCloseOnOppositeSignal = true;      // Inchide pe semnal opus
input bool   InpAllowLong             = true;      // Permite BUY
input bool   InpAllowShort            = true;      // Permite SELL
input long   InpMagic                 = 26042026;  // Magic number

const int FEATURE_COUNT = 10;
const long EXT_INPUT_SHAPE[]  = {1, FEATURE_COUNT};
const long EXT_OUTPUT_SHAPE[] = {1, 1};

CTrade trade;
long   g_model_handle = INVALID_HANDLE;
datetime g_last_bar_time = 0;
int    g_bars_in_trade = 0;


enum SignalDirection
  {
   SIGNAL_SELL = -1,
   SIGNAL_FLAT =  0,
   SIGNAL_BUY  =  1
  };


bool IsNewBar()
  {
   datetime current_bar_time = iTime(_Symbol, _Period, 0);
   if(current_bar_time == 0)
      return false;

   if(g_last_bar_time == 0)
     {
      g_last_bar_time = current_bar_time;
      return false;
     }

   if(current_bar_time != g_last_bar_time)
     {
      g_last_bar_time = current_bar_time;
      return true;
     }
   return false;
  }


double Mean(const double &arr[], int start_shift, int count)
  {
   double sum = 0.0;
   for(int i = start_shift; i < start_shift + count; i++)
      sum += arr[i];
   return sum / count;
  }


double StdDev(const double &arr[], int start_shift, int count)
  {
   double m = Mean(arr, start_shift, count);
   double s = 0.0;
   for(int i = start_shift; i < start_shift + count; i++)
     {
      double d = arr[i] - m;
      s += d * d;
     }
   return MathSqrt(s / MathMax(count - 1, 1));
  }


double CalcATR(const MqlRates &rates[], int start_shift, int period)
  {
   double sum_tr = 0.0;
   for(int i = start_shift; i < start_shift + period; i++)
     {
      double high = rates[i].high;
      double low = rates[i].low;
      double prev_close = rates[i + 1].close;
      double tr1 = high - low;
      double tr2 = MathAbs(high - prev_close);
      double tr3 = MathAbs(low - prev_close);
      double tr = MathMax(tr1, MathMax(tr2, tr3));
      sum_tr += tr;
     }
   return sum_tr / period;
  }


bool BuildFeatureVector(matrixf &features, double &atr14)
  {
   MqlRates rates[];
   ArraySetAsSeries(rates, true);

   // Avem nevoie de bare suficiente pentru toate ferestrele si pentru ATR.
   if(CopyRates(_Symbol, _Period, 0, 80, rates) < 40)
     {
      Print("Nu sunt suficiente bare pentru features.");
      return false;
     }

   double closes[];
   ArrayResize(closes, ArraySize(rates));
   ArraySetAsSeries(closes, true);
   for(int i = 0; i < ArraySize(rates); i++)
      closes[i] = rates[i].close;

   // Folosim doar bare inchise. Shift 1 = ultima bara inchisa.
   int s = 1;

   double ret_1  = (closes[s] / closes[s + 1]) - 1.0;
   double ret_3  = (closes[s] / closes[s + 3]) - 1.0;
   double ret_5  = (closes[s] / closes[s + 5]) - 1.0;
   double ret_10 = (closes[s] / closes[s + 10]) - 1.0;

   double one_bar_returns[];
   ArrayResize(one_bar_returns, 30);
   for(int i = 0; i < 30; i++)
      one_bar_returns[i] = (closes[s + i] / closes[s + i + 1]) - 1.0;

   double vol_10 = StdDev(one_bar_returns, 0, 10);
   double vol_20 = StdDev(one_bar_returns, 0, 20);

   double sma_10 = Mean(closes, s, 10);
   double sma_20 = Mean(closes, s, 20);

   if(sma_10 == 0.0 || sma_20 == 0.0)
      return false;

   double dist_sma_10 = (closes[s] / sma_10) - 1.0;
   double dist_sma_20 = (closes[s] / sma_20) - 1.0;

   double mean_20 = Mean(closes, s, 20);
   double std_20  = StdDev(closes, s, 20);
   double zscore_20 = 0.0;
   if(std_20 > 0.0)
      zscore_20 = (closes[s] - mean_20) / std_20;

   atr14 = CalcATR(rates, s, 14);

   features.Resize(1, FEATURE_COUNT);
   features[0][0] = (float)ret_1;
   features[0][1] = (float)ret_3;
   features[0][2] = (float)ret_5;
   features[0][3] = (float)ret_10;
   features[0][4] = (float)vol_10;
   features[0][5] = (float)vol_20;
   features[0][6] = (float)dist_sma_10;
   features[0][7] = (float)dist_sma_20;
   features[0][8] = (float)zscore_20;
   features[0][9] = (float)atr14;

   return true;
  }


bool PredictNextReturn(double &prediction, double &atr14)
  {
   matrixf x;
   if(!BuildFeatureVector(x, atr14))
      return false;

   vectorf y(1);
   if(!OnnxRun(g_model_handle, ONNX_NO_CONVERSION, x, y))
     {
      Print("OnnxRun failed. Error=", GetLastError());
      return false;
     }

   prediction = (double)y[0];
   return true;
  }


SignalDirection SignalFromPrediction(double prediction)
  {
   if(prediction > InpEntryThreshold && InpAllowLong)
      return SIGNAL_BUY;

   if(prediction < -InpEntryThreshold && InpAllowShort)
      return SIGNAL_SELL;

   return SIGNAL_FLAT;
  }


bool HasOpenPosition(long &pos_type, double &pos_price)
  {
   if(!PositionSelect(_Symbol))
      return false;

   if((long)PositionGetInteger(POSITION_MAGIC) != InpMagic)
      return false;

   pos_type = (long)PositionGetInteger(POSITION_TYPE);
   pos_price = PositionGetDouble(POSITION_PRICE_OPEN);
   return true;
  }


void CloseOpenPosition()
  {
   if(PositionSelect(_Symbol) && (long)PositionGetInteger(POSITION_MAGIC) == InpMagic)
     {
      trade.PositionClose(_Symbol);
     }
  }


void OpenTrade(SignalDirection signal, double atr14)
  {
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);

   double min_stop = (double)SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL) * point;
   double sl_dist = MathMax(atr14 * InpStopAtrMultiple, min_stop);
   double tp_dist = MathMax(atr14 * InpTakeAtrMultiple, min_stop);

   double sl = 0.0;
   double tp = 0.0;

   trade.SetExpertMagicNumber(InpMagic);
   trade.SetDeviationInPoints(20);

   if(signal == SIGNAL_BUY)
     {
      if(InpUseAtrStops)
        {
         sl = ask - sl_dist;
         tp = ask + tp_dist;
        }
      if(trade.Buy(InpLots, _Symbol, ask, sl, tp, "ML buy"))
         g_bars_in_trade = 0;
     }
   else if(signal == SIGNAL_SELL)
     {
      if(InpUseAtrStops)
        {
         sl = bid + sl_dist;
         tp = bid - tp_dist;
        }
      if(trade.Sell(InpLots, _Symbol, bid, sl, tp, "ML sell"))
         g_bars_in_trade = 0;
     }
  }


void ManageExistingPosition(SignalDirection signal)
  {
   long pos_type;
   double pos_price;
   if(!HasOpenPosition(pos_type, pos_price))
      return;

   g_bars_in_trade++;

   bool should_close = false;

   if(InpCloseOnOppositeSignal)
     {
      if(pos_type == POSITION_TYPE_BUY  && signal == SIGNAL_SELL)
         should_close = true;
      if(pos_type == POSITION_TYPE_SELL && signal == SIGNAL_BUY)
         should_close = true;
     }

   if(!should_close && g_bars_in_trade >= InpMaxBarsInTrade)
      should_close = true;

   if(should_close)
      CloseOpenPosition();
  }


int OnInit()
  {
   trade.SetExpertMagicNumber(InpMagic);

   g_model_handle = OnnxCreateFromBuffer(ExtModel, ONNX_DEFAULT);
   if(g_model_handle == INVALID_HANDLE)
     {
      Print("OnnxCreateFromBuffer failed. Error=", GetLastError());
      return INIT_FAILED;
     }

   if(!OnnxSetInputShape(g_model_handle, 0, EXT_INPUT_SHAPE))
     {
      Print("OnnxSetInputShape failed. Error=", GetLastError());
      OnnxRelease(g_model_handle);
      g_model_handle = INVALID_HANDLE;
      return INIT_FAILED;
     }

   if(!OnnxSetOutputShape(g_model_handle, 0, EXT_OUTPUT_SHAPE))
     {
      Print("OnnxSetOutputShape failed. Error=", GetLastError());
      OnnxRelease(g_model_handle);
      g_model_handle = INVALID_HANDLE;
      return INIT_FAILED;
     }

   return INIT_SUCCEEDED;
  }


void OnDeinit(const int reason)
  {
   if(g_model_handle != INVALID_HANDLE)
     {
      OnnxRelease(g_model_handle);
      g_model_handle = INVALID_HANDLE;
     }
  }


void OnTick()
  {
   if(!IsNewBar())
      return;

   double prediction = 0.0;
   double atr14 = 0.0;
   if(!PredictNextReturn(prediction, atr14))
      return;

   SignalDirection signal = SignalFromPrediction(prediction);

   PrintFormat("ML prediction=%.8f threshold=%.8f signal=%d atr14=%.5f", prediction, InpEntryThreshold, signal, atr14);

   ManageExistingPosition(signal);

   long pos_type;
   double pos_price;
   if(HasOpenPosition(pos_type, pos_price))
      return;

   if(signal == SIGNAL_BUY || signal == SIGNAL_SELL)
      OpenTrade(signal, atr14);
  }
