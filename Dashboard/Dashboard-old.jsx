import React, { useState, useEffect, useMemo, useRef } from "react";
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend
} from "recharts";
import "./Dashboard.css";

const Dashboard = () => {
  const [cryptos, setCryptos] = useState([]);
  const [prices, setPrices] = useState({});
  const [wallet, setWallet] = useState(null);
  const [predictions, setPredictions] = useState({});
  const [predictionsLoading, setPredictionsLoading] = useState(false);
  const [predictionsError, setPredictionsError] = useState(null);
  const [predictionThreshold, setPredictionThreshold] = useState(0.6);
  const [predictionsTimestamp, setPredictionsTimestamp] = useState(null);
  const [predictionsCryptoFilter, setPredictionsCryptoFilter] = useState(null);
  const [loading, setLoading] = useState(true);
  const [selectedCrypto, setSelectedCrypto] = useState(null);
  const [showWalletDialog, setShowWalletDialog] = useState(false);
  const [showPriceDialog, setShowPriceDialog] = useState(false);
  const [showPredictionDialog, setShowPredictionDialog] = useState(false);
  const [refreshInterval, setRefreshInterval] = useState(60);
  const [range, setRange] = useState("1h");
  const [isStreaming, setIsStreaming] = useState(false);
  const streamRef = useRef(null);

  const API_BASE_URL = "http://localhost:5000/api";

  const fetchAllData = async () => {
    try {
      setLoading(true);

      // Fetch cryptos list
      const cryptosRes = await fetch(`${API_BASE_URL}/cryptos`);
      const cryptosData = await cryptosRes.json();
      if (cryptosData.success) {
        setCryptos(cryptosData.data);
        if (!selectedCrypto && cryptosData.data.length > 0) {
          setSelectedCrypto(cryptosData.data[0].symbol);
        }
      }

      // Fetch prices (fallback if websocket not connected)
      const pricesRes = await fetch(`${API_BASE_URL}/prices`);
      const pricesData = await pricesRes.json();
      if (pricesData.success) {
        setPrices(pricesData.prices || pricesData.data || {});
      }

      // Fetch wallet
      const walletRes = await fetch(`${API_BASE_URL}/wallet`);
      const walletData = await walletRes.json();
      if (walletData.success) {
        setWallet(walletData.data);
      }

      // Fetch predictions
      const predictionsRes = await fetch(`${API_BASE_URL}/predictions`);
      const predictionsData = await predictionsRes.json();
      if (predictionsData.success) {
        setPredictions(predictionsData.data);
      }

      setLoading(false);
    } catch (error) {
      console.error("Error fetching data:", error);
      setLoading(false);
    }
  };

  const runPredictions = async () => {
    setPredictionsLoading(true);
    setPredictionsError(null);
    try {
      const res = await fetch(`${API_BASE_URL}/predictions/refresh`, {
        method: "POST"
      });
      const data = await res.json();
      if (data.success) {
        setPredictions(data.data || {});
        setPredictionsTimestamp(new Date());
      } else {
        setPredictionsError(data.error || "Failed to run predictions");
      }
    } catch (err) {
      setPredictionsError(err.message || "Failed to run predictions");
    } finally {
      setPredictionsLoading(false);
    }
  };

  // Polling (disabled when streaming is active)
  useEffect(() => {
    fetchAllData();
    if (isStreaming) return undefined;
    const interval = setInterval(fetchAllData, refreshInterval * 1000);
    return () => clearInterval(interval);
  }, [refreshInterval, isStreaming]);

  // Subscribe to SSE stream when available; fallback to polling automatically
  useEffect(() => {
    const base = API_BASE_URL.replace(/\/api$/, "");
    const sseUrl = `${base}/stream/prices`;
    let es;

    try {
      es = new EventSource(sseUrl);
      streamRef.current = es;
      es.onopen = () => setIsStreaming(true);
      es.onmessage = (event) => {
        try {
          const payload = JSON.parse(event.data);
          if (payload?.prices || payload?.data) {
            setPrices(payload.prices || payload.data);
          }
        } catch (err) {
          console.error("SSE parse error", err);
        }
      };
      es.onerror = () => {
        setIsStreaming(false);
        es.close();
      };
    } catch (err) {
      setIsStreaming(false);
    }

    return () => {
      if (streamRef.current) {
        streamRef.current.close();
      }
    };
  }, []);

  const filteredHistory = useMemo(() => {
    if (!selectedCrypto || !prices[selectedCrypto]) return [];
    const history = prices[selectedCrypto].history || [];
    if (history.length === 0) return [];

    const now = new Date();
    const rangeMinutes =
      {
        "15m": 15,
        "1h": 60,
        "6h": 360,
        "24h": 1440
      }[range] || 60;

    const sliced = history.filter((item) => {
      const ts = new Date(item.timestamp);
      const diffMinutes = (now - ts) / (1000 * 60);
      return diffMinutes <= rangeMinutes;
    });

    // Fallback: if not enough points (e.g., backend only has 1h), use whatever is available
    if (sliced.length < 5) return history;
    return sliced;
  }, [prices, selectedCrypto, range]);

  const yDomain = useMemo(() => {
    if (!filteredHistory.length) return ["auto", "auto"];
    const highs = filteredHistory.map((h) => h.high ?? h.close);
    const lows = filteredHistory.map((h) => h.low ?? h.close);
    const max = Math.max(...highs);
    const min = Math.min(...lows);
    const pad = (max - min) * 0.02 || max * 0.01 || 1;
    return [min - pad, max + pad];
  }, [filteredHistory]);

  const getPredictionSignal = (symbol) => {
    const symbolPreds = Object.entries(predictions)
      .filter(([key]) => key.includes(symbol))
      .map(([, value]) => value);

    if (symbolPreds.length === 0) return null;

    // Aggregate predictions
    const buySignals = symbolPreds.filter((p) => p.signal === "BUY").length;
    const sellSignals = symbolPreds.filter((p) => p.signal === "SELL").length;

    if (buySignals > sellSignals) {
      return { signal: "BUY", confidence: buySignals };
    } else if (sellSignals > buySignals) {
      return { signal: "SELL", confidence: sellSignals };
    }
    return { signal: "HOLD", confidence: 0 };
  };

  const formatPrice = (price) => {
    if (price === undefined || price === null) return "N/A";
    return `$${Number(price).toLocaleString(undefined, {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    })}`;
  };

  const formatChange = (change) => {
    if (change === undefined || change === null) return "N/A";
    return `${change > 0 ? "+" : ""}${change.toFixed(2)}%`;
  };

  const formatPercent = (value) => {
    if (value === undefined || value === null) return "N/A";
    return `${(Number(value) * 100).toFixed(1)}%`;
  };

  const getTimeSinceUpdate = () => {
    if (!predictionsTimestamp) return "Never";
    const now = new Date();
    const diff = Math.floor((now - predictionsTimestamp) / 1000);
    if (diff < 60) return `${diff}s ago`;
    if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
    return `${Math.floor(diff / 3600)}h ago`;
  };

  const [timeDisplay, setTimeDisplay] = useState("Never");

  useEffect(() => {
    const timer = setInterval(() => {
      setTimeDisplay(getTimeSinceUpdate());
    }, 1000);
    return () => clearInterval(timer);
  }, [predictionsTimestamp]);

  const filteredPredictions = useMemo(() => {
    const list = Object.values(predictions || {});
    return list
      .filter((p) => Number(p.confidence ?? 0) >= predictionThreshold)
      .filter(
        (p) => !predictionsCryptoFilter || p.symbol === predictionsCryptoFilter
      )
      .sort((a, b) => Number(b.confidence ?? 0) - Number(a.confidence ?? 0));
  }, [predictions, predictionThreshold, predictionsCryptoFilter]);

  if (loading) {
    return (
      <div className="dashboard loading">
        <div className="loader">Loading Dashboard...</div>
      </div>
    );
  }

  return (
    <div className="dashboard">
      {/* Header */}
      <header className="dashboard-header">
        <div className="header-content">
          <h1>📊 Crypto Trading Dashboard</h1>
          <div className="header-controls">
            {isStreaming ? (
              <div
                className="live-indicator"
                title="Live via server-sent events"
              >
                ● Live stream active
              </div>
            ) : (
              <label>
                Auto-refresh:
                <select
                  value={refreshInterval}
                  onChange={(e) => setRefreshInterval(Number(e.target.value))}
                >
                  <option value={30}>30s</option>
                  <option value={60}>1m</option>
                  <option value={300}>5m</option>
                </select>
              </label>
            )}
            <button className="btn-primary" onClick={fetchAllData}>
              ↻ Refresh now
            </button>
            <button
              className="btn-wallet"
              onClick={() => setShowWalletDialog(true)}
            >
              💰 Wallet ({wallet?.asset_count || 0} assets)
            </button>
          </div>
        </div>
      </header>

      {/* Main Grid */}
      <div className="dashboard-grid">
        {/* Crypto Cards */}
        <div className="crypto-section">
          <h2>Market Overview</h2>
          <div className="crypto-grid">
            {cryptos.map((crypto) => {
              const priceData = prices[crypto.symbol];
              const prediction = getPredictionSignal(crypto.symbol);

              if (!priceData || priceData.error) {
                return (
                  <div key={crypto.symbol} className="crypto-card error">
                    <div className="crypto-header">
                      <h3>{crypto.symbol}</h3>
                    </div>
                    <p className="error-message">Data unavailable</p>
                  </div>
                );
              }

              return (
                <div key={crypto.symbol} className="crypto-card">
                  <div className="crypto-header">
                    <h3>{crypto.symbol}</h3>
                    <span
                      className={`badge ${
                        priceData["24h_change"] >= 0 ? "up" : "down"
                      }`}
                    >
                      {formatChange(priceData["24h_change"])}
                    </span>
                  </div>

                  <div className="crypto-price">
                    <div className="price-value">
                      {formatPrice(priceData.current_price)}
                    </div>
                    <div className="price-range">
                      <small>H: {formatPrice(priceData["24h_high"])}</small>
                      <small>L: {formatPrice(priceData["24h_low"])}</small>
                    </div>
                  </div>

                  <div className="prediction-signal">
                    {prediction && (
                      <button
                        className={`signal ${prediction.signal.toLowerCase()}`}
                        onClick={() => {
                          setPredictionsCryptoFilter(crypto.symbol);
                          setShowPredictionDialog(true);
                        }}
                      >
                        <strong>{prediction.signal}</strong>
                      </button>
                    )}
                  </div>

                  <button
                    className="btn-details"
                    onClick={() => {
                      setSelectedCrypto(crypto.symbol);
                      setShowPriceDialog(true);
                    }}
                  >
                    View Chart
                  </button>
                </div>
              );
            })}
          </div>
        </div>

        {/* Portfolio Summary */}
        {wallet && (
          <div className="portfolio-section">
            <h2>Portfolio Summary</h2>
            <div className="portfolio-stats">
              <div className="stat">
                <label>Total Value</label>
                <div className="stat-value">
                  ${wallet.total_value_usdt.toLocaleString()}
                </div>
              </div>
              <div className="stat">
                <label>Assets</label>
                <div className="stat-value">{wallet.asset_count}</div>
              </div>
            </div>
            <button
              className="btn-primary"
              onClick={() => setShowWalletDialog(true)}
            >
              View Detailed Wallet
            </button>
          </div>
        )}
      </div>

      {/* Price Chart Dialog */}
      {showPriceDialog && selectedCrypto && prices[selectedCrypto] && (
        <dialog className="dialog dialog-price" open>
          <div className="dialog-content">
            <button
              className="btn-close"
              onClick={() => setShowPriceDialog(false)}
            >
              ×
            </button>
            <h2>
              {selectedCrypto} - Price Chart ({range})
              {isStreaming ? " • live" : " • polling"}
            </h2>
            <div className="range-toggle">
              {[
                { label: "15m", value: "15m" },
                { label: "1h", value: "1h" },
                { label: "6h", value: "6h" },
                { label: "24h", value: "24h" }
              ].map((opt) => (
                <button
                  key={opt.value}
                  className={`range-btn ${range === opt.value ? "active" : ""}`}
                  onClick={() => setRange(opt.value)}
                >
                  {opt.label}
                </button>
              ))}
            </div>
            <div className="chart-container">
              <ResponsiveContainer width="100%" height={440}>
                <LineChart
                  data={filteredHistory}
                  margin={{ top: 10, right: 36, left: 28, bottom: 20 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#444" />
                  <XAxis
                    dataKey="timestamp"
                    tickFormatter={(val) =>
                      new Date(val).toLocaleTimeString([], {
                        hour: "2-digit",
                        minute: "2-digit"
                      })
                    }
                    tick={{ fill: "#aaa" }}
                    padding={{ left: 24, right: 24 }}
                    minTickGap={20}
                  />
                  <YAxis
                    tick={{ fill: "#aaa" }}
                    domain={yDomain}
                    tickFormatter={(v) => (v || v === 0 ? v.toFixed(2) : v)}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "#1a1a2e",
                      border: "1px solid #16213e"
                    }}
                    labelStyle={{ color: "#0f3460" }}
                    formatter={(value) =>
                      value || value === 0 ? value.toFixed(2) : value
                    }
                    labelFormatter={(label) => new Date(label).toLocaleString()}
                  />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="close"
                    stroke="#00d4ff"
                    dot={false}
                    name="Price"
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
            <div className="chart-stats">
              <div>
                Current: {formatPrice(prices[selectedCrypto].current_price)}
              </div>
              <div>
                24h High: {formatPrice(prices[selectedCrypto]["24h_high"])}
              </div>
              <div>
                24h Low: {formatPrice(prices[selectedCrypto]["24h_low"])}
              </div>
              <div
                className={
                  prices[selectedCrypto]["24h_change"] >= 0
                    ? "positive"
                    : "negative"
                }
              >
                Change: {formatChange(prices[selectedCrypto]["24h_change"])}
              </div>
            </div>
          </div>
        </dialog>
      )}

      {/* Wallet Dialog */}
      {showWalletDialog && wallet && (
        <dialog className="dialog" open>
          <div className="dialog-content dialog-large">
            <button
              className="btn-close"
              onClick={() => setShowWalletDialog(false)}
            >
              ×
            </button>
            <h2>Wallet Details</h2>

            <div className="wallet-summary">
              <div className="wallet-stat">
                <label>Total Value</label>
                <div className="value">
                  ${wallet.total_value_usdt.toLocaleString()}
                </div>
              </div>
              <div className="wallet-stat">
                <label>Assets</label>
                <div className="value">{wallet.asset_count}</div>
              </div>
            </div>

            <table className="wallet-table">
              <thead>
                <tr>
                  <th>Asset</th>
                  <th>Free</th>
                  <th>Locked</th>
                  <th>Total</th>
                  <th>Price (USDT)</th>
                  <th>Value (USDT)</th>
                </tr>
              </thead>
              <tbody>
                {wallet.balances.map((balance) => (
                  <tr key={balance.asset}>
                    <td className="asset-name">{balance.asset}</td>
                    <td>{balance.free.toFixed(4)}</td>
                    <td>{balance.locked.toFixed(4)}</td>
                    <td className="total">{balance.total.toFixed(4)}</td>
                    <td>
                      {balance.price_usdt
                        ? `$${balance.price_usdt.toFixed(2)}`
                        : "N/A"}
                    </td>
                    <td className="value">${balance.value_usdt.toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </dialog>
      )}

      {/* Predictions Dialog */}
      {showPredictionDialog && (
        <dialog className="dialog" open>
          <div className="dialog-content dialog-large">
            <button
              className="btn-close"
              onClick={() => setShowPredictionDialog(false)}
            >
              ×
            </button>
            <div className="predictions-header-inline">
              <div className="predictions-title">
                <h2>Model Predictions</h2>
                <p className="predictions-timestamp">
                  Last fetched: <strong>{timeDisplay}</strong>
                </p>
              </div>
              <div className="predictions-controls">
                <label className="inline-filter">
                  Crypto:
                  <select
                    value={predictionsCryptoFilter || ""}
                    onChange={(e) =>
                      setPredictionsCryptoFilter(e.target.value || null)
                    }
                  >
                    <option value="">All</option>
                    {cryptos.map((c) => (
                      <option key={c.symbol} value={c.symbol}>
                        {c.symbol}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="inline-filter">
                  Min confidence:
                  <select
                    value={predictionThreshold}
                    onChange={(e) =>
                      setPredictionThreshold(Number(e.target.value))
                    }
                  >
                    <option value={0}>All</option>
                    {[0.6, 0.7, 0.8, 0.9].map((v) => (
                      <option key={v} value={v}>{`${(v * 100).toFixed(
                        0
                      )}%`}</option>
                    ))}
                  </select>
                </label>
                {predictionsError && (
                  <span className="error-text">{predictionsError}</span>
                )}
                <button
                  className="btn-primary"
                  onClick={runPredictions}
                  disabled={predictionsLoading}
                >
                  {predictionsLoading ? "Running..." : "Get Predictions"}
                </button>
              </div>
            </div>
            {filteredPredictions.length > 0 ? (
              <div className="predictions-table-wrapper">
                <table className="predictions-table">
                  <thead>
                    <tr>
                      <th>Symbol</th>
                      <th>Interval</th>
                      <th>Signal</th>
                      <th>Direction</th>
                      <th>Confidence</th>
                      <th>Current</th>
                      <th>Target</th>
                      <th>Updated</th>
                      <th>Valid Until</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredPredictions.map((pred, idx) => {
                      const target =
                        pred.price_estimate?.target_price ??
                        pred.price_estimate?.target ??
                        null;
                      return (
                        <tr
                          key={`${pred.symbol}-${pred.interval}-${pred.horizon_minutes}-${idx}`}
                          className={`signal-row ${
                            pred.signal?.toLowerCase() || "neutral"
                          }`}
                        >
                          <td>{pred.symbol}</td>
                          <td>{pred.interval}</td>
                          <td className="signal-cell">
                            {pred.signal || "NEUTRAL"}
                          </td>
                          <td>{pred.predicted_direction || ""}</td>
                          <td>{formatPercent(pred.confidence)}</td>
                          <td>{formatPrice(pred.current_price)}</td>
                          <td>{target ? formatPrice(target) : "-"}</td>
                          <td>
                            {pred.prediction_time
                              ? new Date(
                                  pred.prediction_time
                                ).toLocaleTimeString([], {
                                  hour: "2-digit",
                                  minute: "2-digit",
                                  second: "2-digit"
                                })
                              : "-"}
                          </td>
                          <td>
                            {pred.valid_until
                              ? new Date(pred.valid_until).toLocaleTimeString(
                                  [],
                                  {
                                    hour: "2-digit",
                                    minute: "2-digit",
                                    second: "2-digit"
                                  }
                                )
                              : "-"}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            ) : (
              <p className="no-data">
                No predictions meeting the selected confidence. Click "Get
                Predictions" to run the models or lower the filter.
              </p>
            )}
          </div>
        </dialog>
      )}

      {/* Footer */}
      <footer className="dashboard-footer">
        <p>Last updated: {new Date().toLocaleTimeString()}</p>
        <button
          className="btn-predictions"
          onClick={() => setShowPredictionDialog(true)}
        >
          🤖 View All Predictions
        </button>
      </footer>
    </div>
  );
};

export default Dashboard;
