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
import io from "socket.io-client";
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
  const [range, setRange] = useState("1h");
  const [wsConnected, setWsConnected] = useState(false);
  const socketRef = useRef(null);

  const API_BASE_URL = "http://localhost:5000/api";
  const WS_URL = "http://localhost:5000";

  // Initialize WebSocket connection
  useEffect(() => {
    const socket = io(WS_URL, {
      reconnection: true,
      reconnectionDelay: 1000,
      reconnectionDelayMax: 5000,
      reconnectionAttempts: Infinity,
      transports: ["websocket", "polling"]
    });

    socketRef.current = socket;

    socket.on("connect", () => {
      setWsConnected(true);
      console.log("✅ WebSocket connected");
    });

    socket.on("disconnect", () => {
      setWsConnected(false);
      console.log("❌ WebSocket disconnected");
    });

    // Receive initial data on connect
    socket.on("initial_data", (data) => {
      if (data.prices) setPrices(data.prices);
      if (data.predictions) setPredictions(data.predictions);
      setLoading(false);
    });

    // Real-time price updates
    socket.on("prices_update", (data) => {
      if (data.prices) setPrices(data.prices);
    });

    // Real-time prediction updates
    socket.on("predictions_update", (data) => {
      if (data.predictions) {
        setPredictions(data.predictions);
        setPredictionsTimestamp(new Date());
      }
    });

    // Fetch cryptos and wallet (use REST for these)
    const fetchInitialData = async () => {
      try {
        const cryptosRes = await fetch(`${API_BASE_URL}/cryptos`);
        const cryptosData = await cryptosRes.json();
        if (cryptosData.success) {
          setCryptos(cryptosData.data);
          if (!selectedCrypto && cryptosData.data.length > 0) {
            setSelectedCrypto(cryptosData.data[0].symbol);
          }
        }

        const walletRes = await fetch(`${API_BASE_URL}/wallet`);
        const walletData = await walletRes.json();
        if (walletData.success) {
          setWallet(walletData.data);
        }
      } catch (error) {
        console.error("Error fetching initial data:", error);
      }
    };

    fetchInitialData();

    return () => {
      socket.disconnect();
    };
  }, []);

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

  const formatPrice = (price) => {
    if (!price && price !== 0) return "N/A";
    if (price > 1000) return `$${(price / 1000).toFixed(2)}k`;
    return `$${price.toFixed(4)}`;
  };

  const formatChange = (change) => {
    const symbol = change >= 0 ? "+" : "";
    const className = change >= 0 ? "positive" : "negative";
    return (
      <span className={className}>
        {symbol}
        {change.toFixed(2)}%
      </span>
    );
  };

  const formatPercent = (value) => {
    if (!value && value !== 0) return "N/A";
    return `${(value * 100).toFixed(1)}%`;
  };

  const timeDisplay = useMemo(() => {
    if (!predictionsTimestamp) return "Never";
    const now = new Date();
    const diff = Math.floor((now - predictionsTimestamp) / 1000);
    if (diff < 60) return `${diff}s ago`;
    if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
    return `${Math.floor(diff / 3600)}h ago`;
  }, [predictionsTimestamp]);

  const filteredHistory = useMemo(() => {
    if (!selectedCrypto || !prices[selectedCrypto]) return [];
    const history = prices[selectedCrypto].history || [];
    if (history.length === 0) return [];

    const now = new Date();
    const rangeMinutes =
      { "15m": 15, "1h": 60, "6h": 360, "24h": 1440 }[range] || 60;
    const cutoff = now.getTime() - rangeMinutes * 60 * 1000;

    return history
      .filter((candle) => new Date(candle.timestamp).getTime() >= cutoff)
      .map((candle) => ({
        ...candle,
        timestamp: new Date(candle.timestamp).getTime()
      }));
  }, [selectedCrypto, prices, range]);

  const yDomain = useMemo(() => {
    if (filteredHistory.length === 0) return ["dataMin", "dataMax"];
    const closes = filteredHistory.map((h) => h.close);
    const min = Math.min(...closes);
    const max = Math.max(...closes);
    const padding = (max - min) * 0.1;
    return [min - padding, max + padding];
  }, [filteredHistory]);

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
      <div className="dashboard">
        <div className="loading">
          <div className="spinner"></div>
          <p>Loading dashboard...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="dashboard">
      <header className="dashboard-header">
        <h1>⚡ Crypto Trading Bot</h1>
        <div className="header-status">
          <span
            className={`ws-status ${
              wsConnected ? "connected" : "disconnected"
            }`}
          >
            {wsConnected ? "🟢 Live" : "🔴 Offline"}
          </span>
        </div>
      </header>

      <div className="dashboard-grid">
        {/* Price Cards */}
        <section className="section">
          <h2>Market Prices</h2>
          <div className="price-cards">
            {cryptos.map((crypto) => {
              const p = prices[crypto.symbol];
              if (!p) return null;
              return (
                <div
                  key={crypto.symbol}
                  className="price-card"
                  onClick={() => {
                    setSelectedCrypto(crypto.symbol);
                    setShowPriceDialog(true);
                  }}
                >
                  <h3>{crypto.symbol}</h3>
                  <div className="price">{formatPrice(p.current_price)}</div>
                  <div
                    className={p["24h_change"] >= 0 ? "positive" : "negative"}
                  >
                    {p["24h_change"] >= 0 ? "↑" : "↓"}{" "}
                    {Math.abs(p["24h_change"]).toFixed(2)}%
                  </div>
                </div>
              );
            })}
          </div>
        </section>

        {/* Wallet Summary */}
        {wallet && (
          <section className="section">
            <h2>Wallet Summary</h2>
            <div
              className="wallet-summary-card"
              onClick={() => setShowWalletDialog(true)}
            >
              <div className="stat">
                <label>Total Value</label>
                <div className="value">
                  ${wallet.total_value_usdt.toLocaleString()}
                </div>
              </div>
              <div className="stat">
                <label>Assets</label>
                <div className="value">{wallet.asset_count}</div>
              </div>
            </div>
          </section>
        )}

        {/* Predictions Summary */}
        <section className="section">
          <h2>AI Predictions</h2>
          <div className="predictions-summary">
            <div className="stat-box buy">
              <span className="count">
                {filteredPredictions.filter((p) => p.signal === "BUY").length}
              </span>
              <span>Buy Signals</span>
            </div>
            <div className="stat-box sell">
              <span className="count">
                {filteredPredictions.filter((p) => p.signal === "SELL").length}
              </span>
              <span>Sell Signals</span>
            </div>
            <div className="stat-box hold">
              <span className="count">
                {filteredPredictions.filter((p) => p.signal === "HOLD").length}
              </span>
              <span>Hold Signals</span>
            </div>
          </div>
          <button
            className="btn-predictions"
            onClick={() => setShowPredictionDialog(true)}
          >
            🤖 View All Predictions
          </button>
        </section>
      </div>

      {/* Price Chart Modal */}
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
                    formatter={(value) => value.toFixed(2)}
                  />
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
          <div className="dialog-content">
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
                          key={`${pred.symbol}-${pred.interval}-${idx}`}
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
                No predictions. Click "Get Predictions" to run models.
              </p>
            )}
          </div>
        </dialog>
      )}

      <footer className="dashboard-footer">
        <p>Last updated: {new Date().toLocaleTimeString()}</p>
        <p>
          {wsConnected ? "🟢 Real-time via WebSocket" : "🔴 Connection issues"}
        </p>
      </footer>
    </div>
  );
};

export default Dashboard;
