import React, { useEffect, useState } from "react";
import { Bar, BarChart, CartesianGrid, ErrorBar, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { getResults } from "./api";

export default function ComparisonModal({ onClose }) {
  const [rows, setRows] = useState([]);
  const [error, setError] = useState("");

  useEffect(() => {
    function onKey(event) {
      if (event.key === "Escape") onClose();
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  useEffect(() => {
    getResults()
      .then((body) => {
        const split = body.comparison.test;
        setRows(Object.keys(split).map((name) => ({
          name,
          rate: split[name].similarity_pass_rate,
          wins: split[name].wins,
          n: split[name].n,
          error: [
            split[name].similarity_pass_rate - split[name].wilson_low,
            split[name].wilson_high - split[name].similarity_pass_rate,
          ],
        })));
      })
      .catch((err) => setError(err.message));
  }, []);

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal card" onClick={(event) => event.stopPropagation()} role="dialog">
        <h3>Test similarity pass rate</h3>
        <p className="muted">Cosine similarity above 0.52. Not factual accuracy. Bars show 95% Wilson intervals.</p>
        {error && <p className="error-text">{error}</p>}
        {rows.length > 0 && (
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={rows}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis domain={[0, 1]} />
              <Tooltip formatter={(value, _name, item) => [`${item.payload.wins}/${item.payload.n} (${(value * 100).toFixed(0)}%)`, "Similarity pass rate"]} />
              <Bar dataKey="rate" fill="#2f6fed">
                <ErrorBar dataKey="error" width={4} stroke="#16325c" />
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        )}
        <button type="button" className="primary" onClick={onClose}>Close</button>
      </div>
    </div>
  );
}
