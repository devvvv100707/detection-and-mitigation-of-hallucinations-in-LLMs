import React, { useEffect, useRef, useState } from "react";
import { Bar, BarChart, CartesianGrid, ErrorBar, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { getResults } from "./api";

export default function ComparisonModal({ onClose }) {
  const dialogRef = useRef(null);
  const [rows, setRows] = useState([]);
  const [error, setError] = useState("");

  useEffect(() => {
    const previous = document.activeElement;
    const bodyOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    const root = dialogRef.current;
    const focusable = () => Array.from(root.querySelectorAll("button, a, [href], input, select, textarea, [tabindex]:not([tabindex='-1'])"));
    focusable()[0]?.focus();
    function onKey(event) {
      if (event.key === "Escape") onClose();
      if (event.key !== "Tab") return;
      const items = focusable();
      if (!items.length) return;
      const first = items[0];
      const last = items[items.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    }
    window.addEventListener("keydown", onKey);
    return () => {
      document.body.style.overflow = bodyOverflow;
      window.removeEventListener("keydown", onKey);
      previous?.focus?.();
    };
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
      <div className="modal card" role="dialog" aria-modal="true" aria-labelledby="compare-title" ref={dialogRef} onClick={(event) => event.stopPropagation()}>
        <h2 id="compare-title" className="section-title">Test similarity pass rate</h2>
        <p className="muted">Cosine similarity above 0.52. Not factual accuracy. Bars show 95% Wilson intervals.</p>
        {error && <p className="banner banner-danger">{error}</p>}
        {rows.length > 0 && (
          <div className="chart-body">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={rows}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" interval={0} tick={{ fontSize: 12 }} />
                <YAxis domain={[0, 1]} tick={{ fontSize: 12 }} />
                <Tooltip formatter={(value, _name, item) => [`${item.payload.wins}/${item.payload.n} (${(value * 100).toFixed(0)}%)`, "Similarity pass rate"]} />
                <Bar dataKey="rate" fill="#2f5fd0">
                  <ErrorBar dataKey="error" width={4} stroke="#1c2740" />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
        <button type="button" className="btn btn-primary" onClick={onClose}>Close</button>
      </div>
    </div>
  );
}
