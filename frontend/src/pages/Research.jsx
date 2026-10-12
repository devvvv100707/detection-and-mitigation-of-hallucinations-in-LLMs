import React, { useEffect, useState } from "react";
import { Bar, BarChart, CartesianGrid, ErrorBar, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { figureUrl, getResults } from "../api";

const ORDER = ["A", "B", "C", "D", "E-nov", "E"];

export default function Research() {
  const [body, setBody] = useState(null);
  const [error, setError] = useState("");
  const [split, setSplit] = useState("test");

  useEffect(() => {
    getResults().then(setBody).catch((err) => setError(err.message));
  }, []);

  if (error) return <p className="card error-text">{error}</p>;
  if (!body) return <p className="card">Loading saved results...</p>;

  const block = body.comparison[split];
  const rows = ORDER.map((name) => ({
    name,
    rate: block[name].similarity_pass_rate,
    wins: block[name].wins,
    n: block[name].n,
    error: [block[name].similarity_pass_rate - block[name].wilson_low, block[name].wilson_high - block[name].similarity_pass_rate],
    pubmed: block[name].pubmed_rate,
    pubmedWins: block[name].pubmed_wins,
    pubmedN: block[name].pubmed_n,
    pubmedError: [block[name].pubmed_rate - block[name].pubmed_wilson[0], block[name].pubmed_wilson[1] - block[name].pubmed_rate],
    hcm: block[name].hcm_rate,
    hcmWins: block[name].hcm_wins,
    hcmN: block[name].hcm_n,
    hcmError: [block[name].hcm_rate - block[name].hcm_wilson[0], block[name].hcm_wilson[1] - block[name].hcm_rate],
    latency: block[name].mean_latency_seconds,
  }));
  const tiers = split === "test" ? body.comparison.test.E.tier_counts : body.comparison.validation.E.tier_counts;
  const tierRows = [
    { name: "accept", count: tiers.accept || 0 },
    { name: "corrected", count: tiers.corrected || 0 },
    { name: "uncertain", count: tiers.uncertain || 0 },
  ];
  const figures = new Set(body.figures);
  const reproduced = body.reproduced_baseline;
  const original = body.original_experiment;

  return (
    <main>
      <section className="card">
        <h2>Validation / test</h2>
        <div className="switch">
          <button type="button" className={split === "validation" ? "primary" : ""} onClick={() => setSplit("validation")}>Validation</button>
          <button type="button" className={split === "test" ? "primary" : ""} onClick={() => setSplit("test")}>Test</button>
        </div>
        <p className="muted">{body.similarity_pass_rate_definition}</p>
      </section>

      <ChartCard title="Similarity pass rate" figure={figures.has(`comparison_${split}.png`) ? `comparison_${split}.png` : ""}>
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={rows}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" />
            <YAxis domain={[0, 1]} />
            <Tooltip formatter={(value, _n, item) => [`${item.payload.wins}/${item.payload.n}`, "Similarity pass rate"]} />
            <Bar dataKey="rate" fill="#2f6fed" name="Similarity pass rate">
              <ErrorBar dataKey="error" width={4} stroke="#16325c" />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </ChartCard>

      <ChartCard title="Dataset-wise similarity pass rate" figure={figures.has(`dataset_${split}.png`) ? `dataset_${split}.png` : ""}>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={rows}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" />
            <YAxis domain={[0, 1]} />
            <Legend />
            <Tooltip formatter={(value, name, item) => {
              const wins = name === "PubMedQA" ? item.payload.pubmedWins : item.payload.hcmWins;
              const total = name === "PubMedQA" ? item.payload.pubmedN : item.payload.hcmN;
              return [`${wins}/${total} (${(value * 100).toFixed(0)}%)`, name];
            }} />
            <Bar dataKey="pubmed" fill="#3498db" name="PubMedQA">
              <ErrorBar dataKey="pubmedError" width={4} stroke="#16325c" />
            </Bar>
            <Bar dataKey="hcm" fill="#1abc9c" name="HealthCareMagic">
              <ErrorBar dataKey="hcmError" width={4} stroke="#16325c" />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </ChartCard>

      <ChartCard title="E tier distribution" figure={split === "test" && figures.has("tiers_test.png") ? "tiers_test.png" : ""}>
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={tierRows}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" />
            <YAxis allowDecimals={false} />
            <Tooltip />
            <Bar dataKey="count" fill="#e67e22" name="Questions" />
          </BarChart>
        </ResponsiveContainer>
      </ChartCard>

      <ChartCard title="Mean latency (seconds)" figure={figures.has(`latency_${split}.png`) ? `latency_${split}.png` : ""}>
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={rows}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" />
            <YAxis />
            <Tooltip />
            <Bar dataKey="latency" fill="#8e44ad" name="Mean latency (seconds)" />
          </BarChart>
        </ResponsiveContainer>
      </ChartCard>

      <section className="card">
        <h2>Counts</h2>
        <table>
          <thead>
            <tr>
              <th>Config</th>
              <th>Similarity pass</th>
              <th>PubMedQA</th>
              <th>HealthCareMagic</th>
              <th>Mean similarity</th>
              <th>Mean latency (s)</th>
            </tr>
          </thead>
          <tbody>
            {ORDER.map((name) => (
              <tr key={name}>
                <td>{name}</td>
                <td>{block[name].wins}/{block[name].n}</td>
                <td>{block[name].pubmed_wins}/{block[name].pubmed_n}</td>
                <td>{block[name].hcm_wins}/{block[name].hcm_n}</td>
                <td>{block[name].mean_similarity.toFixed(3)}</td>
                <td>{block[name].mean_latency_seconds.toFixed(2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      <section className="card">
        <h2>Preserved original vs reproduced baseline</h2>
        <p>Original figure: similarity pass rate {Math.round(original.similarity_pass_rate * 100)}% ({original.similarity_pass_count}), detector flags {original.detector_flag_rate_count}, mean final similarity {original.mean_final_similarity}. Not reproducible.</p>
        <p>Reproduced baseline, the reference for later runs: similarity pass rate {Math.round(reproduced.similarity_pass_rate * 100)}% , detector flags {reproduced.detector_flags}/20, mean final similarity {reproduced.mean_final_similarity.toFixed(3)}.</p>
      </section>

      <section className="card">
        <h2>Manual-label metrics</h2>
        {body.label_metrics.status === "labels_missing" ? (
          <p>Pending: labels not filled in yet</p>
        ) : (
          <LabelMetrics metrics={body.label_metrics} figures={figures} />
        )}
      </section>

      <section className="card">
        <h2>Metric definitions & limitations</h2>
        <ul>
          <li>The reported comparison is a small pilot: 20 validation and 20 test questions, 10 PubMedQA and 10 HealthCareMagic in each.</li>
          <li>A similarity pass rate is cosine similarity to the reference text above 0.52. It is not factual correctness.</li>
          <li>HealthCareMagic doctor replies are references, not verified ground truth.</li>
          <li>MiniLM similarity cannot detect that a context contradicts an answer. Verification returns only Supported or Insufficient evidence.</li>
          <li>The verify cutoff was selected on validation by similarity to the reference, which favors permissive cutoffs. The chosen value is 0.30.</li>
          <li>E and D also differ in context length. E-nov keeps D’s mitigation on the capped context, so it separates the cap from verification.</li>
          <li>On the test split, E’s similarity pass count was 7/20 and D’s was 8/20. E’s mean similarity was lower. That result is reported as measured.</li>
        </ul>
      </section>
    </main>
  );
}

function ChartCard({ title, figure, children }) {
  return (
    <section className="card">
      <div className="row-actions">
        <h2>{title}</h2>
        {figure && <a className="primary button-link" href={figureUrl(figure)} download={figure}>Download PNG</a>}
      </div>
      {children}
    </section>
  );
}

function LabelMetrics({ metrics, figures }) {
  const detector = metrics.detector_vs_student_retrieval_labels;
  return (
    <div>
      <p>Precision {formatScore(detector.precision)}, recall {formatScore(detector.recall)}, F1 {formatScore(detector.f1)}. Uncertain labels excluded: {detector.uncertain_labels_excluded}.</p>
      {figures.has("label_prf.png") && <a href={figureUrl("label_prf.png")} download>Download precision/recall/F1 PNG</a>}
      {figures.has("label_confusion.png") && <a href={figureUrl("label_confusion.png")} download>Download confusion matrix PNG</a>}
    </div>
  );
}

function formatScore(value) {
  return value == null ? "n/a" : value.toFixed(3);
}
