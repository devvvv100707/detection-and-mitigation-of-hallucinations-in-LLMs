import React, { useEffect, useState } from "react";
import { Bar, BarChart, CartesianGrid, ErrorBar, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { figureUrl, getResults } from "../api";
import { ChartCard } from "../components/ui";

const ORDER = ["A", "B", "C", "D", "E-nov", "E"];

export default function Research() {
  const [body, setBody] = useState(null);
  const [error, setError] = useState("");
  const [split, setSplit] = useState("test");
  const [openLimits, setOpenLimits] = useState(false);

  useEffect(() => {
    getResults().then(setBody).catch((err) => setError(err.message));
  }, []);

  if (error) return <main className="container"><p className="banner banner-danger">{error}</p></main>;
  if (!body) return <main className="container"><p className="card">Loading saved results...</p></main>;

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
  const tiers = block.E.tier_counts || {};
  const tierRows = [
    { name: "accept", count: tiers.accept || 0 },
    { name: "corrected", count: tiers.corrected || 0 },
    { name: "uncertain", count: tiers.uncertain || 0 },
  ];
  const figures = new Set(body.figures);
  const reproduced = body.reproduced_baseline;
  const original = body.original_experiment;

  return (
    <main className="container stack">
      <div>
        <h2 className="section-title">Research Results</h2>
        <p className="muted">Saved comparison for the seed-42 split. A similarity pass rate is cosine similarity to the reference text above 0.52. It is not factual accuracy.</p>
      </div>
      <div className="segmented" role="group" aria-label="Split">
        <button type="button" aria-pressed={split === "validation"} onClick={() => setSplit("validation")}>Validation</button>
        <button type="button" aria-pressed={split === "test"} onClick={() => setSplit("test")}>Test</button>
      </div>
      <p className="muted">{body.similarity_pass_rate_definition}</p>
      <div className="chart-grid">
        <ChartCard title="Similarity pass rate" description="Share of answers above the 0.52 cutoff, with 95% Wilson intervals." figure={figures.has(`comparison_${split}.png`) ? `comparison_${split}.png` : ""}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={rows}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" interval={0} tick={{ fontSize: 12 }} />
              <YAxis domain={[0, 1]} tick={{ fontSize: 12 }} />
              <Tooltip formatter={(value, _n, item) => [`${item.payload.wins}/${item.payload.n}`, "Similarity pass rate"]} />
              <Bar dataKey="rate" fill="#2f5fd0" name="Similarity pass rate">
                <ErrorBar dataKey="error" width={4} stroke="#1c2740" />
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>
        <ChartCard title="Dataset-wise similarity pass rate" description="PubMedQA and HealthCareMagic counted separately." figure={figures.has(`dataset_${split}.png`) ? `dataset_${split}.png` : ""}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={rows}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" interval={0} tick={{ fontSize: 12 }} />
              <YAxis domain={[0, 1]} tick={{ fontSize: 12 }} />
              <Legend />
              <Tooltip formatter={(value, name, item) => {
                const wins = name === "PubMedQA" ? item.payload.pubmedWins : item.payload.hcmWins;
                const total = name === "PubMedQA" ? item.payload.pubmedN : item.payload.hcmN;
                return [`${wins}/${total} (${(value * 100).toFixed(0)}%)`, name];
              }} />
              <Bar dataKey="pubmed" fill="#2f5fd0" name="PubMedQA"><ErrorBar dataKey="pubmedError" width={4} stroke="#1c2740" /></Bar>
              <Bar dataKey="hcm" fill="#1f8a70" name="HealthCareMagic"><ErrorBar dataKey="hcmError" width={4} stroke="#1c2740" /></Bar>
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>
        <ChartCard title="E tier distribution" description="Accept, corrected, and uncertain counts for this split." figure={split === "test" && figures.has("tiers_test.png") ? "tiers_test.png" : ""}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={tierRows}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" tick={{ fontSize: 12 }} />
              <YAxis allowDecimals={false} tick={{ fontSize: 12 }} />
              <Tooltip />
              <Bar dataKey="count" fill="#c47b00" name="Questions" />
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>
        <ChartCard title="Mean latency" description="Mean seconds per question. Not a similarity pass rate." figure={figures.has(`latency_${split}.png`) ? `latency_${split}.png` : ""}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={rows}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" interval={0} tick={{ fontSize: 12 }} />
              <YAxis tick={{ fontSize: 12 }} />
              <Tooltip />
              <Bar dataKey="latency" fill="#3d4fbf" name="Mean latency (seconds)" />
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>
      </div>
      <section className="card">
        <h2 className="section-title">Counts</h2>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Config</th>
                <th className="num">Similarity pass</th>
                <th className="num">PubMedQA</th>
                <th className="num">HealthCareMagic</th>
                <th className="num">Mean similarity</th>
                <th className="num">Mean latency (s)</th>
              </tr>
            </thead>
            <tbody>
              {ORDER.map((name) => (
                <tr key={name}>
                  <td>{name}</td>
                  <td className="num">{block[name].wins}/{block[name].n}</td>
                  <td className="num">{block[name].pubmed_wins}/{block[name].pubmed_n}</td>
                  <td className="num">{block[name].hcm_wins}/{block[name].hcm_n}</td>
                  <td className="num">{block[name].mean_similarity.toFixed(3)}</td>
                  <td className="num">{block[name].mean_latency_seconds.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
      <section className="card">
        <h2 className="section-title">Preserved original vs reproduced baseline</h2>
        <div className="compare">
          <div>
            <h3 className="card-title">Original figure</h3>
            <p>Similarity pass rate {Math.round(original.similarity_pass_rate * 100)}% ({original.similarity_pass_count}), detector flags {original.detector_flag_rate_count}, mean final similarity {original.mean_final_similarity}. Not reproducible.</p>
          </div>
          <div>
            <h3 className="card-title">Reproduced baseline</h3>
            <p>Similarity pass rate {Math.round(reproduced.similarity_pass_rate * 100)}%, detector flags {reproduced.detector_flags}/20, mean final similarity {reproduced.mean_final_similarity.toFixed(3)}. This is the reference for later runs.</p>
          </div>
        </div>
      </section>
      <section className="card">
        <h2 className="section-title">Manual-label metrics</h2>
        {body.label_metrics.status === "labels_missing" ? (
          <div className="pending">Pending: labels not filled in yet</div>
        ) : (
          <LabelMetrics metrics={body.label_metrics} figures={figures} />
        )}
      </section>
      <section className="card">
        <h2 className="visually-hidden">Metric definitions</h2>
        <button type="button" className="acc-trigger" aria-expanded={openLimits} onClick={() => setOpenLimits((open) => !open)}>
          Metric definitions & limitations <span aria-hidden="true">{openLimits ? "−" : "+"}</span>
        </button>
        {openLimits && (
          <ul>
            <li>The reported comparison is a small pilot: 20 validation and 20 test questions, 10 PubMedQA and 10 HealthCareMagic in each.</li>
            <li>A similarity pass rate is cosine similarity to the reference text above 0.52. It is not factual correctness.</li>
            <li>HealthCareMagic doctor replies are references, not verified ground truth.</li>
            <li>MiniLM similarity cannot detect that a context contradicts an answer. Verification returns only Supported or Insufficient evidence.</li>
            <li>The verify cutoff was selected on validation by similarity to the reference, which favors permissive cutoffs. The chosen value is 0.30.</li>
            <li>E and D also differ in context length. E-nov keeps D’s mitigation on the capped context, so it separates the cap from verification.</li>
            <li>On the test split, E’s similarity pass count was 7/20 and D’s was 8/20. E’s mean similarity was lower. That result is reported as measured.</li>
          </ul>
        )}
      </section>
    </main>
  );
}

function LabelMetrics({ metrics, figures }) {
  const detector = metrics.detector_vs_student_retrieval_labels;
  return (
    <div>
      <p>Precision {formatScore(detector.precision)}, recall {formatScore(detector.recall)}, F1 {formatScore(detector.f1)}. Uncertain labels excluded: {detector.uncertain_labels_excluded}.</p>
      {figures.has("label_prf.png") && <a className="btn btn-secondary" href={figureUrl("label_prf.png")} download>Download precision/recall/F1 PNG</a>}
      {figures.has("label_confusion.png") && <a className="btn btn-secondary" href={figureUrl("label_confusion.png")} download>Download confusion matrix PNG</a>}
    </div>
  );
}

function formatScore(value) {
  return value == null ? "Not computed" : value.toFixed(3);
}
