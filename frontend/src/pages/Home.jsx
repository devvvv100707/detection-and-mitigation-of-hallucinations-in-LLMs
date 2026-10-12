import React, { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { analyzeQuestion, getResults } from "../api";
import ComparisonModal from "../ComparisonModal";

const TIER_LABEL = {
  accepted: "Accepted answer",
  revised: "Revised Answer",
  uncertain: "Uncertainty Warning",
};

function percent(value) {
  return `${Math.round(value * 100)}%`;
}

export default function Home({ health }) {
  const [question, setQuestion] = useState("");
  const [example, setExample] = useState("");
  const [original, setOriginal] = useState(null);
  const [resultsError, setResultsError] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [analysis, setAnalysis] = useState(null);
  const [showEvidence, setShowEvidence] = useState(false);
  const [modal, setModal] = useState(false);

  useEffect(() => {
    getResults()
      .then((body) => {
        setOriginal(body.original_experiment);
        setExample(body.example_question || "");
      })
      .catch((err) => setResultsError(err.message));
  }, []);

  async function submit(event) {
    event.preventDefault();
    if (!question.trim() || busy || !health.ready) return;
    setBusy(true);
    setError("");
    setAnalysis(null);
    try {
      const body = await analyzeQuestion(question.trim());
      setAnalysis(body);
      setShowEvidence(false);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  const supported = analysis && (
    analysis.tier === "uncertain"
      ? false
      : analysis.tier === "revised"
        ? analysis.teacher_verification === "Supported"
        : analysis.student_verification === "Supported"
  );

  return (
    <main>
      <p className="disclaimer">
        Research prototype, not medical advice. Outputs may be wrong. Do not use this for medical decisions.
      </p>
      <section className="card">
        <h2>Ask a Healthcare Question</h2>
        <form onSubmit={submit}>
          <textarea
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) submit(event);
            }}
            maxLength={500}
            rows={4}
            placeholder="Type a healthcare question"
          />
          {example && (
            <button type="button" className="linkish" onClick={() => setQuestion(example)}>
              {example}
            </button>
          )}
          <button type="submit" className="primary" disabled={!question.trim() || busy || !health.ready}>
            {busy ? "Analyzing..." : "Analyze Answer"}
          </button>
        </form>
        {error && <p className="error-text">{error}</p>}
      </section>

      {busy && <p className="card">Loading analysis...</p>}

      {analysis && (
        <section className="grid">
          <article className="card">
            <h3>Student Response <span className="pill">FLAN-T5-small</span></h3>
            <p>{analysis.student_answer}</p>
          </article>
          <article className="card">
            <h3>Teacher Response <span className="pill">FLAN-T5-base</span></h3>
            <p>{analysis.teacher_answer || "The teacher model was not called because the student answer was Supported."}</p>
          </article>
          <article className="card">
            <h3>Final Response <span className={`pill tier-${analysis.tier}`}>{TIER_LABEL[analysis.tier]}</span></h3>
            <p>{analysis.final_answer}</p>
            <h4>Why this response?</h4>
            <ul>{analysis.reasons.map((reason) => <li key={reason}>{reason}</li>)}</ul>
          </article>
          <article className="card">
            <h3>
              Verification & Analysis
              <span className={`pill ${supported ? "ok" : "review"}`}>{supported ? "Supported" : "Needs Review"}</span>
            </h3>
            <Bar label="Student–teacher similarity" value={analysis.similarities.student_teacher} />
            <Bar label="Student–evidence similarity" value={analysis.similarities.student_evidence} />
            <Bar label="Teacher–evidence similarity" value={analysis.similarities.teacher_evidence} />
            <p className="footnote">Similarity scores measure text similarity, not factual correctness.</p>
            {analysis.evidence && (
              <div>
                <h4>Retrieved Evidence (Top result)</h4>
                <p className="muted">{analysis.evidence.source} · similarity {analysis.evidence.similarity.toFixed(3)}</p>
                <p className={showEvidence ? "" : "clamp"}>{analysis.evidence.text}</p>
                <button type="button" className="linkish" onClick={() => setShowEvidence((open) => !open)}>
                  {showEvidence ? "Show less" : "Show full evidence"}
                </button>
              </div>
            )}
          </article>
        </section>
      )}

      <section className="card">
        <h2>Research Results (Existing Experiment)</h2>
        {resultsError && <p className="error-text">{resultsError}</p>}
        {original && (
          <>
            <p className="muted">Original preliminary experiment, not reproducible. These are similarity-threshold results, not verified factual accuracy.</p>
            <div className="tiles">
              <Tile label="Questions Evaluated" value={String(original.n_questions)} />
              <Tile label="Detections Recorded" value={original.detector_flag_rate_count} />
              <Tile label="Similarity-Threshold Pass Rate" value={percent(original.similarity_pass_rate)} />
              <Tile label="PubMedQA pass rate" value={percent(original.pubmed_similarity_pass_rate)} />
              <Tile label="HealthCareMagic pass rate" value={percent(original.healthcaremagic_similarity_pass_rate)} />
            </div>
            <div className="row-actions">
              <Link to="/results">View Detailed Results →</Link>
              <button type="button" className="primary" onClick={() => setModal(true)}>View comparison graph</button>
            </div>
          </>
        )}
      </section>
      {modal && <ComparisonModal onClose={() => setModal(false)} />}
    </main>
  );
}

function Bar({ label, value }) {
  const width = value == null ? 0 : Math.max(0, Math.min(1, value)) * 100;
  return (
    <div className="bar-row">
      <span>{label}</span>
      <div className="bar"><span style={{ width: `${width}%` }} /></div>
      <span>{value == null ? "n/a" : value.toFixed(2)}</span>
    </div>
  );
}

function Tile({ label, value }) {
  return (
    <div className="tile">
      <strong>{value}</strong>
      <span>{label}</span>
    </div>
  );
}
