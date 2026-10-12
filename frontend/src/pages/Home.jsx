import React, { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { analyzeQuestion, getResults } from "../api";
import ComparisonModal from "../ComparisonModal";
import { CheckIcon, DatabaseIcon, DocumentIcon, SearchIcon, WarningIcon } from "../components/Icons";
import { Badge, Card, SimilarityBar, StatTile } from "../components/ui";

const TIER_LABEL = {
  accepted: "Accepted answer",
  revised: "Revised Answer",
  uncertain: "Uncertainty Warning",
};
const TIER_TONE = { accepted: "accepted", revised: "revised", uncertain: "uncertain" };

function percent(value) {
  return `${Math.round(value * 100)}%`;
}

function formatSeconds(value) {
  return `${Number(value).toFixed(2)}s`;
}

export default function Home({ health }) {
  const [question, setQuestion] = useState("");
  const [examples, setExamples] = useState([]);
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
        setExamples(body.example_question ? [body.example_question] : []);
      })
      .catch((err) => setResultsError(err.message));
  }, []);

  async function submit(event) {
    event?.preventDefault();
    if (!question.trim() || busy || !health.ready) return;
    setBusy(true);
    setError("");
    try {
      const body = await analyzeQuestion(question.trim());
      setAnalysis(body);
      setShowEvidence(false);
    } catch (err) {
      setAnalysis(null);
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  const disabledReason = !question.trim()
    ? "Enter a question to enable analysis."
    : !health.ready
      ? "Analysis is disabled until the models are ready."
      : busy
        ? "Analysis is running."
        : "";
  const supported = analysis && (
    analysis.tier === "uncertain"
      ? false
      : analysis.tier === "revised"
        ? analysis.teacher_verification === "Supported"
        : analysis.student_verification === "Supported"
  );

  return (
    <main className="container stack">
      <Card>
        <h2 className="section-title">Ask a Healthcare Question</h2>
        <p className="card-desc">The student and teacher models answer from retrieved evidence, then configuration E keeps, revises, or withholds the answer.</p>
        <form onSubmit={submit}>
          <div className="ask-row">
            <label className="visually-hidden" htmlFor="question">Healthcare question</label>
            <textarea
              id="question"
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) submit(event);
              }}
              maxLength={500}
              rows={2}
              placeholder="Type a healthcare question"
            />
            <button type="submit" className="btn btn-primary" disabled={Boolean(disabledReason)}>
              {busy ? "Analyzing..." : "Analyze Answer"}
            </button>
          </div>
          <p className="counter" aria-live="polite">{question.length}/500 {disabledReason}</p>
          {examples.length > 0 && (
            <div className="chips">
              <span className="muted">Try an example:</span>
              {examples.map((item) => (
                <button key={item} type="button" className="chip" onClick={() => setQuestion(item)}>{item}</button>
              ))}
            </div>
          )}
        </form>
      </Card>

      <div aria-live="polite">
        {!analysis && !busy && !error && (
          <Card>
            <h2 className="section-title">Analysis Results</h2>
            <p>Ask a question to see the student response, the teacher response when it is called, the final tier, evidence similarity, and the retrieved passage.</p>
          </Card>
        )}
        {busy && (
          <Card>
            <h2 className="section-title"><span className="spinner" aria-hidden="true" /> Loading analysis...</h2>
            <div className="skeleton" />
            <div className="skeleton" />
            <div className="skeleton" />
          </Card>
        )}
        {error && (
          <Card>
            <h2 className="section-title">Analysis failed</h2>
            <p>{error}</p>
            <button type="button" className="btn btn-secondary" onClick={() => submit()}>Retry</button>
          </Card>
        )}
        {analysis && (
          <>
            <div className="results-grid">
              <Card>
                <h2 className="card-title">Student Response <Badge tone="model">FLAN-T5-small</Badge></h2>
                <p className="answer-text">{analysis.student_answer}</p>
              </Card>
              <Card>
                <h2 className="card-title">Teacher Response <Badge tone="model">FLAN-T5-base</Badge></h2>
                <p className="answer-text">{analysis.teacher_answer == null ? "Not computed" : analysis.teacher_answer}</p>
                {analysis.teacher_answer == null && <p className="muted">The teacher model was not called for this question.</p>}
              </Card>
              <Card>
                <h2 className="card-title">Final Response <Badge tone={TIER_TONE[analysis.tier]}>{TIER_LABEL[analysis.tier]}</Badge></h2>
                <p className="answer-text">{analysis.final_answer}</p>
                <h3 className="card-title">Why this response?</h3>
                <ul className="why">{analysis.reasons.map((reason) => <li key={reason}>{reason}</li>)}</ul>
              </Card>
              <Card>
                <h2 className="card-title">
                  Verification & Analysis
                  <Badge tone={supported ? "supported" : "review"}>{supported ? "Supported" : "Needs Review"}</Badge>
                </h2>
                <p className="muted">Supported means the answer text is similar to the retrieved evidence. It does not confirm medical correctness.</p>
                <SimilarityBar label="Student–teacher similarity" value={analysis.similarities.student_teacher} />
                <SimilarityBar label="Student–evidence similarity" value={analysis.similarities.student_evidence} />
                <SimilarityBar label="Teacher–evidence similarity" value={analysis.similarities.teacher_evidence} />
                <p className="footnote muted">Similarity scores measure text similarity, not factual correctness.</p>
                {analysis.evidence && (
                  <div>
                    <h3 className="card-title">Retrieved Evidence (Top result)</h3>
                    <p className="muted"><Badge tone="model">{analysis.evidence.source}</Badge> similarity {analysis.evidence.similarity.toFixed(3)}</p>
                    <p className={showEvidence ? "evidence open answer-text" : "evidence answer-text"}>{analysis.evidence.text}</p>
                    <button type="button" className="btn btn-ghost" onClick={() => setShowEvidence((open) => !open)}>
                      {showEvidence ? "Show less" : "Show full evidence"}
                    </button>
                  </div>
                )}
              </Card>
            </div>
            {analysis.latency && (
              <p className="meta">
                {Object.entries(analysis.latency).map(([name, seconds]) => (
                  <span key={name}>{name.replaceAll("_", " ")} {formatSeconds(seconds)}</span>
                ))}
              </p>
            )}
          </>
        )}
      </div>

      <Card>
        <h2 className="section-title">Research Results (Existing Experiment)</h2>
        {resultsError && <p className="banner banner-danger">{resultsError}</p>}
        {original && (
          <>
            <p className="muted">Original preliminary experiment, not reproducible. These are similarity-threshold results, not independently verified factual accuracy.</p>
            <div className="stat-row">
              <StatTile icon={<DocumentIcon />} label="Questions Evaluated" value={String(original.n_questions)} />
              <StatTile icon={<WarningIcon />} label="Detections Recorded" value={original.detector_flag_rate_count} />
              <StatTile icon={<CheckIcon />} label="Similarity-Threshold Pass Rate" value={percent(original.similarity_pass_rate)} />
              <StatTile icon={<DatabaseIcon />} label="PubMedQA pass rate" value={percent(original.pubmed_similarity_pass_rate)} />
              <StatTile icon={<SearchIcon />} label="HealthCareMagic pass rate" value={percent(original.healthcaremagic_similarity_pass_rate)} />
            </div>
            <div className="row-actions">
              <Link className="btn btn-ghost" to="/results">View Detailed Results →</Link>
              <button type="button" className="btn btn-primary" onClick={() => setModal(true)}>View comparison graph</button>
            </div>
          </>
        )}
      </Card>
      {modal && <ComparisonModal onClose={() => setModal(false)} />}
    </main>
  );
}
