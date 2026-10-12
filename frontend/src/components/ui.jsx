import { figureUrl } from "../api";

export function Card({ title, children, className = "" }) {
  return (
    <section className={`card ${className}`.trim()}>
      {title && <h2 className="card-title">{title}</h2>}
      {children}
    </section>
  );
}

const BADGE_CLASS = {
  accepted: "badge badge-success",
  revised: "badge badge-primary",
  uncertain: "badge badge-warning",
  supported: "badge badge-success",
  review: "badge badge-review",
  model: "badge badge-neutral",
  prototype: "badge badge-primary",
};

export function Badge({ tone = "neutral", children }) {
  return <span className={BADGE_CLASS[tone] || "badge badge-neutral"}>{children}</span>;
}

export function StatTile({ icon, label, value }) {
  return (
    <div className="stat-tile">
      <span className="stat-icon" aria-hidden="true">{icon}</span>
      <span className="stat-value">{value}</span>
      <span className="stat-label">{label}</span>
    </div>
  );
}

export function SimilarityBar({ label, value }) {
  const missing = value == null || value === "";
  const width = missing ? 0 : Math.max(0, Math.min(1, Number(value))) * 100;
  return (
    <div className="sim-row">
      <span className="sim-label" id={`sim-${label}`}>{label}</span>
      <div className="sim-track" role="img" aria-labelledby={`sim-${label}`}>
        <span className="sim-fill" style={{ width: `${width}%` }} />
      </div>
      <span className="sim-value">{missing ? "Not computed" : Number(value).toFixed(2)}</span>
      {missing && <span className="sim-hint">This score was not produced for this answer.</span>}
    </div>
  );
}

export function ChartCard({ title, description, figure, children }) {
  return (
    <section className="card chart-card">
      <div className="card-header">
        <div>
          <h2 className="card-title">{title}</h2>
          {description && <p className="card-desc">{description}</p>}
        </div>
        {figure ? (
          <a className="btn btn-secondary" href={figureUrl(figure)} download={figure}>Download PNG</a>
        ) : null}
      </div>
      <div className="chart-body">{children}</div>
    </section>
  );
}
