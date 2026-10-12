import React, { useEffect, useState } from "react";
import { NavLink, Route, Routes } from "react-router-dom";
import { getHealth } from "./api";
import { ShieldIcon } from "./components/Icons";
import { Badge } from "./components/ui";
import Home from "./pages/Home";
import About from "./pages/About";
import Research from "./pages/Research";

export default function App() {
  const [health, setHealth] = useState({ ready: false, error: null, unreachable: false });
  const [retryCount, setRetryCount] = useState(0);

  useEffect(() => {
    let stop = false;
    let timer = 0;
    async function poll() {
      try {
        const body = await getHealth();
        if (stop) return;
        setHealth({ ...body, unreachable: false });
        if (!body.ready && !body.error) timer = window.setTimeout(poll, 2000);
      } catch (err) {
        if (!stop) setHealth({ ready: false, error: err.message, unreachable: true });
      }
    }
    poll();
    return () => {
      stop = true;
      window.clearTimeout(timer);
    };
  }, [retryCount]);

  return (
    <div className="app-shell">
      <header className="site-header">
        <div className="container header-row">
          <div className="brand">
            <ShieldIcon />
            <div>
              <h1>HallucinationGuard</h1>
              <p>LLM Hallucination Detection & Mitigation for Healthcare QA</p>
            </div>
          </div>
          <Badge tone="prototype">Research Prototype</Badge>
        </div>
        <div className="container">
          <nav className="tabs" aria-label="Primary">
            <NavLink to="/" end>Home</NavLink>
            <NavLink to="/about">About</NavLink>
            <NavLink to="/results">Research Results</NavLink>
          </nav>
        </div>
      </header>
      <div className="disclaimer">
        <p className="container">Research prototype, not medical advice. Outputs may be wrong. Do not use this for medical decisions.</p>
      </div>
      <div className="container">
        {health.unreachable && (
          <div className="banner banner-danger" role="status">
            <span>Backend is unreachable.</span>
            <button type="button" className="btn btn-secondary" onClick={() => setRetryCount((count) => count + 1)}>Retry</button>
          </div>
        )}
        {!health.unreachable && !health.ready && !health.error && (
          <div className="banner" role="status">Models loading...</div>
        )}
        {health.error && !health.unreachable && (
          <div className="banner banner-danger" role="status">Backend error: {health.error}</div>
        )}
      </div>
      <Routes>
        <Route path="/" element={<Home health={health} />} />
        <Route path="/about" element={<About />} />
        <Route path="/results" element={<Research />} />
      </Routes>
      <footer className="site-footer">
        <p className="container">HallucinationGuard · Research prototype, not medical advice. Similarity scores measure text similarity, not factual correctness.</p>
      </footer>
    </div>
  );
}
