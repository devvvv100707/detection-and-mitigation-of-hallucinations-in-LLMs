import React, { useEffect, useState } from "react";
import { NavLink, Route, Routes } from "react-router-dom";
import { getHealth } from "./api";
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

  function retry() {
    setRetryCount((count) => count + 1);
  }

  return (
    <div className="page">
      <header className="top">
        <div className="brand">
          <span className="shield" aria-hidden="true">⛨</span>
          <div>
            <h1>HallucinationGuard</h1>
            <p>LLM Hallucination Detection & Mitigation for Healthcare QA</p>
          </div>
        </div>
        <span className="badge">Research Prototype</span>
      </header>
      <nav className="tabs">
        <NavLink to="/" end>Home</NavLink>
        <NavLink to="/about">About</NavLink>
        <NavLink to="/results">Research Results</NavLink>
      </nav>
      {health.unreachable && (
        <div className="banner error">
          Backend is unreachable.
          <button type="button" onClick={retry}>Retry</button>
        </div>
      )}
      {!health.unreachable && !health.ready && !health.error && (
        <div className="banner">Models loading...</div>
      )}
      {health.error && !health.unreachable && (
        <div className="banner error">Backend error: {health.error}</div>
      )}
      <Routes>
        <Route path="/" element={<Home health={health} />} />
        <Route path="/about" element={<About />} />
        <Route path="/results" element={<Research />} />
      </Routes>
    </div>
  );
}
