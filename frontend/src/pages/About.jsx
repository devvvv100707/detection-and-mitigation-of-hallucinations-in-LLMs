import React, { Fragment } from "react";
import { Card } from "../components/ui";

const STEPS = ["Question", "Retrieval", "Student / Teacher", "Verification", "Three-tier mitigation", "Final answer"];

export default function About() {
  return (
    <main className="container stack">
      <h2 className="section-title">About</h2>
      <Card title="Problem">
        <p>A smaller healthcare model can produce a fluent answer that does not follow the retrieved evidence. This prototype compares that student answer with a larger teacher model and with the retrieved text, then keeps it, replaces it, or says the evidence is insufficient.</p>
      </Card>
      <Card title="Architecture">
        <div className="flow" aria-label="Pipeline diagram">
          {STEPS.map((step, index) => (
            <Fragment key={step}>
              <div className="flow-box">{step}</div>
              {index < STEPS.length - 1 && <span className="arrow" aria-hidden="true">→</span>}
            </Fragment>
          ))}
        </div>
        <p>The student is FLAN-T5-small and the teacher is FLAN-T5-base. Retrieval uses all-MiniLM-L6-v2 cosine similarity. Configuration E checks whether the answer is similar enough to the retrieved context. The tiers are accept, corrected, and uncertain. Answers are not blended, and retrieved lines are not copied into the response.</p>
      </Card>
      <Card title="Datasets">
        <p>PubMedQA, configuration pqa_labeled, and lavita/ChatDoctor-HealthCareMagic-100k. The static list in knowledge.py is unused.</p>
      </Card>
      <Card title="How to read the scores">
        <p>A similarity pass means MiniLM cosine similarity to the reference text is above 0.52. That is not a judgment that the answer is medically correct. The detector flag is a separate count for low student–teacher agreement, low overlap with retrieved context, or a very short student answer.</p>
      </Card>
      <Card title="Limitations">
        <ul>
          <li>The comparison uses 20 validation and 20 test questions.</li>
          <li>Doctor replies in HealthCareMagic are not verified ground truth.</li>
          <li>The evidence check cannot label a contradiction.</li>
          <li>The verify cutoff was chosen using similarity to the reference on the validation split.</li>
          <li>This is a research prototype, not medical advice.</li>
        </ul>
      </Card>
    </main>
  );
}
