import React from "react";

export default function About() {
  return (
    <main>
      <section className="card">
        <h2>Problem</h2>
        <p>
          A smaller healthcare model can produce an answer that sounds fluent but does not follow the retrieved evidence.
          This prototype compares that student answer with a larger teacher model and with the retrieved text, then either
          keeps the student answer, replaces it with a supported teacher answer, or says the evidence is insufficient.
        </p>
      </section>
      <section className="card">
        <h2>Architecture</h2>
        <p>
          The student is FLAN-T5-small and the teacher is FLAN-T5-base. Retrieval embeds the question with
          all-MiniLM-L6-v2 and keeps nearby knowledge-base rows. Cosine similarity is a text-overlap score.
          Configuration E checks whether that score against the retrieved context reaches the verify cutoff.
          The three tiers are accept, corrected, and uncertain. Student and teacher sentences are not blended,
          and a short answer is not replaced by copying a retrieved line.
        </p>
      </section>
      <section className="card">
        <h2>Datasets</h2>
        <p>
          PubMedQA, configuration pqa_labeled, and lavita/ChatDoctor-HealthCareMagic-100k.
          The static list in knowledge.py is unused.
        </p>
      </section>
      <section className="card">
        <h2>How to read the scores</h2>
        <p>
          A similarity pass means the answer’s MiniLM cosine similarity to the reference text is above 0.52.
          That is not a judgment that the answer is medically correct. The detector flag is a separate count:
          it marks low student–teacher agreement, low overlap with the retrieved context, or a very short student answer.
        </p>
      </section>
      <section className="card">
        <h2>Limitations</h2>
        <ul>
          <li>The comparison uses 20 validation and 20 test questions.</li>
          <li>Doctor replies in HealthCareMagic are not verified ground truth.</li>
          <li>The evidence check cannot label a contradiction.</li>
          <li>The verify cutoff was chosen using similarity to the reference on the validation split.</li>
          <li>This is a research prototype, not medical advice.</li>
        </ul>
      </section>
    </main>
  );
}
