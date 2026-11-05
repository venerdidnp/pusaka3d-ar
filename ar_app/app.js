// Pusaka3D AR Explainer -- Module 5.
// Loads assets/hotspots.json, injects <model-viewer> hotspot slots, and
// drives the info panel, language toggle, and a simple "quiz mode" that
// turns the hotspots into a tap-the-right-part game. Vanilla JS, no
// framework, no build step -- open index.html (served over http, not
// file://, so the module script and the JSON fetch both work) and it runs.

(function () {
  "use strict";

  const viewer = document.getElementById("viewer");
  const infoPanel = document.getElementById("info-panel");
  const infoTitle = document.getElementById("info-title");
  const infoText = document.getElementById("info-text");
  const infoClose = document.getElementById("info-close");
  const langToggle = document.getElementById("lang-toggle");
  const quizToggle = document.getElementById("quiz-toggle");
  const quizBanner = document.getElementById("quiz-banner");
  const quizTargetEl = document.getElementById("quiz-target");
  const quizScoreEl = document.getElementById("quiz-score");

  let lang = "en";          // "en" | "id"
  let quizMode = false;
  let hotspots = [];
  let quizTarget = null;
  let quizScore = { correct: 0, attempts: 0 };

  function t(hotspot, field) {
    return hotspot[`${field}_${lang}`];
  }

  function showInfo(hotspot) {
    infoTitle.textContent = t(hotspot, "name");
    infoText.textContent = t(hotspot, "text");
    infoPanel.hidden = false;
  }

  function hideInfo() {
    infoPanel.hidden = true;
  }

  function pickNewQuizTarget() {
    if (hotspots.length === 0) return;
    const others = hotspots.filter((h) => h !== quizTarget);
    quizTarget = others[Math.floor(Math.random() * others.length)] || hotspots[0];
    quizTargetEl.textContent = t(quizTarget, "name");
    quizScoreEl.textContent = `${quizScore.correct}/${quizScore.attempts}`;
  }

  function handleHotspotClick(hotspot) {
    if (!quizMode) {
      showInfo(hotspot);
      return;
    }
    quizScore.attempts += 1;
    if (hotspot.id === quizTarget.id) {
      quizScore.correct += 1;
      infoTitle.textContent = `Correct! / Benar!`;
      infoText.textContent = `${t(hotspot, "name")} -- ${t(hotspot, "text")}`;
    } else {
      infoTitle.textContent = `Not quite / Belum tepat`;
      infoText.textContent = lang === "en"
        ? `That was the ${t(hotspot, "name")}. Looking for the ${t(quizTarget, "name")}.`
        : `Itu adalah ${t(hotspot, "name")}. Yang dicari adalah ${t(quizTarget, "name")}.`;
    }
    infoPanel.hidden = false;
    pickNewQuizTarget();
  }

  function buildHotspotButton(hotspot) {
    const button = document.createElement("button");
    button.slot = `hotspot-${hotspot.id}`;
    button.className = "hotspot";
    button.dataset.position = hotspot.position.join(" ");
    button.dataset.normal = hotspot.normal.join(" ");
    button.textContent = "+";
    button.setAttribute("aria-label", hotspot.name_en);

    const label = document.createElement("div");
    label.className = "hotspot-annotation";
    label.slot = `hotspot-${hotspot.id}`; // model-viewer also supports a second slotted annotation
    label.dataset.position = hotspot.position.join(" ");
    label.dataset.normal = hotspot.normal.join(" ");
    label.textContent = hotspot.name_en;
    label.hidden = true; // shown only via CSS ::after in a richer build; kept simple here

    button.addEventListener("click", () => handleHotspotClick(hotspot));
    return button;
  }

  function loadHotspots() {
    fetch("assets/hotspots.json")
      .then((res) => res.json())
      .then((data) => {
        hotspots = data.hotspots;
        hotspots.forEach((hotspot) => {
          viewer.appendChild(buildHotspotButton(hotspot));
        });
        pickNewQuizTarget();
      })
      .catch((err) => {
        console.error("Could not load hotspots.json -- run make_temple_glb.py first.", err);
      });
  }

  langToggle.addEventListener("click", () => {
    lang = lang === "en" ? "id" : "en";
    langToggle.textContent = lang === "en" ? "EN / ID" : "ID / EN";
    if (!infoPanel.hidden && quizTarget) {
      // refresh any open panel text in the new language
      quizTargetEl.textContent = t(quizTarget, "name");
    }
  });

  quizToggle.addEventListener("click", () => {
    quizMode = !quizMode;
    quizToggle.textContent = quizMode ? "Quiz mode: on" : "Quiz mode: off";
    quizToggle.setAttribute("aria-pressed", String(quizMode));
    quizBanner.hidden = !quizMode;
    hideInfo();
    if (quizMode) {
      quizScore = { correct: 0, attempts: 0 };
      pickNewQuizTarget();
    }
  });

  infoClose.addEventListener("click", hideInfo);

  // This script is loaded at the end of <body> without `defer`, so the DOM
  // (including <model-viewer>) is already parsed by the time it runs --
  // call directly rather than waiting for an event that may already have fired.
  loadHotspots();
})();
