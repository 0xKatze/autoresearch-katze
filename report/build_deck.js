/* build_deck.js — N24 diffusion node-injection attack deck */
const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const fa = require("react-icons/fa");

// ---------- palette ----------
const NAVY = "21295C";
const TEAL = "1C7293";
const BLUE = "065A82";
const CRIMSON = "E63946";
const GREEN = "2A9D8F";
const GOLD = "E9C46A";
const LIGHT = "F4F6FB";
const WHITE = "FFFFFF";
const GREY = "8D99AE";
const INK = "1A1F3D";

const HEAD = "Trebuchet MS";
const BODY = "Calibri";

const FIG = (n) => `figures/${n}`;

// ---------- icon helper ----------
async function icon(IconComponent, color, size = 256) {
  const svg = ReactDOMServer.renderToStaticMarkup(
    React.createElement(IconComponent, { color: "#" + color, size: String(size) })
  );
  const png = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + png.toString("base64");
}

const shadow = () => ({ type: "outer", color: "000000", blur: 7, offset: 3, angle: 135, opacity: 0.18 });

(async () => {
  const pres = new pptxgen();
  pres.defineLayout({ name: "W", width: 13.333, height: 7.5 });
  pres.layout = "W";
  pres.author = "KATZE AutoResearch";
  pres.title = "Diffusion-Based Node Injection Attacks";

  const W = 13.333, H = 7.5;

  // pre-render icons
  const ic = {
    target: await icon(fa.FaBullseye, CRIMSON),
    shield: await icon(fa.FaShieldAlt, GREEN),
    lock: await icon(fa.FaLock, NAVY),
    eye: await icon(fa.FaEyeSlash, TEAL),
    sitemap: await icon(fa.FaProjectDiagram, TEAL),
    wave: await icon(fa.FaWater, BLUE),
    flask: await icon(fa.FaFlask, GOLD),
    bolt: await icon(fa.FaBolt, CRIMSON),
    check: await icon(fa.FaCheckCircle, GREEN),
    times: await icon(fa.FaTimesCircle, CRIMSON),
    layer: await icon(fa.FaLayerGroup, TEAL),
    chart: await icon(fa.FaChartLine, GOLD),
    bug: await icon(fa.FaBug, CRIMSON),
    cube: await icon(fa.FaCubes, TEAL),
    arrowUp: await icon(fa.FaArrowUp, GREEN),
    network: await icon(fa.FaNetworkWired, GOLD),
    flagW: await icon(fa.FaFlagCheckered, WHITE),
    lightbulb: await icon(fa.FaLightbulb, GOLD),
  };
  // White icon variants — used inside colored circles for contrast
  const icw = {
    eye: await icon(fa.FaEyeSlash, WHITE),
    lock: await icon(fa.FaLock, WHITE),
    network: await icon(fa.FaNetworkWired, WHITE),
    target: await icon(fa.FaBullseye, WHITE),
    cube: await icon(fa.FaCubes, WHITE),
    sitemap: await icon(fa.FaProjectDiagram, WHITE),
    layer: await icon(fa.FaLayerGroup, WHITE),
    flask: await icon(fa.FaFlask, WHITE),
    bolt: await icon(fa.FaBolt, WHITE),
    check: await icon(fa.FaCheckCircle, WHITE),
    bug: await icon(fa.FaBug, WHITE),
    shield: await icon(fa.FaShieldAlt, WHITE),
  };

  // ---------- shared helpers ----------
  function bgLight(s) { s.background = { color: LIGHT }; }
  function bgDark(s) { s.background = { color: NAVY }; }

  function header(s, kicker, title, accent = CRIMSON) {
    s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 0.28, h: H, fill: { color: accent } });
    s.addText(kicker.toUpperCase(), { x: 0.7, y: 0.42, w: 11.5, h: 0.35, fontFace: HEAD,
      fontSize: 13, bold: true, color: accent, charSpacing: 3, margin: 0 });
    s.addText(title, { x: 0.7, y: 0.74, w: 12.0, h: 0.85, fontFace: HEAD,
      fontSize: 30, bold: true, color: INK, margin: 0 });
  }

  function chip(s, x, y, w, h, iconData, title, body, accent) {
    s.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill: { color: WHITE }, shadow: shadow() });
    s.addShape(pres.shapes.RECTANGLE, { x, y, w: 0.09, h, fill: { color: accent } });
    s.addShape(pres.shapes.OVAL, { x: x + 0.28, y: y + 0.28, w: 0.62, h: 0.62, fill: { color: accent } });
    s.addImage({ data: iconData, x: x + 0.41, y: y + 0.41, w: 0.36, h: 0.36 });
    s.addText(title, { x: x + 1.08, y: y + 0.26, w: w - 1.25, h: 0.4, fontFace: HEAD,
      fontSize: 15, bold: true, color: INK, margin: 0, valign: "middle" });
    s.addText(body, { x: x + 1.08, y: y + 0.66, w: w - 1.25, h: h - 0.8, fontFace: BODY,
      fontSize: 11.5, color: "44485E", margin: 0, valign: "top" });
  }

  // ============================================================
  // SLIDE 1 — Title (dark)
  // ============================================================
  let s = pres.addSlide(); bgDark(s);
  // motif: faint nodes
  s.addShape(pres.shapes.OVAL, { x: 11.0, y: 0.7, w: 1.7, h: 1.7, fill: { color: TEAL, transparency: 60 } });
  s.addShape(pres.shapes.OVAL, { x: 12.0, y: 2.1, w: 0.9, h: 0.9, fill: { color: CRIMSON, transparency: 45 } });
  s.addShape(pres.shapes.OVAL, { x: 10.4, y: 2.3, w: 0.55, h: 0.55, fill: { color: GOLD, transparency: 40 } });
  s.addShape(pres.shapes.LINE, { x: 10.95, y: 2.55, w: 1.45, h: -0.3, line: { color: TEAL, width: 1.5, transparency: 40 } });
  s.addShape(pres.shapes.LINE, { x: 11.85, y: 1.55, w: 0.6, h: 1.0, line: { color: GOLD, width: 1.5, transparency: 40 } });

  s.addImage({ data: ic.target, x: 0.9, y: 1.5, w: 0.8, h: 0.8 });
  s.addText("ADVERSARIAL GRAPH ML  ·  EXPERIMENT REPORT", { x: 0.95, y: 2.5, w: 11, h: 0.4,
    fontFace: HEAD, fontSize: 14, bold: true, color: GOLD, charSpacing: 3, margin: 0 });
  s.addText("Diffusion-Based Node\nInjection Attacks", { x: 0.9, y: 2.95, w: 11.5, h: 1.9,
    fontFace: HEAD, fontSize: 50, bold: true, color: WHITE, lineSpacingMultiple: 0.95, margin: 0 });
  s.addText([
    { text: "Black-box · query-only · injection-only attacks on GNN graph classifiers", options: { breakLine: true, color: "CADCFC" } },
    { text: "PROTEINS dataset  ·  GCN / GIN / GAT / SAGE / median-pool victims", options: { color: GREY } },
  ], { x: 0.95, y: 4.95, w: 11, h: 0.9, fontFace: BODY, fontSize: 15, margin: 0, lineSpacingMultiple: 1.2 });

  // big stat strip
  const strip = [["98.43%", "PROTEINS GCN ASR", CRIMSON], ["+3.58 pp", "over baseline", GOLD], ["−69%", "variance", TEAL]];
  strip.forEach((d, i) => {
    const x = 0.95 + i * 3.0;
    s.addText(d[0], { x, y: 6.15, w: 2.8, h: 0.55, fontFace: HEAD, fontSize: 30, bold: true, color: d[2], margin: 0 });
    s.addText(d[1], { x, y: 6.72, w: 2.8, h: 0.4, fontFace: BODY, fontSize: 12, color: "CADCFC", margin: 0 });
  });

  // ============================================================
  // SLIDE 2 — Threat model
  // ============================================================
  s = pres.addSlide(); bgLight(s);
  header(s, "Setup", "The Threat Model", TEAL);
  const tm = [
    [icw.eye, "Black-box, query-only", "Attacker sees only output logits — never gradients or weights. All gradients are zeroth-order (CGE) estimates from queries.", TEAL],
    [icw.lock, "Injection-only", "Original graph is never mutated. Attacker appends new nodes with chosen features + edges. Construction is provably pure.", NAVY],
    [icw.network, "Edge budget = avg degree", "Each injected node connects to at most ⌈2|E|/|N|⌉ original nodes — a stealth constraint, not full connectivity.", GOLD],
    [icw.target, "Goal: flip the prediction", "Untargeted CW-margin objective. Success = the classifier changes its label for a previously-correct graph.", CRIMSON],
  ];
  tm.forEach((d, i) => {
    const col = i % 2, row = Math.floor(i / 2);
    chip(s, 0.7 + col * 6.15, 1.95 + row * 2.4, 5.85, 2.15, d[0], d[1], d[2], d[3]);
  });

  // ============================================================
  // SLIDE 3 — Mechanism journey
  // ============================================================
  s = pres.addSlide(); bgLight(s);
  header(s, "Core Finding", "Why Diffusion Failed 3×, Then Worked", CRIMSON);
  s.addImage({ path: FIG("chart_mechanism_evolution.png"), x: 0.7, y: 1.7, w: 7.5, h: 4.54, shadow: shadow() });
  // right rail explanation
  let rx = 8.55;
  s.addShape(pres.shapes.RECTANGLE, { x: rx, y: 1.7, w: 4.15, h: 4.54, fill: { color: WHITE }, shadow: shadow() });
  s.addShape(pres.shapes.RECTANGLE, { x: rx, y: 1.7, w: 4.15, h: 0.6, fill: { color: NAVY } });
  s.addText("THE INSIGHT", { x: rx + 0.25, y: 1.78, w: 3.5, h: 0.45, fontFace: HEAD, fontSize: 14, bold: true, color: WHITE, charSpacing: 2, margin: 0, valign: "middle" });
  s.addText([
    { text: "Pure feature-space diffusion is inert.", options: { bold: true, color: CRIMSON, breakLine: true, fontSize: 14 } },
    { text: "With a fixed full-connectivity topology, GCN normalization absorbs all feature variation — ~5% of graphs are topology-immune.", options: { color: "44485E", breakLine: true, fontSize: 12.5 } },
    { text: "", options: { breakLine: true, fontSize: 6 } },
    { text: "Make edges part of the diffusion state →", options: { bold: true, color: GREEN, breakLine: true, fontSize: 14 } },
    { text: "the trajectory gains a dimension that actually changes the prediction. Ceiling broken.", options: { color: "44485E", fontSize: 12.5 } },
  ], { x: rx + 0.25, y: 2.5, w: 3.7, h: 3.5, fontFace: BODY, margin: 0, lineSpacingMultiple: 1.05, valign: "top" });

  // ============================================================
  // SLIDE 4 — Tuning stack
  // ============================================================
  s = pres.addSlide(); bgLight(s);
  header(s, "Method", "The N24 Tuning Stack  (94.85% → 98.43%)", TEAL);
  s.addImage({ path: FIG("chart_asr_progression.png"), x: 0.7, y: 1.85, w: 7.4, h: 4.24, shadow: shadow() });
  const tiers = [
    [icw.cube, "Baseline", "5-restart Adam + zeroth-order CGE", NAVY],
    [icw.sitemap, "Joint X+A diffusion", "bounded DDIM + greedy edge search", TEAL],
    [icw.layer, "Budget ≤ k", "edge budget as upper bound", BLUE],
    [icw.flask, "k_sweep", "retry stuck graphs at other k", GOLD],
    [icw.bolt, "Per-node escalation", "m=2 with independent edge masks", CRIMSON],
  ];
  tiers.forEach((d, i) => {
    const y = 1.8 + i * 0.92;
    s.addShape(pres.shapes.RECTANGLE, { x: 8.5, y, w: 4.2, h: 0.8, fill: { color: WHITE }, shadow: shadow() });
    s.addShape(pres.shapes.OVAL, { x: 8.65, y: y + 0.16, w: 0.48, h: 0.48, fill: { color: d[3] } });
    s.addImage({ data: d[0], x: 8.75, y: y + 0.26, w: 0.28, h: 0.28 });
    s.addText(`${i + 1}. ${d[1]}`, { x: 9.25, y: y + 0.08, w: 3.4, h: 0.36, fontFace: HEAD, fontSize: 12.5, bold: true, color: INK, margin: 0, valign: "middle" });
    s.addText(d[2], { x: 9.25, y: y + 0.42, w: 3.4, h: 0.32, fontFace: BODY, fontSize: 10, color: "5A5F75", margin: 0, valign: "middle" });
  });

  // ============================================================
  // SLIDE 5 — Contribution per tier
  // ============================================================
  s = pres.addSlide(); bgLight(s);
  header(s, "Analysis", "Where Each Tier Earns Its Keep", GOLD);
  s.addImage({ path: FIG("chart_diffusion_contribution.png"), x: 0.7, y: 1.75, w: 8.0, h: 4.41, shadow: shadow() });
  let cx = 9.0;
  const facts = [
    [icw.check, "Folds 0 & 4 fully solved by the baseline alone — diffusion never fires.", GREEN],
    [icw.bolt, "Folds 1–2 are where joint diffusion + escalation rescue 30 graphs Adam missed.", CRIMSON],
    [icw.bug, "13 graphs remain stuck — small graphs where one node can't move the readout.", GOLD],
  ];
  facts.forEach((d, i) => {
    const y = 1.95 + i * 1.45;
    s.addShape(pres.shapes.RECTANGLE, { x: cx, y, w: 3.7, h: 1.25, fill: { color: WHITE }, shadow: shadow() });
    s.addShape(pres.shapes.OVAL, { x: cx + 0.22, y: y + 0.34, w: 0.55, h: 0.55, fill: { color: d[2] } });
    s.addImage({ data: d[0], x: cx + 0.33, y: y + 0.45, w: 0.33, h: 0.33 });
    s.addText(d[1], { x: cx + 0.95, y: y + 0.18, w: 2.6, h: 0.9, fontFace: BODY, fontSize: 11.5, color: "44485E", margin: 0, valign: "middle" });
  });

  // ============================================================
  // SLIDE 6 — Robustness sweep
  // ============================================================
  s = pres.addSlide(); bgLight(s);
  header(s, "Generalization", "What Actually Defends?", GREEN);
  s.addImage({ path: FIG("chart_robustness.png"), x: 0.7, y: 1.7, w: 7.9, h: 4.47, shadow: shadow() });
  s.addShape(pres.shapes.RECTANGLE, { x: 8.85, y: 1.85, w: 3.85, h: 2.0, fill: { color: CRIMSON } });
  s.addImage({ data: icw.target, x: 9.1, y: 2.1, w: 0.5, h: 0.5 });
  s.addText("Conv type ≠ defense", { x: 9.7, y: 2.12, w: 2.85, h: 0.5, fontFace: HEAD, fontSize: 15, bold: true, color: WHITE, margin: 0, valign: "middle" });
  s.addText("GIN, GAT, SAGE all use mean-pool and are MORE vulnerable than GCN — 100% ASR with diffusion never firing.", { x: 9.1, y: 2.7, w: 3.45, h: 1.05, fontFace: BODY, fontSize: 12, color: "FFE3E5", margin: 0, valign: "top" });
  s.addShape(pres.shapes.RECTANGLE, { x: 8.85, y: 4.05, w: 3.85, h: 2.05, fill: { color: GREEN } });
  // shield on white circle for contrast against green box
  s.addShape(pres.shapes.OVAL, { x: 9.05, y: 4.28, w: 0.56, h: 0.56, fill: { color: WHITE } });
  s.addImage({ data: ic.shield, x: 9.14, y: 4.37, w: 0.38, h: 0.38 });
  s.addText("Defend the READOUT", { x: 9.72, y: 4.32, w: 2.85, h: 0.5, fontFace: HEAD, fontSize: 15, bold: true, color: WHITE, margin: 0, valign: "middle" });
  s.addText("Only median-pool resists (92.46%). The pooling — not the aggregator — is the true attack surface.", { x: 9.1, y: 4.92, w: 3.45, h: 1.05, fontFace: BODY, fontSize: 12, color: "E2F5F1", margin: 0, valign: "top" });

  // ============================================================
  // SLIDE 7 — Defeating the defense
  // ============================================================
  s = pres.addSlide(); bgLight(s);
  header(s, "Stress Test", "Defeating the Median-Pool Defense", BLUE);
  s.addImage({ path: FIG("chart_defense_breakdown.png"), x: 0.7, y: 1.7, w: 7.6, h: 4.6, shadow: shadow() });
  s.addShape(pres.shapes.RECTANGLE, { x: 8.65, y: 1.9, w: 4.05, h: 4.3, fill: { color: WHITE }, shadow: shadow() });
  s.addShape(pres.shapes.RECTANGLE, { x: 8.65, y: 1.9, w: 4.05, h: 0.6, fill: { color: BLUE } });
  s.addText("WHEN DIFFUSION MATTERS", { x: 8.85, y: 1.98, w: 3.7, h: 0.45, fontFace: HEAD, fontSize: 12.5, bold: true, color: WHITE, charSpacing: 1, margin: 0, valign: "middle" });
  s.addText([
    { text: "63%", options: { fontSize: 30, bold: true, color: GREY, breakLine: true } },
    { text: "baseline-solo ASR on the hardest fold once median-pool defends", options: { fontSize: 11.5, color: "5A5F75", breakLine: true } },
    { text: "", options: { fontSize: 8, breakLine: true } },
    { text: "92.46%", options: { fontSize: 30, bold: true, color: GREEN, breakLine: true } },
    { text: "recovered by joint X+A diffusion + k_sweep", options: { fontSize: 11.5, color: "5A5F75", breakLine: true } },
    { text: "", options: { fontSize: 8, breakLine: true } },
    { text: "The diffusion mechanism earns its keep precisely against defenses — on undefended models a simple baseline suffices.", options: { fontSize: 12, italic: true, color: NAVY } },
  ], { x: 8.85, y: 2.65, w: 3.65, h: 3.4, fontFace: BODY, margin: 0, lineSpacingMultiple: 1.0, valign: "top" });

  // ============================================================
  // SLIDE 8 — Takeaways (dark)
  // ============================================================
  s = pres.addSlide(); bgDark(s);
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 0.28, h: H, fill: { color: GOLD } });
  s.addImage({ data: ic.lightbulb, x: 0.85, y: 0.7, w: 0.7, h: 0.7 });
  s.addText("KEY TAKEAWAYS", { x: 0.85, y: 1.5, w: 11, h: 0.4, fontFace: HEAD, fontSize: 14, bold: true, color: GOLD, charSpacing: 3, margin: 0 });
  s.addText("Three Lessons", { x: 0.85, y: 1.88, w: 11, h: 0.8, fontFace: HEAD, fontSize: 34, bold: true, color: WHITE, margin: 0 });

  const take = [
    [icw.sitemap, "Topology must be in the diffusion state", "Feature-only graph diffusion is inert — it never beats a trivial injection. Edges must be an optimization variable.", TEAL],
    [icw.shield, "The readout is the attack surface", "Changing the convolution (GIN/GAT/SAGE) is not a defense. Defend the pooling: median or trimmed-mean.", GREEN],
    [icw.bolt, "Diffusion proves itself against defenses", "On undefended models a baseline suffices; against median-pool, diffusion is the difference between 63% and 92%.", CRIMSON],
  ];
  take.forEach((d, i) => {
    const y = 2.95 + i * 1.42;
    s.addShape(pres.shapes.RECTANGLE, { x: 0.85, y, w: 11.6, h: 1.25, fill: { color: "2C3666" } });
    s.addShape(pres.shapes.RECTANGLE, { x: 0.85, y, w: 0.1, h: 1.25, fill: { color: d[3] } });
    s.addShape(pres.shapes.OVAL, { x: 1.15, y: y + 0.34, w: 0.58, h: 0.58, fill: { color: d[3] } });
    s.addImage({ data: d[0], x: 1.27, y: y + 0.46, w: 0.34, h: 0.34 });
    s.addText(d[1], { x: 2.0, y: y + 0.18, w: 10.2, h: 0.45, fontFace: HEAD, fontSize: 17, bold: true, color: WHITE, margin: 0, valign: "middle" });
    s.addText(d[2], { x: 2.0, y: y + 0.62, w: 10.2, h: 0.55, fontFace: BODY, fontSize: 12.5, color: "CADCFC", margin: 0, valign: "middle" });
  });

  await pres.writeFile({ fileName: "N24_diffusion_attack_deck.pptx" });
  console.log("Deck written: N24_diffusion_attack_deck.pptx");
})();
