/* build_arch_deck.js — Diffusion-KATZE ARCHITECTURE deep-dive deck */
const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const fa = require("react-icons/fa");

// ---------- palette (matches build_deck.js) ----------
const NAVY = "21295C", TEAL = "1C7293", BLUE = "065A82", CRIMSON = "E63946";
const GREEN = "2A9D8F", GOLD = "E9C46A", LIGHT = "F4F6FB", WHITE = "FFFFFF";
const GREY = "8D99AE", INK = "1A1F3D", MUT = "44485E";
const HEAD = "Trebuchet MS", BODY = "Calibri", MONO = "Consolas";
const FIG = (n) => `figures/${n}`;

async function icon(IconComponent, color, size = 256) {
  const svg = ReactDOMServer.renderToStaticMarkup(
    React.createElement(IconComponent, { color: "#" + color, size: String(size) }));
  const png = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + png.toString("base64");
}
const shadow = () => ({ type: "outer", color: "000000", blur: 7, offset: 3, angle: 135, opacity: 0.18 });

(async () => {
  const pres = new pptxgen();
  pres.defineLayout({ name: "W", width: 13.333, height: 7.5 });
  pres.layout = "W";
  pres.author = "KATZE AutoResearch";
  pres.title = "Diffusion-KATZE — Architecture";
  const W = 13.333, H = 7.5;

  const ic = {
    target: await icon(fa.FaBullseye, CRIMSON), shield: await icon(fa.FaShieldAlt, GREEN),
    lightbulb: await icon(fa.FaLightbulb, GOLD), wave: await icon(fa.FaWaveSquare, BLUE),
    scale: await icon(fa.FaBalanceScale, TEAL),
  };
  const icw = {
    eye: await icon(fa.FaEyeSlash, WHITE), lock: await icon(fa.FaLock, WHITE),
    network: await icon(fa.FaNetworkWired, WHITE), target: await icon(fa.FaBullseye, WHITE),
    cube: await icon(fa.FaCubes, WHITE), sitemap: await icon(fa.FaProjectDiagram, WHITE),
    layer: await icon(fa.FaLayerGroup, WHITE), flask: await icon(fa.FaFlask, WHITE),
    bolt: await icon(fa.FaBolt, WHITE), check: await icon(fa.FaCheckCircle, WHITE),
    times: await icon(fa.FaTimesCircle, WHITE), bug: await icon(fa.FaBug, WHITE),
    wave: await icon(fa.FaWaveSquare, WHITE), random: await icon(fa.FaRandom, WHITE),
    cogs: await icon(fa.FaCogs, WHITE), sliders: await icon(fa.FaSlidersH, WHITE),
    exchange: await icon(fa.FaExchangeAlt, WHITE), scale: await icon(fa.FaBalanceScale, WHITE),
    dice: await icon(fa.FaDiceD6, WHITE), chart: await icon(fa.FaChartLine, WHITE),
    shield: await icon(fa.FaShieldAlt, WHITE), lightbulb: await icon(fa.FaLightbulb, WHITE),
  };

  const bgLight = (s) => { s.background = { color: LIGHT }; };
  const bgDark = (s) => { s.background = { color: NAVY }; };

  function header(s, kicker, title, accent = CRIMSON) {
    s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 0.28, h: H, fill: { color: accent } });
    s.addText(kicker.toUpperCase(), { x: 0.7, y: 0.42, w: 11.5, h: 0.35, fontFace: HEAD,
      fontSize: 13, bold: true, color: accent, charSpacing: 3, margin: 0 });
    s.addText(title, { x: 0.7, y: 0.74, w: 12.2, h: 0.85, fontFace: HEAD,
      fontSize: 29, bold: true, color: INK, margin: 0 });
  }
  function chip(s, x, y, w, h, iconData, title, body, accent) {
    s.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill: { color: WHITE }, shadow: shadow() });
    s.addShape(pres.shapes.RECTANGLE, { x, y, w: 0.09, h, fill: { color: accent } });
    const cd = 0.62, cy = y + (h - cd) / 2;
    s.addShape(pres.shapes.OVAL, { x: x + 0.28, y: cy, w: cd, h: cd, fill: { color: accent } });
    s.addImage({ data: iconData, x: x + 0.41, y: cy + 0.13, w: 0.36, h: 0.36 });
    s.addText(title, { x: x + 1.08, y: y + 0.22, w: w - 1.3, h: 0.4, fontFace: HEAD,
      fontSize: 15, bold: true, color: INK, margin: 0, valign: "middle" });
    s.addText(body, { x: x + 1.08, y: y + 0.62, w: w - 1.3, h: h - 0.82, fontFace: BODY,
      fontSize: 11.5, color: MUT, margin: 0, valign: "top", lineSpacingMultiple: 1.02 });
  }
  // flow box with optional title/sub — auto dark text on light (gold/white) fills
  function fbox(s, x, y, w, h, title, sub, fill) {
    const lightFill = (fill === GOLD || fill === LIGHT || fill === WHITE);
    const fg = lightFill ? INK : WHITE;
    const subc = lightFill ? "6B5A22" : "DCE3F5";
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08,
      fill: { color: fill }, shadow: shadow() });
    s.addText([
      { text: title, options: { fontFace: HEAD, fontSize: 12.5, bold: true, color: fg, breakLine: true } },
      ...(sub ? [{ text: sub, options: { fontFace: BODY, fontSize: 9.5, color: subc } }] : []),
    ], { x: x + 0.08, y, w: w - 0.16, h, align: "center", valign: "middle", margin: 0, lineSpacingMultiple: 0.98 });
  }
  function arrow(s, x1, y1, x2, y2, color = GREY) {
    s.addShape(pres.shapes.LINE, { x: x1, y: y1, w: x2 - x1, h: y2 - y1,
      line: { color, width: 2.25, endArrowType: "triangle" } });
  }

  // ============================================================ S1 Title
  let s = pres.addSlide(); bgDark(s);
  s.addShape(pres.shapes.LINE, { x: 11.85, y: 1.55, w: 0.6, h: 1.0, line: { color: TEAL, width: 1.5, transparency: 55 } });
  s.addShape(pres.shapes.LINE, { x: 10.95, y: 2.55, w: 0.95, h: -0.45, line: { color: GOLD, width: 1.5, transparency: 55 } });
  s.addShape(pres.shapes.OVAL, { x: 11.0, y: 0.7, w: 1.7, h: 1.7, fill: { color: TEAL, transparency: 60 } });
  s.addShape(pres.shapes.OVAL, { x: 12.0, y: 2.1, w: 0.9, h: 0.9, fill: { color: CRIMSON, transparency: 45 } });
  s.addShape(pres.shapes.OVAL, { x: 10.4, y: 2.3, w: 0.55, h: 0.55, fill: { color: GOLD, transparency: 40 } });
  s.addImage({ data: ic.wave, x: 0.9, y: 1.45, w: 0.85, h: 0.85 });
  s.addText("ADVERSARIAL GRAPH ML  ·  ARCHITECTURE DEEP-DIVE", { x: 0.95, y: 2.5, w: 11, h: 0.4,
    fontFace: HEAD, fontSize: 14, bold: true, color: GOLD, charSpacing: 3, margin: 0 });
  s.addText("Diffusion-KATZE", { x: 0.9, y: 2.92, w: 11.5, h: 1.1,
    fontFace: HEAD, fontSize: 54, bold: true, color: WHITE, margin: 0 });
  s.addText("How a discrete-time diffusion process injects adversarial nodes into a graph", {
    x: 0.95, y: 4.15, w: 11, h: 0.5, fontFace: HEAD, fontSize: 19, color: "CADCFC", margin: 0 });
  s.addText([
    { text: "Black-box · query-only · injection-only attack on GNN graph classifiers", options: { breakLine: true, color: "CADCFC" } },
    { text: "The core idea: put edge topology — not just features — into the diffusion state", options: { italic: true, color: "A9B4D6" } },
  ], { x: 0.95, y: 4.75, w: 11.4, h: 0.9, fontFace: BODY, fontSize: 15, margin: 0, lineSpacingMultiple: 1.2 });
  const strip = [["98.43%", "PROTEINS GCN ASR", CRIMSON], ["X + A", "joint diffusion state", TEAL], ["1 node", "injection budget", GOLD]];
  strip.forEach((d, i) => {
    const x = 0.95 + i * 3.0;
    s.addText(d[0], { x, y: 6.15, w: 2.8, h: 0.55, fontFace: HEAD, fontSize: 30, bold: true, color: d[2], margin: 0 });
    s.addText(d[1], { x, y: 6.72, w: 2.8, h: 0.4, fontFace: BODY, fontSize: 12, color: "CADCFC", margin: 0 });
  });

  // ============================================================ S2 Threat model
  s = pres.addSlide(); bgLight(s);
  header(s, "Setup", "Threat Model — What the Attacker May Do", TEAL);
  const tm = [
    [icw.eye, "Black-box, query-only", "Sees only output logits — never gradients or weights. Every gradient is a zeroth-order (CGE) estimate from queries.", TEAL],
    [icw.lock, "Injection-only", "The original graph is never mutated. The attacker appends new nodes with chosen features + edges; construction is a pure function.", NAVY],
    [icw.network, "Tiny budget", "node_budget = 1 injected node; edges per node capped (1 ≤ k ≤ avg-degree). A stealth constraint, not full connectivity.", GOLD],
    [icw.target, "Goal: flip the label", "Untargeted CW-margin objective. Success = the classifier changes its prediction on a previously-correct graph.", CRIMSON],
  ];
  tm.forEach((d, i) => {
    const col = i % 2, row = Math.floor(i / 2);
    chip(s, 0.7 + col * 6.15, 2.35 + row * 2.1, 5.85, 1.85, d[0], d[1], d[2], d[3]);
  });

  // ============================================================ S3 End-to-end pipeline
  s = pres.addSlide(); bgLight(s);
  header(s, "Architecture", "End-to-End Pipeline — A Cascade of Fallbacks", NAVY);
  s.addText("Each stage fires only when the previous one fails. Cheap methods first; the diffusion machinery is the last resort for the hard graphs.",
    { x: 0.7, y: 1.62, w: 12.0, h: 0.4, fontFace: BODY, fontSize: 13, italic: true, color: MUT, margin: 0 });
  // row 1
  const by1 = 2.35, bh = 0.95, bw = 2.55;
  const r1 = [
    ["Input graph G", "correctly classified", BLUE],
    ["Select 1 node\n+ edge targets", "topk-deg / full", TEAL],
    ["Adversarial\ngenerator Gθ", "feature proposal", NAVY],
    ["ZO-CGE\ngradient", "query-only ∇", GREEN],
  ];
  r1.forEach((d, i) => {
    const x = 0.7 + i * 3.12;
    fbox(s, x, by1, bw, bh, d[0], d[1], d[2]);
    if (i < 3) arrow(s, x + bw + 0.04, by1 + bh / 2, x + 3.12 - 0.02, by1 + bh / 2, GREY);
  });
  // down connector
  arrow(s, 0.7 + 3 * 3.12 + bw / 2, by1 + bh + 0.02, 0.7 + 3 * 3.12 + bw / 2, by1 + bh + 0.42, GREY);
  // row 2 (right-to-left visual flow back)
  const by2 = by1 + bh + 0.55;
  const r2 = [
    ["5-restart Adam\nbaseline", "94.85% solo", NAVY],
    ["Diffusion restart\n(DDIM on x)", "feature retry", BLUE],
    ["Joint X+A diffusion\n+ k_sweep", "edges in state", CRIMSON],
    ["Per-node\nescalation", "m=2 masks", GOLD],
  ];
  r2.forEach((d, i) => {
    const x = 0.7 + i * 3.12;
    fbox(s, x, by2, bw, bh, d[0], d[1], d[2]);
    if (i < 3) arrow(s, x + bw + 0.04, by2 + bh / 2, x + 3.12 - 0.02, by2 + bh / 2, GREY);
  });
  arrow(s, 0.7 + 3 * 3.12 + bw / 2, by2 + bh + 0.02, 0.7 + 3 * 3.12 + bw / 2, by2 + bh + 0.42, GREEN);
  fbox(s, 0.7 + 3 * 3.12, by2 + bh + 0.52, bw, 0.72, "Perturbed graph G′", "label flipped ✓", GREEN);
  // balance note (bottom-left)
  const ny = by2 + bh + 0.52;
  s.addShape(pres.shapes.RECTANGLE, { x: 0.7, y: ny, w: 8.6, h: 0.72, fill: { color: WHITE }, shadow: shadow() });
  s.addShape(pres.shapes.RECTANGLE, { x: 0.7, y: ny, w: 0.09, h: 0.72, fill: { color: NAVY } });
  s.addText([
    { text: "Economics of the cascade:  ", options: { bold: true, color: INK } },
    { text: "the 5-restart baseline alone flips ~95% of graphs; the diffusion core + escalation exist only to rescue the few topology-immune ones the optimizer misses.", options: { color: MUT } },
  ], { x: 1.0, y: ny, w: 8.1, h: 0.72, fontFace: BODY, fontSize: 11.5, valign: "middle", margin: 0, lineSpacingMultiple: 1.0 });
  // legend
  s.addText("Blue = data · Navy = generator / optimizer · Crimson = diffusion core · Gold = escalation · Green = success",
    { x: 0.7, y: 6.95, w: 12, h: 0.35, fontFace: BODY, fontSize: 10.5, color: GREY, italic: true, margin: 0 });

  // ============================================================ S4 Generator + ZO gradient
  s = pres.addSlide(); bgLight(s);
  header(s, "Component A", "Generator + Zeroth-Order Gradient", GREEN);
  const ca = [
    [icw.cube, "Adversarial generator Gθ", "A small MLP maps noise → feature vector for the injected node. 5 random restarts (different lr / epochs) escape bad local optima.", NAVY],
    [icw.exchange, "CGE zeroth-order ∇", "No weights, no autograd. Coordinate-wise Gradient Estimation perturbs the input ±σ and reads logit changes to estimate ∇.", GREEN],
    [icw.sliders, "feat_scale = 2√N", "Injected features are amplified to ≈2√N. This single lever — not the optimizer — is what lifts ASR from ~30% to ~95%.", CRIMSON],
    [icw.target, "CW-margin loss", "Carlini-Wagner margin (κ = −0.1) drives the true-class logit below the best other class — a smooth flip objective.", TEAL],
  ];
  ca.forEach((d, i) => {
    const col = i % 2, row = Math.floor(i / 2);
    chip(s, 0.7 + col * 6.15, 2.35 + row * 2.1, 5.85, 1.85, d[0], d[1], d[2], d[3]);
  });

  // ============================================================ S5 Restart ladder
  s = pres.addSlide(); bgLight(s);
  header(s, "Component B", "The Multi-Restart Baseline Ladder", BLUE);
  s.addText("Before any diffusion, 5 Adam restarts try to flip the graph. The best (lowest-margin) x₀ is cached as the diffusion anchor.",
    { x: 0.7, y: 1.62, w: 12, h: 0.4, fontFace: BODY, fontSize: 13, italic: true, color: MUT, margin: 0 });
  const restarts = [
    ["1", "lr 5e-3 · 50 ep", BLUE], ["2", "lr 1e-2 · 50 ep", BLUE], ["3", "lr 2e-2 · 75 ep", TEAL],
    ["4", "lr 5e-3 · 100 ep", TEAL], ["5", "lr 1e-2 · 100 ep", NAVY],
  ];
  restarts.forEach((d, i) => {
    const x = 0.7 + i * 2.46;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 2.45, w: 2.2, h: 1.15, rectRadius: 0.08, fill: { color: WHITE }, shadow: shadow() });
    s.addShape(pres.shapes.OVAL, { x: x + 0.82, y: 2.6, w: 0.56, h: 0.56, fill: { color: d[2] } });
    s.addText(d[0], { x: x + 0.82, y: 2.6, w: 0.56, h: 0.56, fontFace: HEAD, fontSize: 22, bold: true, color: WHITE, align: "center", valign: "middle", margin: 0 });
    s.addText(d[1], { x, y: 3.2, w: 2.2, h: 0.35, fontFace: BODY, fontSize: 11, color: MUT, align: "center", margin: 0 });
    if (i < 4) arrow(s, x + 2.2 + 0.02, 3.0, x + 2.46 - 0.02, 3.0, GREY);
  });
  chip(s, 0.7, 4.1, 12.0, 1.05, icw.check, "Cache the best x₀ across restarts",
    "If any restart flips the graph → done (this alone solves ~95% of graphs). Otherwise the lowest-margin candidate becomes the anchor x_init that the diffusion process denoises toward.", GREEN);
  chip(s, 0.7, 5.35, 12.0, 1.05, icw.lightbulb, "Why an anchor matters",
    "The early failure (N23) ran diffusion from x₀ = 0, where DDIM predicts x̂₀ = 0 — testing only a zero-feature injection. Anchoring at the best baseline x₀ gives the score a meaningful place to denoise from.", GOLD);

  // ============================================================ S6 N23 -> N24 evolution
  s = pres.addSlide(); bgLight(s);
  header(s, "Why It's Built This Way", "N23 → N24: Why Feature-Only Diffusion Failed", CRIMSON);
  s.addImage({ path: FIG("chart_mechanism_evolution.png"), x: 0.7, y: 1.75, w: 7.5, h: 4.54, shadow: shadow() });
  const rx = 8.55;
  s.addShape(pres.shapes.RECTANGLE, { x: rx, y: 1.75, w: 4.15, h: 4.54, fill: { color: WHITE }, shadow: shadow() });
  s.addShape(pres.shapes.RECTANGLE, { x: rx, y: 1.75, w: 4.15, h: 0.6, fill: { color: NAVY } });
  s.addText("THE ROOT CAUSE", { x: rx + 0.25, y: 1.83, w: 3.6, h: 0.45, fontFace: HEAD, fontSize: 14, bold: true, color: WHITE, charSpacing: 2, margin: 0, valign: "middle" });
  s.addText([
    { text: "v1–v3 diffused features only.", options: { bold: true, color: CRIMSON, breakLine: true, fontSize: 14 } },
    { text: "Under a fixed full-edge topology, GCN normalization + feat_scale saturate — feature content stops mattering. ~5% of graphs are topology-immune: no feature vector flips them.", options: { color: MUT, breakLine: true, fontSize: 12 } },
    { text: "", options: { breakLine: true, fontSize: 6 } },
    { text: "Fix → make edges a variable.", options: { bold: true, color: GREEN, breakLine: true, fontSize: 14 } },
    { text: "Joint X+A diffusion adds a topological dimension the score can actually move.", options: { color: MUT, fontSize: 12 } },
  ], { x: rx + 0.25, y: 2.5, w: 3.7, h: 2.5, fontFace: BODY, margin: 0, lineSpacingMultiple: 1.04, valign: "top" });
  s.addShape(pres.shapes.RECTANGLE, { x: rx + 0.25, y: 5.18, w: 3.65, h: 0.92, fill: { color: "EAF6F2" } });
  s.addShape(pres.shapes.RECTANGLE, { x: rx + 0.25, y: 5.18, w: 0.09, h: 0.92, fill: { color: GREEN } });
  s.addText([
    { text: "67.3%  →  97.7%", options: { fontSize: 18, bold: true, color: GREEN, breakLine: true } },
    { text: "the moment edges entered the state", options: { fontSize: 10.5, color: "3A6157" } },
  ], { x: rx + 0.45, y: 5.26, w: 3.4, h: 0.76, fontFace: HEAD, margin: 0, valign: "middle" });

  // ============================================================ S7 Core idea: joint X+A
  s = pres.addSlide(); bgDark(s);
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 0.28, h: H, fill: { color: CRIMSON } });
  s.addText("CORE IDEA", { x: 0.7, y: 0.5, w: 11, h: 0.4, fontFace: HEAD, fontSize: 14, bold: true, color: GOLD, charSpacing: 3, margin: 0 });
  s.addText("The Diffusion State is (X, A) — Not Just X", { x: 0.7, y: 0.9, w: 12, h: 0.8, fontFace: HEAD, fontSize: 30, bold: true, color: WHITE, margin: 0 });
  // left: feature-only (fail), right: joint (win)
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.8, y: 2.1, w: 5.7, h: 4.05, rectRadius: 0.06, fill: { color: "2C355F" }, line: { color: CRIMSON, width: 1.5 } });
  s.addImage({ data: icw.times, x: 1.1, y: 2.35, w: 0.5, h: 0.5 });
  s.addText("Feature-only diffusion", { x: 1.75, y: 2.35, w: 4.5, h: 0.5, fontFace: HEAD, fontSize: 17, bold: true, color: WHITE, valign: "middle", margin: 0 });
  s.addText([
    { text: "state = x   (node features)", options: { fontFace: MONO, fontSize: 13, color: "FFB3B8", breakLine: true } },
    { text: "", options: { breakLine: true, fontSize: 6 } },
    { text: "• denoise x toward an adversarial feature vector", options: { breakLine: true, color: "CCD4EC" } },
    { text: "• edges fixed (full connect)", options: { breakLine: true, color: "CCD4EC" } },
    { text: "• GCN norm absorbs it → inert", options: { breakLine: true, color: "FFB3B8", bold: true } },
  ], { x: 1.1, y: 3.1, w: 5.2, h: 2.15, fontFace: BODY, fontSize: 13, margin: 0, lineSpacingMultiple: 1.55, valign: "top" });
  s.addText("67.31% — no gain over a zero-feature injection", { x: 1.1, y: 5.45, w: 5.2, h: 0.5, fontFace: HEAD, fontSize: 14, bold: true, color: CRIMSON, margin: 0 });

  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 6.85, y: 2.1, w: 5.7, h: 4.05, rectRadius: 0.06, fill: { color: "1E3A4C" }, line: { color: GREEN, width: 1.5 } });
  s.addImage({ data: icw.check, x: 7.15, y: 2.35, w: 0.5, h: 0.5 });
  s.addText("Joint X + A diffusion", { x: 7.8, y: 2.35, w: 4.5, h: 0.5, fontFace: HEAD, fontSize: 17, bold: true, color: WHITE, valign: "middle", margin: 0 });
  s.addText([
    { text: "state = (x, A_sv)   features + edges", options: { fontFace: MONO, fontSize: 13, color: "A8E6D5", breakLine: true } },
    { text: "", options: { breakLine: true, fontSize: 6 } },
    { text: "• denoise features AND search the edge mask", options: { breakLine: true, color: "CCD4EC" } },
    { text: "• edge cardinality bounded by budget k", options: { breakLine: true, color: "CCD4EC" } },
    { text: "• adds a topological dimension → flips", options: { breakLine: true, color: "A8E6D5", bold: true } },
  ], { x: 7.15, y: 3.1, w: 5.2, h: 2.15, fontFace: BODY, fontSize: 13, margin: 0, lineSpacingMultiple: 1.55, valign: "top" });
  s.addText("97.7%+ — the ceiling breaks", { x: 7.15, y: 5.45, w: 5.2, h: 0.5, fontFace: HEAD, fontSize: 14, bold: true, color: GREEN, margin: 0 });

  // ============================================================ S8 DDIM math
  s = pres.addSlide(); bgLight(s);
  header(s, "Mechanism", "The DDIM Engine + Adversarial-Margin Guidance", BLUE);
  // left math panel
  s.addShape(pres.shapes.RECTANGLE, { x: 0.7, y: 1.8, w: 6.4, h: 4.5, fill: { color: WHITE }, shadow: shadow() });
  s.addShape(pres.shapes.RECTANGLE, { x: 0.7, y: 1.8, w: 6.4, h: 0.55, fill: { color: NAVY } });
  s.addText("DISCRETE-TIME DDIM", { x: 0.95, y: 1.86, w: 6, h: 0.43, fontFace: HEAD, fontSize: 13, bold: true, color: WHITE, charSpacing: 2, valign: "middle", margin: 0 });
  s.addText([
    { text: "Bounded linear schedule", options: { bold: true, color: BLUE, breakLine: true, fontSize: 13 } },
    { text: "ᾱ : 0.99 → 0.50  over T = 20 steps", options: { fontFace: MONO, color: INK, breakLine: true, fontSize: 12.5 } },
    { text: "", options: { breakLine: true, fontSize: 7 } },
    { text: "Forward (noise the anchor)", options: { bold: true, color: BLUE, breakLine: true, fontSize: 13 } },
    { text: "x_T = √ᾱ_T · x_init + √(1−ᾱ_T) · ε₀", options: { fontFace: MONO, color: INK, breakLine: true, fontSize: 12.5 } },
    { text: "", options: { breakLine: true, fontSize: 7 } },
    { text: "Reverse step (predict-x₀ form)", options: { bold: true, color: BLUE, breakLine: true, fontSize: 13 } },
    { text: "x̂₀ = (x_t − √(1−ᾱ_t)·ε) / √ᾱ_t", options: { fontFace: MONO, color: INK, breakLine: true, fontSize: 12.5 } },
    { text: "x_{t−1} = √ᾱ_{t−1}·x̂₀ + √(1−ᾱ_{t−1})·ε̂", options: { fontFace: MONO, color: INK, fontSize: 12.5 } },
  ], { x: 0.95, y: 2.5, w: 6.0, h: 2.9, fontFace: BODY, margin: 0, lineSpacingMultiple: 1.12, valign: "top" });
  s.addShape(pres.shapes.RECTANGLE, { x: 0.95, y: 5.45, w: 5.9, h: 0.7, fill: { color: "EAF1F8" } });
  s.addShape(pres.shapes.RECTANGLE, { x: 0.95, y: 5.45, w: 0.09, h: 0.7, fill: { color: BLUE } });
  s.addText("Bounded (ᾱ never reaches 0 or 1): the anchor is only partially noised, so the trajectory stays near the adversarial manifold instead of wandering to pure noise.",
    { x: 1.15, y: 5.45, w: 5.6, h: 0.7, fontFace: BODY, fontSize: 10.5, italic: true, color: "2E5A7A", valign: "middle", margin: 0, lineSpacingMultiple: 1.0 });
  // right guidance panel
  s.addShape(pres.shapes.RECTANGLE, { x: 7.3, y: 1.8, w: 5.35, h: 4.5, fill: { color: WHITE }, shadow: shadow() });
  s.addShape(pres.shapes.RECTANGLE, { x: 7.3, y: 1.8, w: 5.35, h: 0.55, fill: { color: CRIMSON } });
  s.addText("ADVERSARIAL-MARGIN GUIDANCE", { x: 7.55, y: 1.86, w: 5, h: 0.43, fontFace: HEAD, fontSize: 13, bold: true, color: WHITE, charSpacing: 1, valign: "middle", margin: 0 });
  s.addText([
    { text: "At each reverse step:", options: { bold: true, color: CRIMSON, breakLine: true, fontSize: 13 } },
    { text: "1. ZO-CGE estimates the score", options: { breakLine: true, color: MUT, fontSize: 12.5 } },
    { text: "   s = ∇_{x̂₀} CW(model, graph)", options: { fontFace: MONO, color: INK, breakLine: true, fontSize: 12 } },
    { text: "2. steer the noise toward lower margin", options: { breakLine: true, color: MUT, fontSize: 12.5 } },
    { text: "   ε̂ = ε₀ − λ·√(1−ᾱ_t)·s", options: { fontFace: MONO, color: INK, breakLine: true, fontSize: 12 } },
    { text: "   (λ = guidance = 2.0)", options: { fontFace: MONO, color: GREY, breakLine: true, fontSize: 11 } },
    { text: "3. take the DDIM step → repeat", options: { color: MUT, fontSize: 12.5, breakLine: true } },
    { text: "", options: { breakLine: true, fontSize: 7 } },
    { text: "Over T = 20 steps the injection is denoised straight toward the decision boundary; the lowest-margin x̂₀ is kept.", options: { color: "5A5F75", italic: true, fontSize: 11.5 } },
  ], { x: 7.55, y: 2.5, w: 5.0, h: 2.65, fontFace: BODY, margin: 0, lineSpacingMultiple: 1.15, valign: "top" });
  s.addShape(pres.shapes.RECTANGLE, { x: 7.55, y: 5.25, w: 4.85, h: 0.95, fill: { color: "FDF3E2" } });
  s.addShape(pres.shapes.RECTANGLE, { x: 7.55, y: 5.25, w: 0.09, h: 0.95, fill: { color: GOLD } });
  s.addText("Queries replace gradients: the diffusion score is itself a zeroth-order estimate — the whole engine stays black-box.",
    { x: 7.75, y: 5.25, w: 4.6, h: 0.95, fontFace: BODY, fontSize: 11.5, italic: true, color: "8A6D2A", valign: "middle", margin: 0 });

  // ============================================================ S9 Edge search internals
  s = pres.addSlide(); bgLight(s);
  header(s, "Mechanism", "Searching the Edge Mask (the A in X+A)", GOLD);
  const es = [
    [icw.network, "Per-node edge mask", "Each injected node i carries a binary mask edge_mask[i] ∈ {0,1}^N over original nodes, with ‖mask[i]‖₁ ≤ k (cardinality budget).", TEAL],
    [icw.layer, "Budget as upper bound", "k is a ceiling, not a target. Greedy search keeps only positive-value edges, so the attack uses the fewest edges that still flip.", BLUE],
    [icw.flask, "k_sweep on stuck graphs", "If the default k fails, retry with k ∈ {0.25, 0.5, 1.5, 2}× — different cardinalities suit different graph sizes.", GOLD],
    [icw.exchange, "2-swap refinement", "After top-k projection, try top-K-in × top-K-out swaps (K²=16) to escape coupled-edge optima. Same ASR, 35% faster.", CRIMSON],
    [icw.bolt, "Per-node escalation (m=2)", "Add a 2nd injected node with an INDEPENDENT mask. Shared masks make duplicate hubs that interfere (−1.85 pp); independent masks add +0.36 pp.", NAVY],
    [icw.dice, "Multi-seed", "2 random seeds per diffusion attempt — cheap variance reduction on the stochastic edge search.", GREEN],
  ];
  es.forEach((d, i) => {
    const col = i % 2, row = Math.floor(i / 2);
    chip(s, 0.7 + col * 6.15, 2.05 + row * 1.58, 5.85, 1.35, d[0], d[1], d[2], d[3]);
  });

  // ============================================================ S10 Fallback ladder when each fires
  s = pres.addSlide(); bgLight(s);
  header(s, "Putting It Together", "When Each Tier Fires  (94.85% → 98.43%)", TEAL);
  s.addImage({ path: FIG("chart_asr_progression.png"), x: 0.7, y: 1.9, w: 7.4, h: 4.24, shadow: shadow() });
  const tiers = [
    [icw.cube, "Baseline", "5-restart Adam + ZO-CGE", NAVY],
    [icw.sitemap, "Joint X+A diffusion", "bounded DDIM + edge search", TEAL],
    [icw.layer, "Budget ≤ k", "edges as upper bound", BLUE],
    [icw.flask, "k_sweep", "retry stuck graphs at other k", GOLD],
    [icw.bolt, "Per-node escalation", "m=2, independent masks", CRIMSON],
  ];
  tiers.forEach((d, i) => {
    const y = 1.9 + i * 0.87;
    s.addShape(pres.shapes.RECTANGLE, { x: 8.5, y, w: 4.2, h: 0.74, fill: { color: WHITE }, shadow: shadow() });
    s.addShape(pres.shapes.OVAL, { x: 8.65, y: y + 0.13, w: 0.48, h: 0.48, fill: { color: d[3] } });
    s.addImage({ data: d[0], x: 8.75, y: y + 0.23, w: 0.28, h: 0.28 });
    s.addText(`${i + 1}. ${d[1]}`, { x: 9.25, y: y + 0.06, w: 3.4, h: 0.34, fontFace: HEAD, fontSize: 12.5, bold: true, color: INK, margin: 0, valign: "middle" });
    s.addText(d[2], { x: 9.25, y: y + 0.40, w: 3.4, h: 0.30, fontFace: BODY, fontSize: 10, color: "5A5F75", margin: 0, valign: "middle" });
  });

  // ============================================================ S11 Key insight: readout
  s = pres.addSlide(); bgLight(s);
  header(s, "Design Insight", "The Readout — Not the Convolution — is the Attack Surface", CRIMSON);
  s.addImage({ path: FIG("chart_robustness.png"), x: 0.7, y: 1.85, w: 7.6, h: 4.3, shadow: shadow() });
  const ins = [
    [icw.check, "Mean-pool falls", "GIN (sum), GAT (attention), SAGE (sampling) all use mean-pool readout — and all hit 99.9–100% ASR. Changing the conv is no defense.", CRIMSON],
    [icw.shield, "Median-pool resists", "Only a median / trimmed-mean readout ignores the injected outlier node — dropping ASR to 92.46%. Defend the pooling.", GREEN],
    [icw.bolt, "Diffusion earns its keep here", "On undefended models a baseline suffices; against median-pool, diffusion is the difference between ~63% and 92%.", GOLD],
  ];
  ins.forEach((d, i) => {
    const y = 1.9 + i * 1.45;
    s.addShape(pres.shapes.RECTANGLE, { x: 8.6, y, w: 4.1, h: 1.25, fill: { color: WHITE }, shadow: shadow() });
    s.addShape(pres.shapes.OVAL, { x: 8.8, y: y + 0.35, w: 0.55, h: 0.55, fill: { color: d[3] } });
    s.addImage({ data: d[0], x: 8.91, y: y + 0.46, w: 0.33, h: 0.33 });
    s.addText([{ text: d[1] + "  ", options: { bold: true, color: INK } }, { text: d[2], options: { color: MUT } }],
      { x: 9.5, y: y + 0.12, w: 3.05, h: 1.0, fontFace: BODY, fontSize: 10.5, margin: 0, valign: "middle", lineSpacingMultiple: 1.0 });
  });

  // ============================================================ S12 Results + fair comparison
  s = pres.addSlide(); bgLight(s);
  header(s, "Results", "Fair, Budget-Matched Comparison (PROTEINS GCN)", GREEN);
  s.addText("All methods at node_budget = 1, 5-fold. The original 54.26% → 98.43% was NOT budget-matched; under one identical budget the gap is honest.",
    { x: 0.7, y: 1.62, w: 12, h: 0.45, fontFace: BODY, fontSize: 13, italic: true, color: MUT, margin: 0 });
  // table
  const tx = 0.7, tyt = 2.3, cw = [4.6, 3.5, 3.5], rh = 0.62;
  const cols = ["Method", "GZOO budget (1 edge)", "N24 budget (full)"];
  cols.forEach((c, i) => {
    const x = tx + cw.slice(0, i).reduce((a, b) => a + b, 0);
    s.addShape(pres.shapes.RECTANGLE, { x, y: tyt, w: cw[i], h: rh, fill: { color: NAVY } });
    s.addText(c, { x: x + 0.12, y: tyt, w: cw[i] - 0.2, h: rh, fontFace: HEAD, fontSize: 12.5, bold: true, color: WHITE, valign: "middle", margin: 0 });
  });
  const rows = [
    ["GZOO-origin (fs=1.0)", "29.65 ± 12.04", "75.25 ± 4.46", CRIMSON],
    ["gzoo-katze (fs=auto_x2)", "92.31 ± 5.42", "94.85 ± 7.44", BLUE],
    ["diffusion-katze (N24)", "95.38 ± 3.87", "98.43 ± 2.29", GREEN],
    ["GZOO-paper (cited)", "54.26", "n/a", GREY],
  ];
  rows.forEach((r, ri) => {
    const y = tyt + rh + ri * rh;
    const band = ri % 2 ? "FFFFFF" : "EAEEF7";
    s.addShape(pres.shapes.RECTANGLE, { x: tx, y, w: cw[0] + cw[1] + cw[2], h: rh, fill: { color: band } });
    s.addShape(pres.shapes.RECTANGLE, { x: tx, y, w: 0.08, h: rh, fill: { color: r[3] } });
    s.addText(r[0], { x: tx + 0.2, y, w: cw[0] - 0.3, h: rh, fontFace: BODY, fontSize: 12, bold: ri === 2, color: INK, valign: "middle", margin: 0 });
    s.addText(r[1], { x: tx + cw[0] + 0.12, y, w: cw[1] - 0.2, h: rh, fontFace: MONO, fontSize: 12, bold: ri === 2, color: ri === 2 ? GREEN : MUT, valign: "middle", margin: 0 });
    s.addText(r[2], { x: tx + cw[0] + cw[1] + 0.12, y, w: cw[2] - 0.2, h: rh, fontFace: MONO, fontSize: 12, bold: ri === 2, color: ri === 2 ? GREEN : MUT, valign: "middle", margin: 0 });
  });
  // takeaway callouts
  const co = [
    ["+65.7 pp", "diffusion-katze vs GZOO-origin\nat the SAME 1-edge budget", GREEN],
    ["feat_scale", "the dominant lever — fs=1.0→auto_x2\nmoves ASR far more than the optimizer", CRIMSON],
    ["edges matter", "every method gains from full edges;\ntopology is a real budget axis", GOLD],
  ];
  co.forEach((d, i) => {
    const x = 0.7 + i * 4.1;
    s.addShape(pres.shapes.RECTANGLE, { x, y: 5.7, w: 3.85, h: 1.35, fill: { color: WHITE }, shadow: shadow() });
    s.addShape(pres.shapes.RECTANGLE, { x, y: 5.7, w: 3.85, h: 0.09, fill: { color: d[2] } });
    s.addText(d[0], { x: x + 0.2, y: 5.86, w: 3.5, h: 0.5, fontFace: HEAD, fontSize: 22, bold: true, color: d[2], margin: 0 });
    s.addText(d[1], { x: x + 0.2, y: 6.36, w: 3.5, h: 0.62, fontFace: BODY, fontSize: 10.5, color: MUT, margin: 0, lineSpacingMultiple: 1.0 });
  });

  // ============================================================ S13 Summary (dark)
  s = pres.addSlide(); bgDark(s);
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 0.28, h: H, fill: { color: GOLD } });
  s.addImage({ data: ic.lightbulb, x: 0.9, y: 0.7, w: 0.7, h: 0.7 });
  s.addText("Architecture in Three Sentences", { x: 0.9, y: 1.5, w: 11.5, h: 0.8, fontFace: HEAD, fontSize: 32, bold: true, color: WHITE, margin: 0 });
  const sum = [
    ["1", "Cascade of fallbacks", "Cheap 5-restart Adam + ZO-CGE solves most graphs; the diffusion engine is the last resort for the hard, topology-immune ones.", TEAL],
    ["2", "Topology in the state", "The decisive design choice is making edges part of the diffusion state (X+A). Feature-only diffusion is provably inert under mean-pool GCNs.", CRIMSON],
    ["3", "Black-box throughout", "Even the diffusion score is a zeroth-order estimate — no gradients, no weights, 1 injected node. Defend the readout, not the convolution.", GOLD],
  ];
  sum.forEach((d, i) => {
    const y = 2.6 + i * 1.4;
    s.addShape(pres.shapes.OVAL, { x: 0.95, y, w: 0.85, h: 0.85, fill: { color: d[3] } });
    s.addText(d[0], { x: 0.95, y, w: 0.85, h: 0.85, fontFace: HEAD, fontSize: 30, bold: true, color: WHITE, align: "center", valign: "middle", margin: 0 });
    s.addText([{ text: d[1] + "  —  ", options: { bold: true, color: WHITE, fontSize: 17 } }, { text: d[2], options: { color: "CADCFC", fontSize: 14 } }],
      { x: 2.1, y: y - 0.05, w: 10.5, h: 0.95, fontFace: BODY, margin: 0, valign: "middle", lineSpacingMultiple: 1.05 });
  });
  s.addText("Diffusion-KATZE  ·  autoresearch-katze  ·  PROTEINS / GNN victims", { x: 0.9, y: 6.95, w: 11.5, h: 0.4, fontFace: BODY, fontSize: 12, color: GREY, italic: true, margin: 0 });

  await pres.writeFile({ fileName: "diffusion_katze_architecture.pptx" });
  console.log("WROTE diffusion_katze_architecture.pptx with", pres.slides.length, "slides");
})();
