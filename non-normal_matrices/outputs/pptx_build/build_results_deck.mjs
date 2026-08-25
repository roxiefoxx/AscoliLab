import fs from "node:fs/promises";
import { FileBlob, PresentationFile } from "@oai/artifact-tool";

const SOURCE_PPTX = "C:/Users/ranic/OneDrive - George Mason University - O365 Production/04_2026 Spring/BINF751 - Biochem Modeling/Final Project/presntation/BINF751_finalproject_RVA_v8.pptx";
const FINAL_PPTX = "C:/Users/ranic/Documents/GitHub/non-normal_matrices/outputs/BINF751_finalproject_RVA_results_v9.pptx";
const OUT_DIR = "C:/Users/ranic/Documents/GitHub/non-normal_matrices/outputs/pptx_build/final_render";
const DATA_DIR = "C:/Users/ranic/Documents/GitHub/non-normal_matrices/outputs/mij_paper_replication";

const green = "#005138";
const yellow = "#FFC733";
const lightGreen = "#E8F2EE";
const lightYellow = "#FFF3CC";
const blue = "#5B9BD5";
const gray = "#E7E6E6";
const text = "#111111";

async function readCsv(path) {
  const raw = await fs.readFile(path, "utf8");
  const lines = raw.trim().split(/\r?\n/);
  const headers = parseCsvLine(lines[0]);
  return lines.slice(1).filter(Boolean).map((line) => {
    const values = parseCsvLine(line);
    return Object.fromEntries(headers.map((h, i) => [h, values[i] ?? ""]));
  });
}

function parseCsvLine(line) {
  const out = [];
  let cur = "";
  let inQuotes = false;
  for (let i = 0; i < line.length; i += 1) {
    const ch = line[i];
    if (ch === '"' && line[i + 1] === '"') {
      cur += '"';
      i += 1;
    } else if (ch === '"') {
      inQuotes = !inQuotes;
    } else if (ch === "," && !inQuotes) {
      out.push(cur);
      cur = "";
    } else {
      cur += ch;
    }
  }
  out.push(cur);
  return out;
}

function num(value) {
  return Number.parseFloat(value);
}

function fmt(value, digits = 3) {
  const n = Number(value);
  if (!Number.isFinite(n)) return String(value);
  if (Math.abs(n) >= 1e5 || (Math.abs(n) > 0 && Math.abs(n) < 1e-3)) {
    return n.toExponential(2);
  }
  return n.toFixed(digits);
}

function setText(presentation, id, value, style = {}) {
  const shape = presentation.resolve(id);
  shape.text = value;
  shape.text.style = {
    fontSize: style.fontSize ?? 27,
    color: style.color ?? text,
    bold: style.bold ?? false,
    alignment: style.alignment ?? "left",
  };
}

function addText(slide, value, position, style = {}) {
  const shape = slide.shapes.add({
    geometry: "textbox",
    position,
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  shape.text = value;
  shape.text.style = {
    fontSize: style.fontSize ?? 24,
    color: style.color ?? text,
    bold: style.bold ?? false,
    alignment: style.alignment ?? "left",
  };
  return shape;
}

function addMetric(slide, label, value, position, fill = lightGreen) {
  const box = slide.shapes.add({
    geometry: "roundRect",
    position,
    fill,
    line: { style: "solid", fill: green, width: 1 },
    borderRadius: "rounded-md",
  });
  box.text = `${value}\n${label}`;
  box.text.style = { fontSize: 22, bold: true, color: green, alignment: "center" };
  return box;
}

function addBlockBox(slide, title, body, position, fill) {
  const box = slide.shapes.add({
    geometry: "rect",
    position,
    fill,
    line: { style: "solid", fill: green, width: 1 },
  });
  box.text = `${title}\n${body}`;
  box.text.style = { fontSize: 21, color: text, bold: false, alignment: "center" };
  return box;
}

function addSources(slide, body, sources) {
  slide.speakerNotes.clear();
  slide.speakerNotes.textFrame.setText(`${body}\n\n[Sources]\n${sources.map((s) => `- ${s}`).join("\n")}`);
  slide.speakerNotes.setVisible(true);
}

async function main() {
  await fs.mkdir(OUT_DIR, { recursive: true });

  const metadata = (await readCsv(`${DATA_DIR}/metadata.csv`))[0];
  const blocks = await readCsv(`${DATA_DIR}/ei_block_summary.csv`);
  const enrichment = await readCsv(`${DATA_DIR}/binary_motif_enrichment.csv`);
  const sensitivity = await readCsv(`${DATA_DIR}/dominant_eigenvalue_block_sensitivity.csv`);
  const sweep = await readCsv(`${DATA_DIR}/chain_enrichment_sweep.csv`);

  const motifMeans = {};
  for (const row of enrichment) {
    const key = row.motif;
    motifMeans[key] ??= [];
    motifMeans[key].push(num(row.enrichment_ratio));
  }
  const motifSummary = Object.fromEntries(
    Object.entries(motifMeans).map(([key, values]) => [
      key,
      values.reduce((a, b) => a + b, 0) / values.length,
    ]),
  );

  const block = (receiver, sender) =>
    blocks.find((row) => row.receiver_group === receiver && row.sender_group === sender);

  const sweepDiag = sweep.filter((row) => row.receiver_group === "E" && row.input_group === "E");
  const sweepX = sweepDiag.map((row) => num(row.chain_enrichment_multiplier));
  const sweepY = sweepDiag.map((row) => num(row.spectral_radius));

  const presentation = await PresentationFile.importPptx(await FileBlob.load(SOURCE_PPTX));

  // Slide 11
  setText(presentation, "sh/9gza9sze", "My extension: empirical J construction", { fontSize: 48, color: green });
  setText(
    presentation,
    "sh/n69grmpw",
    [
      "Experimental dataset: Hippocampome M_ij matrix and netlist",
      "85 cell types, 1,641 directed nonzero weighted connections",
      "CSV orientation: sender rows, receiver columns",
      "Paper orientation: J[receiver, sender], so the matrix is transposed before analysis",
      `E/I labels: ${metadata.n_excitatory} excitatory, ${metadata.n_inhibitory} inhibitory from mij_netlist.csv`,
      `Normalization: raw rho(J) = ${fmt(metadata.raw_spectral_radius, 2)} -> normalized rho(J) = 1.000`,
    ].join("\n"),
    { fontSize: 25 },
  );
  addSources(
    presentation.slides.getItem(10),
    "Use this slide to shift from the paper summary into the empirical extension. Emphasize that no synthetic networks were generated.",
    [
      "data/mij_matrix.csv",
      "data/mij_netlist.csv",
      "outputs/mij_paper_replication/metadata.csv",
    ],
  );

  // Slide 12
  const slide12 = presentation.slides.getItem(11);
  setText(presentation, "sh/xw3q94ne", "J construction: E/I blocks after normalization", { fontSize: 48, color: green });
  setText(
    presentation,
    "sh/ra943il8",
    [
      "The signed empirical J is summarized as four E/I blocks.",
      "Rows are receiver classes; columns are sender classes.",
      "Mean weights and densities are computed after spectral normalization.",
    ].join("\n"),
    { fontSize: 25 },
  );
  addBlockBox(slide12, "E <- E", `mean ${fmt(block("E", "E").mean_weight)}\ndensity ${fmt(block("E", "E").density_nonzero)}`, { left: 690, top: 160, width: 210, height: 135 }, lightYellow);
  addBlockBox(slide12, "E <- I", `mean ${fmt(block("E", "I").mean_weight)}\ndensity ${fmt(block("E", "I").density_nonzero)}`, { left: 925, top: 160, width: 210, height: 135 }, gray);
  addBlockBox(slide12, "I <- E", `mean ${fmt(block("I", "E").mean_weight)}\ndensity ${fmt(block("I", "E").density_nonzero)}`, { left: 690, top: 320, width: 210, height: 135 }, lightYellow);
  addBlockBox(slide12, "I <- I", `mean ${fmt(block("I", "I").mean_weight)}\ndensity ${fmt(block("I", "I").density_nonzero)}`, { left: 925, top: 320, width: 210, height: 135 }, gray);
  addSources(
    slide12,
    "Block summary: note the strong I-receiver/E-sender mean and the sparse E-receiver/I-sender block.",
    ["outputs/mij_paper_replication/ei_block_summary.csv"],
  );

  // Slide 13
  const slide13 = presentation.slides.getItem(12);
  setText(presentation, "sh/gb6x0zm9", "Eigenspectrum: normalized J sits on the boundary", { fontSize: 48, color: green });
  setText(
    presentation,
    "sh/o3i5w3ad",
    [
      `Full network spectral radius: ${fmt(metadata.spectral_radius)}`,
      `Max real eigenvalue: ${fmt(metadata.max_real_eigenvalue)}`,
      `cond(I - J): ${fmt(metadata.condition_number_I_minus_J)}`,
      `Numerical abscissa: ${fmt(metadata.numerical_abscissa)}`,
      "Interpretation: radius-one response is a near-critical, ill-conditioned limit.",
    ].join("\n"),
    { fontSize: 25 },
  );
  slide13.charts.add("bar", {
    position: { left: 765, top: 175, width: 360, height: 300 },
    categories: ["rho(J)", "max Re", "J_eff rho"],
    series: [{ name: "Value", values: [num(metadata.spectral_radius), num(metadata.max_real_eigenvalue), num(metadata.jeff_spectral_radius)], fill: "accent1" }],
    hasLegend: false,
    dataLabels: { showValue: true, position: "outEnd" },
  });
  addSources(
    slide13,
    "The full empirical matrix is normalized to spectral radius 1. The very large condition number is why the radius-one response should be read as an instability-boundary diagnostic.",
    [
      "outputs/mij_paper_replication/metadata.csv",
      "outputs/mij_paper_replication/leading_eigenvalues.csv",
    ],
  );

  // Slide 14
  const slide14 = presentation.slides.getItem(13);
  setText(presentation, "sh/x4ni9ofy", "Meet Jeff: motif-corrected dynamics cross unity", { fontSize: 48, color: green });
  for (const id of ["im/0vud8v2d", "im/rq1cvq1g"]) {
    try {
      presentation.resolve(id).delete();
    } catch {}
  }
  addText(
    slide14,
    [
      "J_eff = J0 + block-averaged empirical Z^2",
      `rho(J_eff) = ${fmt(metadata.jeff_spectral_radius)}`,
      `max Re(lambda_eff) = ${fmt(metadata.jeff_max_real_eigenvalue)}`,
      `cond(I - J_eff) = ${fmt(1364.316, 1)}`,
      "J_eff is derived from normalized J, then left un-renormalized to expose motif-driven instability.",
    ].join("\n"),
    { left: 60, top: 150, width: 540, height: 350 },
    { fontSize: 26 },
  );
  slide14.charts.add("bar", {
    position: { left: 665, top: 150, width: 430, height: 330 },
    categories: ["J", "J_eff"],
    series: [{ name: "Spectral radius", values: [1.0, num(metadata.jeff_spectral_radius)], fill: "accent5" }],
    hasLegend: false,
    dataLabels: { showValue: true, position: "outEnd" },
    yAxis: { majorGridlines: { style: "solid", fill: "#D9D9D9", width: 1 } },
  });
  addSources(
    slide14,
    "Jeff summarizes the paper's effective-connectivity approximation on the empirical matrix. Its radius is above 1 because Jeff is a derived matrix, not the normalized input J.",
    [
      "outputs/mij_paper_replication/metadata.csv",
      "outputs/mij_paper_replication/jeff_leading_eigenvalues.csv",
    ],
  );

  // Slide 15
  const slide15 = presentation.slides.getItem(14);
  setText(presentation, "sh/hsvat0fa", "Motif analysis: enrichment is not uniform", { fontSize: 48, color: green });
  try {
    presentation.resolve("im/wbalg3a1").delete();
  } catch {}
  addText(
    slide15,
    [
      "Observed binary motifs were compared with block-density independence.",
      `Mean enrichment ratios: chain ${fmt(motifSummary.chain, 2)}, reciprocal ${fmt(motifSummary.reciprocal, 2)}, divergent ${fmt(motifSummary.divergent, 2)}, convergent ${fmt(motifSummary.convergent, 2)}.`,
      "Reciprocal motifs are strongest on average, while chain enrichment is heterogeneous across E/I paths.",
    ].join("\n"),
    { left: 58, top: 145, width: 525, height: 385 },
    { fontSize: 24 },
  );
  slide15.charts.add("bar", {
    position: { left: 650, top: 145, width: 470, height: 340 },
    categories: ["Chain", "Reciprocal", "Divergent", "Convergent"],
    series: [
      {
        name: "Mean enrichment",
        values: [motifSummary.chain, motifSummary.reciprocal, motifSummary.divergent, motifSummary.convergent],
        fill: "accent1",
      },
    ],
    hasLegend: false,
    dataLabels: { showValue: true, position: "outEnd" },
  });
  addSources(
    slide15,
    "This slide reports deterministic motif enrichment from the empirical binary graph, not a synthetic-network simulation.",
    [
      "outputs/mij_paper_replication/binary_motif_enrichment.csv",
      "outputs/mij_paper_replication/motif_correlations.csv",
    ],
  );

  // Slide 16
  const slide16 = presentation.slides.getItem(15);
  setText(presentation, "sh/e1sf65o3", "Results: chain enrichment moves the stability boundary", { fontSize: 46, color: green });
  setText(
    presentation,
    "sh/1ojy10ne",
    [
      "Chain component sweep:",
      "rho = 0.974 at 0.50x enrichment",
      "rho = 1.009 at 0.75x enrichment",
      "E <- E perturbations push the dominant mode most strongly",
      "Radius-one response is ill-conditioned; use Jeff and pseudoinverse diagnostics",
      "Next: response-scale sweep and Schur/non-normal mode analysis",
    ].join("\n"),
    { fontSize: 23 },
  );
  presentation.resolve("sh/1ojy10ne").position = { left: 43.45, top: 145, width: 620, height: 370 };
  slide16.charts.add("line", {
    position: { left: 770, top: 168, width: 310, height: 245 },
    categories: sweepX.map((x) => `${x}x`),
    series: [{ name: "rho", values: sweepY, stroke: { fill: green, width: 3 }, fill: "accent1" }],
    hasLegend: false,
    yAxis: { majorGridlines: { style: "solid", fill: "#D9D9D9", width: 1 } },
  });
  addMetric(slide16, "E <- E weighted sensitivity", fmt(sensitivity.find((row) => row.receiver_group === "E" && row.sender_group === "E").sum_weighted_sensitivity), { left: 775, top: 455, width: 280, height: 90 }, lightYellow);
  addSources(
    slide16,
    "The chain sweep perturbs the empirical Jeff chain component deterministically. The crossing between 0.50x and 0.75x supports the stability-boundary interpretation.",
    [
      "outputs/mij_paper_replication/chain_enrichment_sweep.csv",
      "outputs/mij_paper_replication/dominant_eigenvalue_block_sensitivity.csv",
      "outputs/mij_paper_replication/population_responses.csv",
    ],
  );

  const after = await presentation.inspect({
    kind: "slide,textbox,shape,image,chart,notes",
    include: "id,slide,name,title,textPreview,bbox,chartType",
    maxChars: 30000,
  });
  await fs.writeFile(`${OUT_DIR}/final-inspect.ndjson`, after.ndjson);

  const montage = await presentation.export({ format: "webp", montage: true, scale: 1 });
  await fs.writeFile(`${OUT_DIR}/final-montage.webp`, new Uint8Array(await montage.arrayBuffer()));
  const firstSlide = presentation.slides.getItem(10);
  const lastSlide = presentation.slides.getItem(15);
  for (const [name, slide] of [["slide-11", firstSlide], ["slide-16", lastSlide]]) {
    const png = await presentation.export({ slide, format: "png", scale: 1 });
    await fs.writeFile(`${OUT_DIR}/${name}.png`, new Uint8Array(await png.arrayBuffer()));
  }

  const pptx = await PresentationFile.exportPptx(presentation);
  await pptx.save(FINAL_PPTX);
  console.log(FINAL_PPTX);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
