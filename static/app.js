"use strict";

const $ = (id) => document.getElementById(id);

function num(value, digits = 2) {
  if (value === null || value === undefined) return "—";
  return Number(value).toFixed(digits);
}

function metric(label, value, suffix = "", good = false) {
  return `<div class="metric">
    <div class="metric-label">${label}</div>
    <div class="metric-value${good ? " good" : ""}">${value}${suffix}</div>
  </div>`;
}

function renderFlow(flows) {
  const container = $("flow-cards");
  container.innerHTML = "";

  flows.forEach((f) => {
    const card = document.createElement("section");
    card.className = "flow-card";

    const heading = document.createElement("h3");
    heading.innerHTML = `<span class="tag">Video ${f.id}</span> ${f.motion}`;

    const metrics = document.createElement("div");
    metrics.className = "metrics";
    metrics.innerHTML =
      metric("LK mean error", num(f.lk_mean_err, 3), " px", true) +
      metric("LK median error", num(f.lk_median_err, 3), " px") +
      metric("Dense (Farnebäck) MEE", num(f.dense_grid_mee, 3), " px") +
      metric("Corners tracked", f.n_corners, ` @ t=${f.t}`);

    const panels = document.createElement("div");
    panels.className = "flow-panels";

    // colour-wheel flow animation (looping GIF) + full-resolution download
    const animFig = document.createElement("figure");
    animFig.className = "panel";
    const anim = document.createElement("img");
    anim.src = f.flow_gif;
    anim.alt = `optical flow visualization, video ${f.id}`;
    anim.loading = "lazy";
    const animCap = document.createElement("figcaption");
    animCap.innerHTML = `flow visualization (colour wheel) · <a href="${f.flow_video}" download>download full mp4</a>`;
    animFig.appendChild(anim);
    animFig.appendChild(animCap);

    // two-frame montage with tracking arrows vs ground truth
    const pairFig = document.createElement("figure");
    pairFig.className = "panel";
    const pair = document.createElement("img");
    pair.src = f.pair_image;
    pair.alt = `tracking validation, video ${f.id}`;
    pair.loading = "lazy";
    const pairCap = document.createElement("figcaption");
    pairCap.textContent = "two frames + dense flow + LK (green) vs ground-truth (red) arrows";
    pairFig.appendChild(pair);
    pairFig.appendChild(pairCap);

    panels.appendChild(animFig);
    panels.appendChild(pairFig);

    card.appendChild(heading);
    card.appendChild(metrics);
    card.appendChild(panels);
    container.appendChild(card);
  });
}

function renderSfm(sfm) {
  // metrics
  $("s-homography").textContent = num(sfm.cameras.reduce((a, c) => a + c.homography_rel_err, 0) / sfm.cameras.length, 6);
  $("s-planar").textContent = num(sfm.planar_mean_err, 3) + " mm";
  $("s-tri").textContent = num(sfm.tri_mean_err, 3) + " mm";
  $("s-area").textContent = `${num(sfm.boundary_area_rec, 0)} / ${num(sfm.boundary_area_true, 0)} mm²`;

  // intrinsics
  $("s-k").textContent = sfm.K
    .map((row) => "  " + row.map((v) => Number(v).toFixed(1)).join("  "))
    .join("\n");

  // camera table
  let table = "<thead><tr><th>View</th><th>Camera centre C (mm)</th><th>Homography rel. error</th></tr></thead><tbody>";
  sfm.cameras.forEach((c) => {
    const C = c.C.map((v) => Number(v).toFixed(1)).join(", ");
    table += `<tr><td>${c.view}</td><td>(${C})</td><td>${num(c.homography_rel_err, 6)}</td></tr>`;
  });
  table += "</tbody></table>";
  $("s-cameras").innerHTML = table;

  // viewpoint images
  const strip = $("sfm-views");
  strip.innerHTML = "";
  sfm.views.forEach((src, i) => {
    const fig = document.createElement("figure");
    const img = document.createElement("img");
    img.src = src;
    img.alt = `viewpoint ${i + 1}`;
    img.loading = "lazy";
    const cap = document.createElement("figcaption");
    cap.textContent = `view ${i + 1} · C = (${sfm.cameras[i].C.map((v) => Number(v).toFixed(0)).join(", ")})`;
    fig.appendChild(img);
    fig.appendChild(cap);
    strip.appendChild(fig);
  });

  $("sfm-reconstruction").src = sfm.reconstruction;
}

async function loadResults() {
  try {
    const resp = await fetch("/api/results");
    if (!resp.ok) throw new Error("HTTP " + resp.status);
    const data = await resp.json();
    if (data.error) throw new Error(data.error);

    renderFlow(data.optical_flow);
    renderSfm(data.sfm);

    $("flow").hidden = false;
    $("sfm").hidden = false;
  } catch (err) {
    const el = $("error");
    el.textContent = "Error: " + err.message;
    el.hidden = false;
  }
}

loadResults();
