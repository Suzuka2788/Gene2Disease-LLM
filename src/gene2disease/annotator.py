"""The click-only annotation page for the evidence assessment.

One item at a time; every answer is a click (codes, subcodes, undecided reasons,
full-text consultation); time on each item is measured automatically and paused while
the tab is hidden; progress is kept in the browser; Export writes the labels CSV in the
exact format that evidence.read_labels checks. The page carries only what the annotator
may see: item id, title, section, the gene and disease names and the passage.
"""
from __future__ import annotations

import json

PAGE = r"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--ink:#1f2933;--line:#cbd5e1;--panel:#f8fafc;--blue:#0b5cad;--gene:#dcebfa;--dis:#fbe3d6;--cite:#fff3b0;--ok:#1b7f4b;--warn:#b4233c}
*{box-sizing:border-box}body{margin:0;font-family:system-ui,"Segoe UI","Microsoft YaHei",sans-serif;color:var(--ink);background:#fff;line-height:1.6}
header{position:sticky;top:0;background:#fff;border-bottom:1px solid var(--line);padding:10px 16px;display:flex;gap:12px;align-items:center;flex-wrap:wrap;z-index:2}
header b{font-size:1.05em}.bar{flex:1;min-width:120px;height:8px;background:var(--panel);border:1px solid var(--line);border-radius:4px;overflow:hidden}
.bar i{display:block;height:100%;background:var(--ok);width:0}main{max-width:900px;margin:0 auto;padding:16px}
.meta{font-size:.92em;background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:8px 12px}
.passage{border:1px solid var(--line);border-radius:6px;padding:12px 14px;margin:12px 0}
.gene{background:var(--gene);border-bottom:2px solid var(--blue)}.disease{background:var(--dis);border-bottom:2px solid #c2410c}.cited{background:var(--cite)}
.q{border:1px solid var(--line);border-radius:8px;padding:10px 14px;margin:14px 0}.q h3{margin:4px 0 8px;font-size:1em}
.opt{display:block;padding:7px 10px;margin:4px 0;border:1px solid var(--line);border-radius:6px;cursor:pointer}
.opt:hover{background:var(--panel)}.opt input{margin-right:8px}.opt small{color:var(--ink);opacity:.8}
.sub{margin:4px 0 6px 28px}.sub .opt{display:inline-block;margin:3px 6px 3px 0;padding:4px 10px}
.nav{display:flex;gap:8px;justify-content:space-between;margin:18px 0;flex-wrap:wrap}
button{font:inherit;padding:8px 16px;border-radius:6px;border:1px solid var(--blue);background:#fff;color:var(--blue);cursor:pointer}
button.primary{background:var(--blue);color:#fff}button:disabled{opacity:.4;cursor:default}
.help{font-size:.9em;border-left:3px solid var(--blue);padding:4px 12px;margin:8px 0;background:var(--panel)}
.status{font-size:.9em}.missing{color:var(--warn)}.done{color:var(--ok)}
.jump{display:flex;flex-wrap:wrap;gap:4px;margin:8px 0}.jump button{padding:2px 8px;font-size:.85em}
.jump button.complete{background:var(--ok);border-color:var(--ok);color:#fff}
</style></head><body>
<header><b>__TITLE__</b><span class="status" id="count"></span><div class="bar"><i id="bar"></i></div>
<button id="export" class="primary">导出 CSV</button></header>
<main>
<div class="help" id="help"></div>
<div class="jump" id="jump"></div>
<div id="item"></div>
<div class="nav"><button id="prev">← 上一条</button><span class="status" id="itemstatus"></span><button id="next" class="primary">下一条 →</button></div>
</main>
<script>
const STAGE = __STAGE__, ITEMS = __ITEMS__, STORE = __STORE__, FILENAME = __FILENAME__;
const CODES = [
  ["supported", "支持", "文本明确陈述了这个基因和这个疾病的关系（BioRED 口径，不要求同一句）"],
  ["speculative", "仅推测", "只有推测性的表述（may、might、suggest 等），没有明确陈述"],
  ["negative", "否定", "只报告没有关系（不显著、未观察到相关）"],
  ["no_relation", "无关系", "没有陈述二者的关系"],
  ["undecided", "无法判断", "需要选一个原因"]];
const SUBCODES = [["co_mention", "只是共现"], ["prognostic", "只作预后因子"], ["symptom", "只与症状有关"],
  ["wrong_entity", "不是给定的基因或疾病"], ["other", "其他"]];
const REASONS = ["指代不清，全文也确认不了", "段落不完整或被截断", "需要的专业知识超出判断范围", "表述含糊，介于支持和推测之间"];
const MENTIONS_NOTE = "蓝色和橙色只标出语料标注过的提及；没有高亮的同义说法、缩写或代词也可能指同一个基因或疾病，是不是同一个，可以根据上下文判断（这属于确认指代）。";
const HELP = STAGE === 1
  ? "<b>问题 1：</b>这个段落是否支持高亮的基因和疾病之间的关系？有多种表述时，按 支持 → 仅推测 → 否定 的顺序取第一个符合的。全文只用来确认指代（缩写、代词、是不是给定的实体），每条不超过 3 分钟。" + MENTIONS_NOTE + "快捷键：1–5 选择，← → 翻页。"
  : "<b>问题 2：</b>每一组<span class='cited'>黄色的引用句</span>本身是否支持这个关系？段落其余部分只用来确认指代。" + MENTIONS_NOTE + "快捷键：← → 翻页。";
let state = load(), index = state.index || 0, shownAt = null;

function load() { try { return JSON.parse(localStorage.getItem(STORE)) || {answers: {}, ms: {}}; } catch (e) { return {answers: {}, ms: {}}; } }
function save() { state.index = index; try { localStorage.setItem(STORE, JSON.stringify(state)); } catch (e) {} }
function units(item) { return STAGE === 1 ? [item.item_id] : item.sets.map(s => s.set_id); }
function answer(unit) { return state.answers[unit] || (state.answers[unit] = {}); }
function unitComplete(unit) {
  const a = state.answers[unit] || {};
  if (!a.code) return false;
  if (a.code === "no_relation" && !a.subcode) return false;
  if (a.code === "undecided" && !a.reason) return false;
  return true;
}
function itemComplete(item) { return units(item).every(unitComplete); }
function tick() { if (shownAt !== null) { const id = ITEMS[index].item_id; state.ms[id] = (state.ms[id] || 0) + (Date.now() - shownAt); shownAt = Date.now(); save(); } }
document.addEventListener("visibilitychange", () => { if (document.hidden) { tick(); shownAt = null; } else { shownAt = Date.now(); } });
window.addEventListener("beforeunload", tick);

function question(unit, title, passage) {
  const a = answer(unit);
  let html = `<div class="q" data-unit="${unit}"><h3>${title}</h3>`;
  if (passage) html += `<div class="passage">${passage}</div>`;
  CODES.forEach(([code, label, desc], k) => {
    html += `<label class="opt"><input type="radio" name="code-${unit}" value="${code}" ${a.code === code ? "checked" : ""}>${k + 1}. <b>${label}</b> <small>${desc}</small></label>`;
    if (code === "no_relation" && a.code === "no_relation") {
      html += `<div class="sub">` + SUBCODES.map(([s, l]) => `<label class="opt"><input type="radio" name="sub-${unit}" value="${s}" ${a.subcode === s ? "checked" : ""}>${l}</label>`).join("") + `</div>`;
    }
    if (code === "undecided" && a.code === "undecided") {
      html += `<div class="sub">` + REASONS.map(r => `<label class="opt"><input type="radio" name="reason-${unit}" value="${r}" ${a.reason === r ? "checked" : ""}>${r}</label>`).join("") + `</div>`;
    }
  });
  return html + `</div>`;
}

function render() {
  const item = ITEMS[index], first = units(item)[0];
  let html = `<h2>${item.item_id}</h2><div class="meta">${item.header}</div>`;
  if (STAGE === 1) {
    html += `<div class="passage">${item.passage}</div>` + question(item.item_id, "这个段落是否支持这个关系？", "");
    const a = answer(first);
    html += `<label class="opt"><input type="checkbox" id="fulltext" ${a.fulltext === "yes" ? "checked" : ""}>我打开了全文（只用于确认指代）</label>`;
  } else {
    item.sets.forEach((s, k) => { html += question(s.set_id, `引用组 ${String.fromCharCode(65 + k)}：这些高亮的句子本身是否支持这个关系？`, s.passage); });
  }
  document.getElementById("item").innerHTML = html;
  document.querySelectorAll("#item input[type=radio]").forEach(input => input.addEventListener("change", onChoose));
  const box = document.getElementById("fulltext");
  if (box) box.addEventListener("change", () => { answer(first).fulltext = box.checked ? "yes" : "no"; save(); });
  document.getElementById("prev").disabled = index === 0;
  document.getElementById("next").textContent = index === ITEMS.length - 1 ? "最后一条" : "下一条 →";
  status();
  window.scrollTo(0, 0);
}

function onChoose(event) {
  const input = event.target, unit = input.closest(".q").dataset.unit, a = answer(unit);
  if (input.name.startsWith("code-")) { a.code = input.value; if (a.code !== "no_relation") delete a.subcode; if (a.code !== "undecided") delete a.reason; render(); }
  else if (input.name.startsWith("sub-")) { a.subcode = input.value; }
  else if (input.name.startsWith("reason-")) { a.reason = input.value; }
  save(); status();
}

function status() {
  const done = ITEMS.filter(itemComplete).length;
  document.getElementById("count").textContent = `已完成 ${done} / ${ITEMS.length}`;
  document.getElementById("bar").style.width = `${100 * done / ITEMS.length}%`;
  const item = ITEMS[index];
  const el = document.getElementById("itemstatus");
  el.textContent = itemComplete(item) ? "这一条已完成" : "这一条还没选完";
  el.className = "status " + (itemComplete(item) ? "done" : "missing");
  document.getElementById("jump").innerHTML = ITEMS.map((it, k) => `<button data-k="${k}" class="${itemComplete(it) ? "complete" : ""}">${it.item_id}</button>`).join("");
  document.querySelectorAll("#jump button").forEach(b => b.addEventListener("click", () => go(+b.dataset.k)));
}

function go(k) { if (k < 0 || k >= ITEMS.length) return; tick(); index = k; shownAt = Date.now(); save(); render(); }
document.getElementById("prev").addEventListener("click", () => go(index - 1));
document.getElementById("next").addEventListener("click", () => go(index + 1));
document.addEventListener("keydown", event => {
  if (event.target.tagName === "TEXTAREA") return;
  if (event.key === "ArrowRight") go(index + 1);
  else if (event.key === "ArrowLeft") go(index - 1);
  else if (STAGE === 1 && /^[1-5]$/.test(event.key)) {
    const input = document.querySelector(`input[name="code-${ITEMS[index].item_id}"][value="${CODES[+event.key - 1][0]}"]`);
    if (input) { input.checked = true; input.dispatchEvent(new Event("change")); }
  }
});

function minutes(id) { return Math.max(0.1, Math.round((state.ms[id] || 0) / 6000) / 10).toFixed(1); }
function cell(value) { return '"' + String(value ?? "").replace(/"/g, '""') + '"'; }
document.getElementById("export").addEventListener("click", () => {
  tick();
  const missing = ITEMS.filter(it => !itemComplete(it)).map(it => it.item_id);
  if (missing.length) { alert("还有没选完的条目：" + missing.join(", ")); go(ITEMS.findIndex(it => !itemComplete(it))); return; }
  let rows;
  if (STAGE === 1) {
    rows = [["item_id", "q1", "subcode", "undecided_reason", "fulltext_consulted", "minutes", "notes"]];
    ITEMS.forEach(it => { const a = state.answers[it.item_id]; rows.push([it.item_id, a.code, a.subcode || "", a.reason || "", a.fulltext === "yes" ? "yes" : "no", minutes(it.item_id), ""]); });
  } else {
    rows = [["item_id", "set_id", "q2", "subcode", "undecided_reason", "minutes", "notes"]];
    ITEMS.forEach(it => it.sets.forEach((s, k) => { const a = state.answers[s.set_id]; rows.push([it.item_id, s.set_id, a.code, a.subcode || "", a.reason || "", k === 0 ? minutes(it.item_id) : "", ""]); }));
  }
  const text = "﻿" + rows.map(r => r.map(cell).join(",")).join("\r\n") + "\r\n";
  const link = document.createElement("a");
  link.href = URL.createObjectURL(new Blob([text], {type: "text/csv;charset=utf-8"}));
  link.download = FILENAME; link.click();
});

document.getElementById("help").innerHTML = HELP;
shownAt = Date.now();
render();
</script></body></html>
"""


def page(title: str, stage: int, items: list[dict], store_key: str, filename: str) -> str:
    """`items`: stage 1 dicts with item_id, header, passage; stage 2 dicts with
    item_id, header and sets (set_id, passage)."""
    values = {"__TITLE__": title, "__STAGE__": json.dumps(stage),
              "__ITEMS__": json.dumps(items, ensure_ascii=False).replace("</", "<\\/"),
              "__STORE__": json.dumps(store_key), "__FILENAME__": json.dumps(filename)}
    text = PAGE
    for placeholder, value in values.items():
        text = text.replace(placeholder, value)
    return text
