
const $=s=>document.querySelector(s), $$=s=>[...document.querySelectorAll(s)];
let statusData=null, analysisData=null;
let selected=new Set(["site","cluster","cpu_model"]);

async function jsonFetch(url,opts={}){
  const r=await fetch(url,opts);
  if(!r.ok) throw new Error(await r.text());
  return r.json();
}
function fmt(n){return new Intl.NumberFormat().format(n??0)}
function setLoading(x){$("#loading").classList.toggle("hidden",!x)}
function showError(e){$("#error").textContent=e;$("#error").classList.remove("hidden")}
function clearError(){$("#error").classList.add("hidden")}

function renderPicker(cols){
  $("#columnPicker").innerHTML=cols.map(c=>`
    <label class="pick ${selected.has(c)?"selected":""}">
      <input type="checkbox" value="${c}" ${selected.has(c)?"checked":""}/>
      ${c}
    </label>`).join("");
  $$("#columnPicker input").forEach(i=>i.onchange=()=>{
    if(i.checked) selected.add(i.value);
    else selected.delete(i.value);
    renderPicker(cols);
  });
}

async function loadStatus(){
  clearError();setLoading(true);
  try{
    const s=await jsonFetch("/api/status");
    if(!s.ok) throw new Error(s.error);
    statusData=s;
    $("#nodes").textContent=fmt(s.rows);
    $("#sites").textContent=fmt(s.sites);
    $("#clusters").textContent=fmt(s.clusters);
    renderPicker(s.columns);
  }catch(e){showError(e.message)}
  finally{setLoading(false)}
}

async function analyze(k=null){
  if(selected.size<2){showError("Select at least 2 columns.");return}
  clearError();setLoading(true);
  try{
    const cols=[...selected].join(",");
    const alg=$("#algorithmSelect").value;
    let url=`/api/analyze?columns=${encodeURIComponent(cols)}&algorithm=${encodeURIComponent(alg)}`;
    if(k) url+=`&k=${k}`;
    const a=await jsonFetch(url);
    analysisData=a;
    renderAnalysis(a);
  }catch(e){showError(e.message)}
  finally{setLoading(false)}
}

function pct(x){return `${(100*(x??0)).toFixed(1)}%`}

function renderComparison(rows){
  $("#comparisonBody").innerHTML=rows.map(r=>`
    <tr>
      <td><b>${r.algorithm}</b></td>
      <td>${r.metrics.k}</td>
      <td>${r.metrics.silhouette.toFixed(3)}</td>
      <td>${pct(r.metrics.compression)}</td>
      <td>${pct(r.metrics.outlier_rate)}</td>
      <td><b>${r.metrics.score.toFixed(3)}</b></td>
    </tr>`).join("");
}

function renderCurve(a){
  if(!a.curve || !a.curve.length){
    $("#curve").innerHTML=`<div class="muted-inline">This method chooses groups automatically; no fixed-k curve.</div>`;
    return;
  }
  const maxScore=Math.max(...a.curve.map(p=>p.score),0.001);
  $("#curve").innerHTML=a.curve.map(p=>{
    const h=Math.max(5,Math.round((p.score/maxScore)*150));
    const label=p.k ? `k=${p.k}` : `mcs=${p.min_cluster_size}`;
    return `<div class="barwrap">
      <div class="score">${p.score.toFixed(3)}</div>
      <div class="bar" style="height:${h}px"></div>
      <div class="barlabel">${label}</div>
    </div>`
  }).join("");
}

function clusterClass(c){
  if(c===-1) return "noise";
  return `c${Math.abs(c)%10}`;
}
function renderPlot(points){
  const svg=$("#clusterPlot");
  svg.innerHTML=`
    <rect x="0" y="0" width="900" height="360" fill="white"/>
    ${points.map(p=>{
      const x=28+p.x*844;
      const y=28+(1-p.y)*304;
      return `<circle cx="${x}" cy="${y}" r="5" class="pt ${clusterClass(p.cluster)}"><title>Cluster ${p.cluster===-1?"Outlier":p.cluster+1}</title></circle>`;
    }).join("")}
  `;
}

function renderGroups(a){
  $("#groupSubtitle").textContent=`${a.selected_algorithm} on: ${a.columns.join(" + ")}`;
  $("#groups").innerHTML=a.groups.map(g=>`
    <div class="group">
      <div class="grouphead">
        <b>${g.group==="Outliers"?"Outliers":`Group ${g.group}`}</b>
        <div class="canon">${g.canonical ? Object.entries(g.canonical).map(([k,v])=>`<span class="chip">${k}: ${v}</span>`).join("") : ""}</div>
        <div class="count">${g.configuration_count} configs</div>
        <div class="count">${fmt(g.node_count)} nodes</div>
      </div>
      <div class="cluster-explain">${g.explanation}</div>
      <div class="members">
        ${g.members.map(m=>`
          <div class="member">
            <div class="vals">${Object.entries(m.values).map(([k,v])=>`<span class="val">${k}: ${v}</span>`).join("")}</div>
            <div>${fmt(m.node_count)} nodes</div>
          </div>`).join("")}
      </div>
    </div>`).join("");
}

function renderAnalysis(a){
  $("#resultArea").classList.remove("hidden");
  $("#uniqueCfg").textContent=fmt(a.unique_configurations);
  $("#fromCount").textContent=fmt(a.unique_configurations);
  $("#selectedAlgorithm").textContent=a.selected_algorithm;
  $("#selectedK").textContent=a.selected_k;
  $("#consensusText").textContent=`${a.consensus.agreement}. Votes: ${a.consensus.votes.join(", ")}`;
  $("#selectedMetrics").textContent=`Silhouette ${a.metrics.silhouette.toFixed(3)} · Compression ${pct(a.metrics.compression)} · Outliers ${pct(a.metrics.outlier_rate)} · Score ${a.metrics.score.toFixed(3)}`;
  renderComparison(a.comparison);
  renderCurve(a);
  renderPlot(a.projection);
  renderGroups(a);
}

$("#analyzeBtn").onclick=()=>analyze();
$("#algorithmSelect").onchange=()=>{ if(!$("#resultArea").classList.contains("hidden")) analyze(); };

loadStatus();
