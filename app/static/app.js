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

$$(".nav").forEach(b=>b.onclick=()=>{
  $$(".nav").forEach(x=>x.classList.toggle("active",x===b));
  $$(".page").forEach(x=>x.classList.add("hidden"));
  $("#"+b.dataset.page).classList.remove("hidden");
  if(b.dataset.page==="data") loadSample();
});

function renderPicker(cols){
  $("#columnPicker").innerHTML=cols.map(c=>`
    <label class="pick ${selected.has(c)?"selected":""}">
      <input type="checkbox" value="${c}" ${selected.has(c)?"checked":""}/>
      ${c}
    </label>`).join("");
  $$("#columnPicker input").forEach(i=>i.onchange=()=>{
    if(i.checked){
      selected.add(i.value)
    }else{
      selected.delete(i.value)
    }
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
    let url=`/api/analyze?columns=${encodeURIComponent(cols)}`;
    if(k) url+=`&k=${k}`;
    const a=await jsonFetch(url);
    analysisData=a;
    renderAnalysis(a);
  }catch(e){showError(e.message)}
  finally{setLoading(false)}
}

function renderAnalysis(a){
  $("#resultArea").classList.remove("hidden");
  $("#uniqueCfg").textContent=fmt(a.unique_configurations);
  $("#fromCount").textContent=fmt(a.unique_configurations);
  $("#recommendedK").textContent=a.recommended_k;
  $("#customK").max=Math.max(2,a.unique_configurations-1);
  $("#customK").value=a.selected_k;
  $("#recommendedBtn").onclick=()=>analyze(a.recommended_k);

  $("#curve").innerHTML=a.curve.map(p=>{
    const h=Math.max(5,Math.round(p.score*160));
    return `<div class="barwrap">
      <div class="score">${p.score.toFixed(3)}</div>
      <div class="bar ${p.k===a.recommended_k?"best":""}" style="height:${h}px"></div>
      <div class="barlabel">k=${p.k}</div>
    </div>`
  }).join("");

  $("#groupSubtitle").textContent=`Showing ${a.selected_k} groups for: ${a.columns.join(" + ")}`;
  $("#groups").innerHTML=a.groups.map(g=>`
    <div class="group">
      <div class="grouphead">
        <b>Group ${g.group}</b>
        <div class="canon">${Object.entries(g.canonical).map(([k,v])=>`<span class="chip">${k}: ${v}</span>`).join("")}</div>
        <div class="count">${g.configuration_count} configs</div>
        <div class="count">${fmt(g.node_count)} nodes</div>
      </div>
      <div class="members">
        ${g.members.map(m=>`
          <div class="member">
            <div class="vals">${Object.entries(m.values).map(([k,v])=>`<span class="val">${k}: ${v}</span>`).join("")}</div>
            <div>${fmt(m.node_count)} nodes</div>
          </div>`).join("")}
      </div>
    </div>`).join("");
}

$("#analyzeBtn").onclick=()=>analyze();
$$("[data-k]").forEach(b=>b.onclick=()=>analyze(parseInt(b.dataset.k)));
$("#customBtn").onclick=()=>analyze(parseInt($("#customK").value));



async function loadSample(){
  if($("#sampleBody").children.length) return;
  try{
    const d=await jsonFetch("/api/sample");
    if(!d.rows.length)return;
    const cols=Object.keys(d.rows[0]);
    $("#sampleHead").innerHTML="<tr>"+cols.map(c=>`<th>${c}</th>`).join("")+"</tr>";
    $("#sampleBody").innerHTML=d.rows.map(r=>"<tr>"+cols.map(c=>`<td>${r[c]??""}</td>`).join("")+"</tr>").join("");
  }catch(e){showError(e.message)}
}

loadStatus();
