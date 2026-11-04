// Escapes text before it is inserted with innerHTML. Report fields include LLM-generated
// descriptions and exchange data, which must never be interpreted as markup.
function escapeHtml(value) {
  return String(value ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

// Formats a report parameter; nested settings (e.g. the Step 1 feature list) are shown as JSON.
function formatParam(value) {
  return value !== null && typeof value === 'object' ? JSON.stringify(value) : value;
}

// Step 2 rules whose inputs a company does not report are null ("not applicable").
function isStep2Evaluable(d) {
  const diag = d.step2_model && d.step2_model.feature_diagnostics;
  return Boolean(diag) && diag.Evaluable !== false;
}

// Number formatting for fields that are missing in older reports or for unranked stocks.
function fmtNumber(value, digits) {
  return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : 'N/A';
}

function fmtPercent(share, digits) {
  return typeof share === 'number' && Number.isFinite(share) ? `${(share * 100).toFixed(digits)}%` : 'N/A';
}

function isPlainObject(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

document.addEventListener('DOMContentLoaded', () => {
  let reportData = [];
  let executionDate = null;
  let parameters = {};
  // Report-level summary of the pooled Step 1 model; absent in reports before 3.0.0
  let step1Model = null;

  // Best-ranked stocks first; stocks without a rank (not eligible, older reports) last
  let currentSort = { column: 'rank', asc: true };

  const tableBodyOther = document.getElementById('table-body');
  const tableBodyBuys = document.getElementById('buys-table-body');
  const searchInput = document.getElementById('search-input');
  const filterSelect = document.getElementById('filter-status');
  const ths = document.querySelectorAll('th[data-sort]');

  const kpiTotal = document.getElementById('kpi-total');
  const kpiBuys = document.getElementById('kpi-buys');
  const kpiUp = document.getElementById('kpi-up');
  const kpiUpLabel = document.getElementById('kpi-up-label');
  const kpiNotUp = document.getElementById('kpi-notup');
  const kpiAuc = document.getElementById('kpi-auc');
  const kpiAucDetail = document.getElementById('kpi-auc-detail');
  const kpiFailedStep2 = document.getElementById('kpi-failed-step2');
  const kpiMissingData = document.getElementById('kpi-missing-data');
  const kpiSuppressed = document.getElementById('kpi-suppressed');
  let suppressedCount = 0;
  let suppressedBuys = 0;
  
  const metaDate = document.getElementById('meta-date');
  const metaParams = document.getElementById('meta-params');

  // Determine correct data URL based on environment (local vs GitHub Pages)
  const isGitHubPages = window.location.hostname.includes('github.io');
  const dataUrl = isGitHubPages ? './data/full_batch_report.json' : '../data/processed/full_batch_report.json';

  // Fetch data
  fetch(dataUrl)
    .then(response => {
      if (!response.ok) throw new Error("Network response was not ok");
      return response.text();
    })
    .then(text => {
      // Python's json.dump writes float('inf') as Infinity which breaks JSON.parse
      const safeText = text.replace(/(:\s*)(Infinity|-Infinity|NaN)\b/g, '$1"$2"');
      return JSON.parse(safeText);
    })
    .then(data => {
      if (Array.isArray(data)) {
         // Fallback for old data
         reportData = data;
      } else {
         reportData = data.predictions || [];
         executionDate = data.execution_date;
         parameters = data.parameters || {};
         step1Model = isPlainObject(data.step1_model) ? data.step1_model : null;
      }
      
      // Stocks priced at or below min_price_eur (settings.yaml, 10 EUR) are hidden from all KPIs and
      // tables; the uplift evaluation excludes them as well (documented in the README)
      const minPrice = typeof parameters.min_price_eur === 'number' ? parameters.min_price_eur : 10;
      const suppressedLabel = document.getElementById('kpi-suppressed-label');
      if (suppressedLabel) suppressedLabel.textContent = `Suppressed (<= ${minPrice}€)`;
      const isSuppressed = d => typeof d.latest_price === 'number' && d.latest_price <= minPrice;
      suppressedCount = reportData.filter(isSuppressed).length;
      suppressedBuys = reportData.filter(d => isSuppressed(d) && d.final_prediction === 'UP_FINAL_BUY').length;
      reportData = reportData.filter(d => !isSuppressed(d));

      renderMetadata();
      calculateKPIs();
      renderModelPanel(minPrice);
      renderTables();
      renderVariables();
    })
    .catch(error => {
      console.error("Failed to load JSON data", error);
      const errorHtml = `<tr><td colspan="11" style="text-align:center; color: var(--status-red);">Failed to load data. Ensure full_batch_report.json is available.</td></tr>`;
      tableBodyOther.innerHTML = errorHtml;
      tableBodyBuys.innerHTML = errorHtml;
    });

  const upliftDataUrl = isGitHubPages ? './data/uplift_report.json' : '../data/processed/uplift_report.json';
  fetch(upliftDataUrl)
    .then(r => r.json())
    .then(upliftData => {
      renderUpliftChart(upliftData);
    })
    .catch(err => {
      console.warn("Failed to load uplift report", err);
    });

  function renderUpliftChart(data) {
    const keys = ["1m", "3m", "6m"];
    const labelsMap = { "1m": "1 Month", "3m": "3 Months", "6m": "6 Months" };
    
    const metaParts = [];
    keys.forEach(k => {
      if (data[k]) {
        let dateStr = data[k].report_date || new Date(data[k].date).toLocaleDateString();
        let dummyFlag = data[k].is_dummy ? ' <span style="color:var(--status-orange)">(Dummy Data)</span>' : '';
        metaParts.push(`<strong>${k.toUpperCase()}</strong>: ${escapeHtml(dateStr)}${dummyFlag}`);
      }
    });
    const priceFilter = keys.map(k => data[k] && data[k].min_price_eur).find(v => typeof v === 'number');
    if (priceFilter !== undefined) metaParts.push(`stocks <= ${escapeHtml(priceFilter)}€ excluded`);
    const metaHtml = metaParts.join(' &nbsp;&nbsp;|&nbsp;&nbsp; ');
    const metaEl = document.getElementById('uplift-metadata');
    if (metaEl) metaEl.innerHTML = metaHtml;
    
    keys.forEach(k => {
      const ctx = document.getElementById(`uplift-chart-${k}`);
      if (!ctx) return;
      
      if (!data[k] || !data[k].metrics) {
        ctx.parentElement.innerHTML = `<div style="height: 100%; display: flex; align-items: center; justify-content: center; color: var(--text-muted); text-align: center;">${labelsMap[k]}: not enough history yet</div>`;
        return;
      }
      // Empty cohorts are null and are not drawn (a missing cohort is not a 0% return)
      const pct = v => (typeof v === 'number' ? v * 100 : null);
      const metrics = data[k].metrics;
      const counts = data[k].counts || {};
      const baseline = pct(metrics.baseline);
      const upFinalBuy = pct(metrics.UP_FINAL_BUY);
      const up = pct(metrics.UP);
      const notUp = pct(metrics.NOT_UP);
      const cohortCounts = [counts.NOT_UP, counts.UP, counts.UP_FINAL_BUY];
      
      new Chart(ctx, {
        type: 'bar',
        data: {
          labels: ['NOT_UP', 'UP', 'UP_FINAL_BUY'],
          datasets: [
            {
              type: 'line',
              label: 'Baseline (Market)',
              data: [baseline, baseline, baseline],
              borderColor: 'rgba(255, 255, 255, 0.8)',
              borderWidth: 2,
              borderDash: [5, 5],
              tension: 0,
              fill: false,
              pointRadius: 0,
              pointHoverRadius: 0
            },
            {
              type: 'bar',
              label: 'Cohort Performance',
              data: [notUp, up, upFinalBuy],
              backgroundColor: [
                'rgba(255, 99, 132, 0.5)',
                'rgba(255, 206, 86, 0.5)',
                'rgba(75, 192, 192, 0.8)'
              ],
              borderColor: [
                'rgba(255, 99, 132, 1)',
                'rgba(255, 206, 86, 1)',
                'rgba(75, 192, 192, 1)'
              ],
              borderWidth: 1
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            title: {
              display: true,
              text: labelsMap[k],
              color: '#ffffff',
              font: { size: 14 }
            },
            legend: { 
              labels: { color: '#ffffff' } 
            },
            tooltip: {
              callbacks: {
                label: function(context) {
                  if (context.parsed.y === null) return context.dataset.label + ': n/a';
                  const n = context.dataset.type === 'bar' ? cohortCounts[context.dataIndex]
                                                           : counts.baseline;
                  return context.dataset.label + ': ' + context.parsed.y.toFixed(2) + '%' +
                         (typeof n === 'number' ? ` (n=${n})` : '');
                }
              }
            }
          },
          scales: {
            y: {
              beginAtZero: true,
              ticks: {
                color: '#ffffff',
                callback: function(value) { return value + '%'; }
              },
              grid: { color: 'rgba(255, 255, 255, 0.1)' }
            },
            x: {
              ticks: { color: '#ffffff' },
              grid: { display: false }
            }
          }
        }
      });
    });
  }

  function renderMetadata() {
    if (executionDate) {
      const d = new Date(executionDate);
      metaDate.textContent = d.toLocaleString(undefined, { timeZone: 'UTC' }) + " UTC";
    } else {
      metaDate.textContent = "N/A (Legacy Data)";
    }
    
    metaParams.innerHTML = '';
    for (const [key, value] of Object.entries(parameters)) {
      const badge = document.createElement('div');
      badge.className = 'param-badge';
      badge.innerHTML = `<strong>${escapeHtml(key)}:</strong> ${escapeHtml(formatParam(value))}`;
      metaParams.appendChild(badge);
    }

    const shareLabel = document.getElementById('up-share-label');
    if (shareLabel && typeof upShare() === 'number') shareLabel.textContent = fmtPercent(upShare(), 0);
  }

  // Share of the ranking marked UP: from the model summary, else from the parameters
  function upShare() {
    const share = step1Model?.cross_section?.up_share ?? parameters.step1_up_share;
    return typeof share === 'number' ? share : null;
  }

  function calculateKPIs() {
    kpiTotal.textContent = reportData.length;

    const buys = reportData.filter(d => d.final_prediction === 'UP_FINAL_BUY').length;
    kpiBuys.textContent = buys;

    // Step 2 only runs for Step 1 UP stocks, so UP and UP_FINAL_BUY together are the Step 1 passes
    const step1Up = reportData.filter(d => d.final_prediction === 'UP' || d.final_prediction === 'UP_FINAL_BUY').length;
    kpiUp.textContent = step1Up;
    if (kpiUpLabel && typeof upShare() === 'number') kpiUpLabel.textContent = `Step 1 UP (Top ${fmtPercent(upShare(), 0)})`;
    
    const notUp = reportData.filter(d => d.final_prediction === 'NOT_UP').length;
    kpiNotUp.textContent = notUp;

    const failedStep2 = reportData.filter(d => d.final_prediction === 'UP' && isStep2Evaluable(d)).length;
    if (kpiFailedStep2) kpiFailedStep2.textContent = failedStep2;

    const missingData = reportData.filter(d => d.final_prediction === 'UP' && !isStep2Evaluable(d)).length;
    kpiMissingData.textContent = missingData;

    if (kpiSuppressed) {
      kpiSuppressed.textContent = suppressedBuys > 0 ? `${suppressedCount} (${suppressedBuys} buys)` : suppressedCount;
    }

    // Walk-forward validation of the pooled model: mean per-date cross-sectional AUC
    const validation = step1Model?.validation;
    const gate = step1Model?.gate;
    if (kpiAuc) kpiAuc.textContent = fmtNumber(validation?.auc_mean, 3);
    if (kpiAucDetail) {
      const ci = Array.isArray(validation?.auc_ci) ? validation.auc_ci : null;
      const ciText = ci ? `95% CI ${fmtNumber(ci[0], 3)}-${fmtNumber(ci[1], 3)}` : 'CI N/A';
      const gateText = typeof gate?.passed === 'boolean'
        ? (gate.passed ? 'gate PASS' : `gate FAIL (${gate.reason || 'unknown'})`)
        : 'gate N/A';
      kpiAucDetail.textContent = step1Model ? `${ciText} | ${gateText}` : 'No model summary in this report';
    }
  }

  // Model-level Step 1 results: coefficients, validation per year and the cross-section
  function renderModelPanel(minDisplayPrice) {
    const container = document.getElementById('model-panel');
    if (!container) return;
    if (!step1Model) {
      container.innerHTML = '<div style="color: var(--text-muted);">No model summary in this report (reports before release 3.0.0 have none).</div>';
      return;
    }
    const cross = isPlainObject(step1Model.cross_section) ? step1Model.cross_section : {};
    const validation = isPlainObject(step1Model.validation) ? step1Model.validation : {};
    const gate = isPlainObject(step1Model.gate) ? step1Model.gate : {};
    const ci = Array.isArray(validation.auc_ci) ? validation.auc_ci : [null, null];
    const gateText = typeof gate.passed === 'boolean'
      ? (gate.passed ? 'PASS' : `FAIL (${gate.reason || 'unknown'})`) : 'N/A';
    const eligiblePrice = typeof parameters.step1_min_price_eur === 'number' ? parameters.step1_min_price_eur : 1;
    const notEligible = isPlainObject(cross.not_eligible)
      ? Object.entries(cross.not_eligible).filter(([, n]) => n > 0).map(([k, n]) => `${k}: ${n}`).join(', ') : '';

    const facts = [
      ['Scoring date', step1Model.scoring_date || 'N/A'],
      ['Ranked stocks', typeof cross.n_ranked === 'number' ? `${cross.n_ranked} of ${cross.n_universe ?? 'N/A'}` : 'N/A'],
      ['UP stocks', typeof cross.n_up === 'number' ? `${cross.n_predicted_up ?? 'N/A'} (top share ${fmtPercent(cross.up_share, 0)} = ${cross.n_up})` : 'N/A'],
      ['Validation AUC (mean per date)', `${fmtNumber(validation.auc_mean, 3)} (95% CI ${fmtNumber(ci[0], 3)}-${fmtNumber(ci[1], 3)}, ${validation.n_dates ?? 'N/A'} dates)`],
      ['Model gate', `${gateText}; lower CI bound must exceed ${fmtNumber(gate.threshold, 2)}`],
      ['Dates with AUC > 0.5', fmtPercent(validation.share_dates_above_half, 0)],
      ['Top-share hit rate vs. base rate', `${fmtPercent(validation.top_hit_rate, 1)} vs. ${fmtPercent(validation.base_rate, 1)}`],
      ['Top share minus universe (forward return)', fmtPercent(validation.top_minus_universe_fwd, 1)],
      ['In-sample vs. out-of-sample AUC', `${fmtNumber(validation.is_auc_mean, 3)} vs. ${fmtNumber(validation.oos_auc_mean, 3)}`],
      ['Not eligible', notEligible || 'none'],
    ];
    const factsHtml = facts.map(([k, v]) =>
      `<div class="var-item"><span class="var-key">${escapeHtml(k)}</span><span class="var-value">${escapeHtml(v)}</span></div>`).join('');

    const coefficients = isPlainObject(step1Model.coefficients) ? Object.entries(step1Model.coefficients) : [];
    coefficients.sort((a, b) => Math.abs(Number(b[1]) || 0) - Math.abs(Number(a[1]) || 0));
    let coefRows = coefficients.map(([k, v]) => `<tr><td>${escapeHtml(k)}</td><td>${fmtNumber(v, 3)}</td></tr>`).join('');
    coefRows += `<tr><td style="color: var(--text-muted);">intercept</td><td>${fmtNumber(step1Model.intercept, 3)}</td></tr>`;

    const years = Array.isArray(validation.years) ? validation.years : [];
    const yearRows = years.map(y => (y && y.status === 'ok')
      ? `<tr><td>${escapeHtml(y.year)}</td><td>${escapeHtml(y.n_test_dates)}</td><td>${fmtNumber(y.auc_mean, 3)}</td>`
        + `<td>${fmtNumber(y.oos_auc, 3)}</td><td>${fmtNumber(y.is_auc, 3)}</td><td>${fmtPercent(y.top_hit_rate, 1)}</td><td>${fmtPercent(y.base_rate, 1)}</td></tr>`
      : `<tr><td>${escapeHtml(y?.year)}</td><td colspan="6" style="color: var(--text-muted);">${escapeHtml(y?.status || 'N/A')}</td></tr>`).join('');

    container.innerHTML = `
      <p class="model-note">One logistic regression with shared coefficients ranks all eligible stocks of the scoring date
        against each other; the top share is UP if the model gate passes. Ranks are computed over all stocks above
        EUR ${escapeHtml(eligiblePrice)}; the tables below hide stocks at or below EUR ${escapeHtml(minDisplayPrice)}, so the displayed ranks have gaps.</p>
      <div class="var-grid model-facts">${factsHtml}</div>
      <div class="model-tables">
        <div class="model-table-wrapper">
          <h4>Coefficients (rank features, sorted by absolute value)</h4>
          <table class="model-table"><thead><tr><th>Feature</th><th>Coefficient</th></tr></thead><tbody>${coefRows}</tbody></table>
        </div>
        <div class="model-table-wrapper">
          <h4>Walk-forward validation per year</h4>
          <table class="model-table"><thead><tr><th>Year</th><th>Dates</th><th>AUC</th><th>OOS AUC</th><th>IS AUC</th><th>Top hit</th><th>Base</th></tr></thead>
            <tbody>${yearRows || '<tr><td colspan="7">N/A</td></tr>'}</tbody></table>
        </div>
      </div>`;
  }

  function getStatusHtml(status) {
    if (status === 'UP_FINAL_BUY') {
      return `<span class="status-pill status-up-final">UP_FINAL_BUY</span>`;
    } else if (status === 'UP') {
      return `<span class="status-pill status-up">UP</span>`;
    } else {
      return `<span class="status-pill status-not-up">${escapeHtml(status || 'NOT_UP')}</span>`;
    }
  }

  function createRowHtml(d) {
    const stockName = escapeHtml(d.stock_name || 'UNKNOWN');
    const companyName = escapeHtml(d.company_name || '');
    const companyDesc = escapeHtml(d.company_description || '');
    const priceStr = typeof d.latest_price === 'number' ? `€${d.latest_price.toFixed(2)}` : 'N/A';
    const s1 = isPlainObject(d.step1_model) ? d.step1_model : {};
    const step1Class = s1.predicted_class || 'N/A';
    const probability = fmtNumber(s1.probability, 3);
    const rankStr = typeof s1.rank === 'number'
      ? `${s1.rank} / ${typeof s1.n_ranked === 'number' ? s1.n_ranked : 'N/A'}`
      : 'N/A';
    const topPercent = typeof s1.top_percent === 'number' ? `${s1.top_percent.toFixed(1)}%` : 'N/A';
    // Reports before 3.0.0 carry a per-stock cv_status instead of the pooled status
    const statusRaw = typeof s1.status === 'string' ? s1.status : (typeof s1.cv_status === 'string' ? s1.cv_status : '');
    const statusHtml = statusRaw
      ? `<span title="${escapeHtml(statusRaw)}">${escapeHtml(statusRaw.replace('not_eligible: ', 'not eligible: '))}</span>`
      : 'N/A';
    const peRatio = typeof d.pe_ratio === 'number' ? d.pe_ratio.toFixed(2) : escapeHtml(d.pe_ratio || 'N/A');
    const beta = typeof d.beta === 'number' ? d.beta.toFixed(2) : escapeHtml(d.beta || 'N/A');
    const liquidityStr = typeof d.liquidity === 'number' ? d.liquidity.toFixed(2) : escapeHtml(d.liquidity || 'N/A');

    let priceHtml = `<td>${priceStr}</td>`;
    
    let liquidityHtml = `<td>${liquidityStr}</td>`;
    if (typeof d.liquidity === 'number' && d.liquidity < 0.5) {
      liquidityHtml = `<td style="color: var(--status-red); font-weight: bold;" title="Warning: Low liquidity (< €0.5M). High risk of orderbook slippage. Large orders may significantly impact the stock price.">
        ${liquidityStr} [!WARN]
      </td>`;
    }

    return `
      <td class="ticker-name">
        <div style="font-weight: bold;">${stockName}</div>
        ${companyName ? `<div style="font-size: 0.85em; color: var(--text-muted); margin-top: 4px; font-weight: normal;">${companyName}</div>` : ''}
        ${companyDesc ? `<div style="font-size: 0.75em; color: var(--text-muted); opacity: 0.8; margin-top: 2px; font-weight: normal; max-width: 300px; white-space: normal;">${companyDesc}</div>` : ''}
      </td>
      ${priceHtml}
      <td>${peRatio}</td>
      <td>${beta}</td>
      ${liquidityHtml}
      <td>${getStatusHtml(d.final_prediction)}</td>
      <td>${probability}</td>
      <td>${escapeHtml(rankStr)}</td>
      <td>${escapeHtml(topPercent)}</td>
      <td>${getStatusHtml(step1Class)}</td>
      <td style="font-size: 0.85em;">${statusHtml}</td>
    `;
  }

  function renderTables() {
    const searchTerm = searchInput.value.toLowerCase();
    const filterStatus = filterSelect.value;

    let filtered = reportData.filter(d => {
      const stockName = d.stock_name || 'UNKNOWN';
      const matchSearch = stockName.toLowerCase().includes(searchTerm);
      const matchStatus = filterStatus === 'ALL' || d.final_prediction === filterStatus;
      return matchSearch && matchStatus;
    });

    // Split into buys and others
    let buysList = filtered.filter(d => d.final_prediction === 'UP_FINAL_BUY');
    let otherList = filtered.filter(d => d.final_prediction !== 'UP_FINAL_BUY');

    // Sorting logic
    const sortFn = (a, b) => {
      let valA, valB;
      const col = currentSort.column;
      
      if (col === 'stock_name') {
        valA = a.stock_name || 'UNKNOWN';
        valB = b.stock_name || 'UNKNOWN';
      } else if (col === 'latest_price') {
        valA = typeof a.latest_price === 'number' ? a.latest_price : -1;
        valB = typeof b.latest_price === 'number' ? b.latest_price : -1;
      } else if (col === 'pe_ratio') {
        valA = typeof a.pe_ratio === 'number' ? a.pe_ratio : -9999;
        valB = typeof b.pe_ratio === 'number' ? b.pe_ratio : -9999;
      } else if (col === 'beta') {
        valA = typeof a.beta === 'number' ? a.beta : -9999;
        valB = typeof b.beta === 'number' ? b.beta : -9999;
      } else if (col === 'liquidity') {
        valA = typeof a.liquidity === 'number' ? a.liquidity : -9999;
        valB = typeof b.liquidity === 'number' ? b.liquidity : -9999;
      } else if (col === 'final_prediction') {
        valA = a.final_prediction || '';
        valB = b.final_prediction || '';
      } else if (col === 'step1_class') {
        valA = a.step1_model?.predicted_class || '';
        valB = b.step1_model?.predicted_class || '';
      } else if (col === 'probability') {
        valA = typeof a.step1_model?.probability === 'number' ? a.step1_model.probability : -1;
        valB = typeof b.step1_model?.probability === 'number' ? b.step1_model.probability : -1;
      } else if (col === 'rank' || col === 'top_percent') {
        // Unranked stocks sort after the ranked ones
        valA = typeof a.step1_model?.[col] === 'number' ? a.step1_model[col] : Infinity;
        valB = typeof b.step1_model?.[col] === 'number' ? b.step1_model[col] : Infinity;
      } else if (col === 'step1_status') {
        valA = a.step1_model?.status ?? a.step1_model?.cv_status ?? '';
        valB = b.step1_model?.status ?? b.step1_model?.cv_status ?? '';
      }

      if (valA < valB) return currentSort.asc ? -1 : 1;
      if (valA > valB) return currentSort.asc ? 1 : -1;
      return 0;
    };

    buysList.sort(sortFn);
    otherList.sort(sortFn);

    // Render Buys Table
    tableBodyBuys.innerHTML = '';
    if (buysList.length === 0) {
      tableBodyBuys.innerHTML = `<tr><td colspan="11" style="text-align:center; color: var(--text-muted);">No confirmed buy candidates match filters.</td></tr>`;
    } else {
      buysList.forEach(d => {
        const tr = document.createElement('tr');
        tr.innerHTML = createRowHtml(d);
        tableBodyBuys.appendChild(tr);
      });
    }

    // Render Other Table
    tableBodyOther.innerHTML = '';
    if (otherList.length === 0) {
      tableBodyOther.innerHTML = `<tr><td colspan="11" style="text-align:center; color: var(--text-muted);">No other stocks match filters.</td></tr>`;
    } else {
      otherList.forEach(d => {
        const tr = document.createElement('tr');
        tr.innerHTML = createRowHtml(d);
        tableBodyOther.appendChild(tr);
      });
    }

    // Update sort icons
    ths.forEach(th => {
      const icon = th.querySelector('.sort-icon');
      if (th.dataset.sort === currentSort.column) {
        icon.textContent = currentSort.asc ? '▲' : '▼';
      } else {
        icon.textContent = '';
      }
    });
  }

  // Event Listeners
  searchInput.addEventListener('input', renderTables);
  filterSelect.addEventListener('change', renderTables);

  ths.forEach(th => {
    th.addEventListener('click', () => {
      const col = th.dataset.sort;
      if (currentSort.column === col) {
        currentSort.asc = !currentSort.asc;
      } else {
        currentSort.column = col;
        currentSort.asc = true; // default asc for new column
      }
      renderTables();
    });
  });

  // Event Listeners for variables
  const varSearchInput = document.getElementById('var-search-input');
  if (varSearchInput) {
    varSearchInput.addEventListener('input', renderVariables);
  }

  function renderVariables() {
    const container = document.getElementById('variables-container');
    if (!container) return;
    
    const search = varSearchInput ? varSearchInput.value.toLowerCase() : '';
    
    let html = '';
    reportData.forEach(d => {
      const rawStockName = d.stock_name || 'UNKNOWN';
      if (search && !rawStockName.toLowerCase().includes(search)) return;
      const stockName = escapeHtml(rawStockName);
      
      let step1Html = '';
      const percentiles = d.step1_model?.feature_percentiles;
      if (isPlainObject(percentiles) && Object.keys(percentiles).length > 0) {
        const contributions = isPlainObject(d.step1_model.logit_contributions) ? d.step1_model.logit_contributions : {};
        let items = '';
        for (const [k, v] of Object.entries(percentiles)) {
          const pct = typeof v === 'number' && Number.isFinite(v) ? Math.min(Math.max(v, 0), 100) : null;
          const bar = pct === null ? '' : `<div class="acc-bar-bg"><div class="acc-bar-fill" style="width: ${pct.toFixed(0)}%"></div></div>`;
          const contribution = contributions[k];
          const contributionStr = typeof contribution === 'number' && Number.isFinite(contribution)
            ? `${contribution >= 0 ? '+' : ''}${contribution.toFixed(3)}` : 'N/A';
          items += `<div class="var-item"><span class="var-key">${escapeHtml(k)}</span>`
            + `<span class="var-value"><span title="Percentile among the ranked stocks">pct ${pct === null ? 'N/A' : pct.toFixed(0)}</span> ${bar} <span title="Contribution to the log-odds relative to a median stock">${contributionStr}</span></span></div>`;
        }
        step1Html = `<h4>Step 1: Feature percentiles and contributions</h4>`
          + `<p class="model-note">Percentile of the stock among the ranked stocks of ${escapeHtml(d.step1_model.scoring_date || 'the scoring date')}; `
          + `contribution = coefficient x (percentile / 100 - 0.5), the change of the log-odds relative to a median stock.</p>`
          + `<div class="var-grid">${items}</div>`;
      }
      
      let step2Html = '';
      if (d.step2_model && d.step2_model.feature_diagnostics) {
        let items = '';
        for (const [k, v] of Object.entries(d.step2_model.feature_diagnostics)) {
          if (typeof v === 'object' && v !== null) {
            items += `<div style="grid-column: 1 / -1; margin-top: 1rem; border-bottom: 1px solid var(--border-color); padding-bottom: 0.5rem; margin-bottom: 0.5rem;"><strong>${escapeHtml(k)}</strong></div>`;
            for (const [subK, subV] of Object.entries(v)) {
              let disp = typeof subV === 'number' ? Number(subV).toFixed(2) : (subV === null ? 'N/A' : escapeHtml(subV));
              items += `<div class="var-item" style="padding-left: 1rem;"><span class="var-key">${escapeHtml(subK)}</span><span class="var-value" style="font-weight: bold;">${disp}</span></div>`;
            }
          } else {
            let disp = escapeHtml(v);
            if (v === null) {
              disp = '<span style="color: var(--text-muted);">N/A</span>';
            } else if (k === 'Evaluable') {
              disp = v ? 'YES' : '<span style="color: var(--status-red);">NO</span>';
            } else if (typeof v === 'boolean') {
              disp = v ? '<span style="color: var(--status-green);">PASS</span>' : '<span style="color: var(--status-red);">FAIL</span>';
            } else if (typeof v === 'number') {
              disp = Number(v).toFixed(2);
            }
            items += `<div class="var-item"><span class="var-key">${escapeHtml(k)}</span><span class="var-value">${disp}</span></div>`;
          }
        }
        step2Html = `<div style="margin-top: 2rem; border-top: 2px dashed var(--border-color); padding-top: 1rem;">
          <h4 style="color: var(--status-orange); margin-bottom: 1rem; text-transform: uppercase;">Step 2: Fundamental Ruleset Diagnostics</h4>
          <div class="var-grid">${items}</div>
        </div>`;
      } else if (d.final_prediction === 'UP') {
        step2Html = `<div style="margin-top: 2rem; border-top: 2px dashed var(--border-color); padding-top: 1rem;">
          <h4 style="color: var(--status-orange); margin-bottom: 1rem; text-transform: uppercase;">Step 2: Fundamental Ruleset Diagnostics</h4>
          <div style="color: var(--status-red); margin-top: 0.5rem;">FAIL: Insufficient quarterly fundamental data to evaluate ruleset.</div>
        </div>`;
      }
      
      if (step1Html || step2Html) {
        html += `
          <details class="stock-variables">
            <summary>${stockName} ${d.company_name ? '- ' + escapeHtml(d.company_name) : ''}</summary>
            <div class="variables-content">
              ${step1Html}
              ${step2Html}
            </div>
          </details>
        `;
      }
    });
    
    if (!html) {
      html = '<div style="color: var(--text-muted);">No variables found for the given search.</div>';
    }
    
    container.innerHTML = html;
  }
});
