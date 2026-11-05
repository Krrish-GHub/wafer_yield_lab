const $ = (id) => document.getElementById(id);
let model;
let samples = [];

function sigmoid(z) {
  if (z >= 0) return 1 / (1 + Math.exp(-z));
  const e = Math.exp(z);
  return e / (1 + e);
}

function evaluate(features) {
  const effects = model.selected_indices.map((index, j) => {
    const raw = features[index] == null ? model.medians[j] : features[index];
    const standardized = (raw - model.means[j]) / model.scales[j];
    return { name: model.feature_names[j], value: standardized * model.coefficients[j] };
  });
  const logit = model.intercept + effects.reduce((sum, item) => sum + item.value, 0);
  return { score: sigmoid(logit), effects };
}

function showSample(index) {
  const sample = samples[index];
  if (!sample) return;
  const result = evaluate(sample.features);
  // Compare JS arithmetic against the saved Python pipeline score to catch export drift.
  const exportDelta = Math.abs(result.score - sample.score);
  $('record-id').textContent = sample.id;
  $('actual').textContent = sample.actual;
  $('actual').style.color = sample.actual === 'fail' ? 'var(--red)' : 'var(--green)';
  $('timestamp').textContent = sample.timestamp;
  $('score').textContent = `${(result.score * 100).toFixed(1)}%`;
  $('score-ring').style.setProperty('--pct', `${result.score * 360}deg`);
  $('threshold-label').textContent = `TRAINING CV THRESHOLD · ${model.threshold.toFixed(3)}`;
  const predictedFail = result.score >= model.threshold;
  $('prediction').textContent = predictedFail ? 'Flagged as fail' : 'Below fail threshold';
  $('prediction').style.color = predictedFail ? 'var(--red)' : 'var(--mint)';
  $('score-explainer').textContent = `Model score for the fail class. Python/JavaScript parity check: ${exportDelta < 1e-9 ? 'pass' : 'check needed'}; score is not calibrated.`;

  const top = [...result.effects].sort((a, b) => Math.abs(b.value) - Math.abs(a.value)).slice(0, 8);
  const max = Math.max(...top.map((item) => Math.abs(item.value)), 1e-9);
  $('contributions').innerHTML = top.map((item) => {
    const positive = item.value >= 0;
    const width = Math.max(2, Math.abs(item.value) / max * 100);
    const left = positive ? 0 : 100 - width;
    return `<div class="contrib-row"><span class="sensor-name">${item.name}</span><div class="bar-track"><span class="bar-fill ${positive ? 'pos' : 'neg'}" style="left:${left}%;width:${width}%"></span></div><span class="contrib-value">${item.value > 0 ? '+' : ''}${item.value.toFixed(2)}</span></div>`;
  }).join('');
}

async function main() {
  try {
    const [modelResponse, sampleResponse] = await Promise.all([fetch('./model.json'), fetch('./samples.json')]);
    if (!modelResponse.ok || !sampleResponse.ok) throw new Error('Model files did not load. Serve this folder over HTTP.');
    [model, samples] = await Promise.all([modelResponse.json(), sampleResponse.json()]);
    $('sample-count').textContent = `${samples.length} EXAMPLES`;
    $('sample-select').innerHTML = samples.map((sample, index) =>
      `<option value="${index}">${sample.id} · actual ${sample.actual}</option>`).join('');
    $('sample-select').addEventListener('change', (event) => showSample(Number(event.target.value)));
    showSample(0);
  } catch (error) {
    $('prediction').textContent = 'Could not load model';
    $('score-explainer').textContent = `${error.message} Try opening the GitHub Pages demo or run a local HTTP server.`;
    $('contributions').innerHTML = '';
  }
}
main();
