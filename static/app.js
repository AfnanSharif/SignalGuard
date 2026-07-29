const form = document.querySelector('#score-form');

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const payload = Object.fromEntries(
    [...new FormData(form)].map(([key, value]) => [key, Number(value)]),
  );
  const response = await fetch('/api/v1/predict', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload),
  });
  const data = await response.json();
  const panel = document.querySelector('#result');
  panel.classList.remove('hidden');
  if (data.error) {
    document.querySelector('#message').textContent = data.error;
    return;
  }
  document.querySelector('#risk').textContent = Math.round(data.risk_score);
  document.querySelector('#verdict').textContent = data.is_anomaly ? 'ANOMALY DETECTED' : 'WITHIN BASELINE';
  document.querySelector('#message').textContent = data.is_anomaly
    ? 'This pattern deserves review.'
    : 'This pattern reconstructs normally.';
  document.querySelector('#detail').textContent =
    `Error ${data.reconstruction_error.toFixed(5)} · threshold ${data.threshold.toFixed(5)}. ` +
    'A flag is a triage signal, not proof of fraud.';

  // Feature names can originate in a user-supplied training CSV. Build DOM
  // nodes with textContent instead of interpolating them into HTML.
  const bars = document.querySelector('#bars');
  bars.replaceChildren();
  const max = Math.max(...Object.values(data.feature_errors), 0.000001);
  Object.entries(data.feature_errors)
    .sort((left, right) => right[1] - left[1])
    .forEach(([key, value]) => {
      const row = document.createElement('div');
      row.className = 'bar';
      const label = document.createElement('span');
      label.textContent = key.replaceAll('_', ' ');
      const track = document.createElement('div');
      track.className = 'track';
      const fill = document.createElement('div');
      fill.className = 'fill';
      fill.style.width = `${100 * value / max}%`;
      track.appendChild(fill);
      const number = document.createElement('code');
      number.textContent = value.toFixed(3);
      row.append(label, track, number);
      bars.appendChild(row);
    });
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  panel.scrollIntoView({behavior: reducedMotion ? 'auto' : 'smooth'});
});
