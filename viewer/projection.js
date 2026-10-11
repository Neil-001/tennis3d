import { courtEdges } from './court.js';

export function drawProjection(canvas, result, index) {
  const { width, height } = result.source;
  const ctx = canvas.getContext('2d');
  canvas.width = 640;
  canvas.height = Math.max(1, Math.round(640 * height / width));
  // Display a bounded canvas without changing the source-coordinate mapping.
  if (canvas.height > 640) { canvas.height = 640; canvas.width = Math.max(1, Math.round(640 * width / height)); }
  ctx.fillStyle = '#09090b'; ctx.fillRect(0,0,canvas.width,canvas.height);
  const frame = result.synthetic_diagnostics.frames[index];
  if (!frame) return;
  const sx = canvas.width / width, sy = canvas.height / height;
  ctx.strokeStyle = '#71717a'; ctx.lineWidth = 1;
  for (const [a,b] of courtEdges) {
    const p = frame.court_uv[a], q = frame.court_uv[b];
    if (!p || !q) continue;
    ctx.beginPath(); ctx.moveTo(p[0]*sx,p[1]*sy); ctx.lineTo(q[0]*sx,q[1]*sy); ctx.stroke();
  }
  ctx.font = '10px "Google Sans Code", monospace'; ctx.fillStyle = '#a1a1aa';
  frame.court_uv.forEach((p,i) => { if (p) ctx.fillText(String(i),p[0]*sx + 3,p[1]*sy - 3); });
  const clean = frame.ball_clean_uv, observed = frame.ball_observed_uv;
  if (clean) {
    ctx.strokeStyle = '#d7ff62'; ctx.lineWidth = 2; ctx.beginPath();
    ctx.arc(clean[0]*sx,clean[1]*sy,5,0,Math.PI*2); ctx.stroke();
  }
  if (observed) {
    const [x,y] = [observed[0]*sx,observed[1]*sy];
    ctx.strokeStyle = '#fbbf24'; ctx.beginPath();
    ctx.moveTo(x-4,y-4); ctx.lineTo(x+4,y+4); ctx.moveTo(x+4,y-4); ctx.lineTo(x-4,y+4); ctx.stroke();
  }
}
