import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { addCourt, ballRadius, eventColors, lines } from './court.js';
import { trailSegments } from './playback.js';

export function createScene(container) {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color('#09090b');
  const camera = new THREE.PerspectiveCamera(45, 1, 0.05, 2000);
  camera.up.set(0, 0, 1);
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.domElement.setAttribute('aria-label', 'Metric 3D tennis court. Drag to orbit and scroll to zoom.');
  renderer.domElement.tabIndex = 0;
  container.append(renderer.domElement);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.minDistance = 2;
  controls.maxDistance = 400;
  // OrbitControls skips the wheel while a button is held, and prevents the default
  // of every wheel event it handles. Dolly here for the ones it skipped, with its
  // own scale and limits. This listener must stay registered after the controls.
  renderer.domElement.addEventListener('wheel', event => {
    if (event.defaultPrevented || !controls.enabled || !controls.enableZoom || !event.deltaY) return;
    event.preventDefault();
    const pixels = event.deltaY * [1, 16, 100][event.deltaMode];
    const offset = camera.position.clone().sub(controls.target);
    const distance = offset.length() * 0.95 ** (controls.zoomSpeed * pixels * -0.01);
    offset.setLength(THREE.MathUtils.clamp(distance, controls.minDistance, controls.maxDistance));
    camera.position.copy(controls.target).add(offset);
    render();
  }, { passive: false });
  const court = new THREE.Group();
  addCourt(court);
  scene.add(court);
  const ball = new THREE.Mesh(new THREE.SphereGeometry(ballRadius * 5, 20, 12), new THREE.MeshBasicMaterial({ color: '#d7ff62' }));
  ball.visible = false;
  scene.add(ball);
  const trail = { estimated: lines([], '#d7ff62'), inferred: lines([], '#fbbf24', true) };
  scene.add(...Object.values(trail));
  const events = new THREE.Group();
  scene.add(events);
  function preset(name) {
    controls.target.set(0, 0, 0.5);
    const scale = Math.max(1, 1 / camera.aspect);
    const p = { perspective: [24,-31,25], top: [0,-0.001,43], side: [40,0,0.5] }[name];
    camera.position.set(p[0] * scale, p[1] * scale, p[2] * scale);
    render();
  }
  const resize = new ResizeObserver(() => {
    const { width, height } = container.getBoundingClientRect();
    if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) return;
    renderer.setSize(width, height);
    const aspect = width / height;
    // Preserve orbit and zoom while maintaining the preset's framing on narrow views.
    const scale = Math.max(1, 1 / aspect) / Math.max(1, 1 / camera.aspect);
    camera.position.sub(controls.target).multiplyScalar(scale).add(controls.target);
    camera.aspect = aspect;
    camera.updateProjectionMatrix();
    render();
  });
  resize.observe(container);
  function draw() {
    renderer.render(scene, camera);
    // The browser tests read the camera position from the canvas.
    renderer.domElement.dataset.camera = camera.position.toArray().map(value => value.toFixed(3)).join(' ');
  }
  function render() { controls.update(); draw(); }
  controls.addEventListener('change', draw);
  function setEvents(data) {
    for (const child of [...events.children]) {
      events.remove(child); child.geometry.dispose(); child.material.dispose();
    }
    for (const event of data) {
      const marker = new THREE.Mesh(new THREE.OctahedronGeometry(0.22), new THREE.MeshBasicMaterial({ color: eventColors[event.kind], wireframe: true }));
      marker.position.fromArray(event.xyz_m);
      marker.userData.t_s = event.t_s;
      events.add(marker);
    }
  }
  function update(frames, time, sample, seconds) {
    ball.visible = !!sample.xyz;
    if (sample.xyz) ball.position.fromArray(sample.xyz);
    ball.material.color.set(sample.state === 'inferred' ? '#fbbf24' : '#d7ff62');
    const segments = trailSegments(frames, time, seconds);
    for (const key of Object.keys(trail)) {
      trail[key].geometry.dispose();
      trail[key].geometry = new THREE.BufferGeometry();
      trail[key].geometry.setAttribute('position', new THREE.Float32BufferAttribute(segments[key], 3));
      if (key === 'inferred') trail[key].computeLineDistances();
    }
    for (const marker of events.children) marker.visible = marker.userData.t_s <= time;
    render();
  }
  const initialAspect = container.clientWidth / container.clientHeight;
  camera.aspect = Number.isFinite(initialAspect) && initialAspect > 0 ? initialAspect : 1;
  preset('perspective');
  return { preset, setEvents, update, setCourtVisible(visible) { court.visible = visible; render(); } };
}
