import * as THREE from 'three';
import geometry from './generated/court.json';
const courtPoints = geometry.points;
export const ballRadius = geometry.radius;
export const courtEdges = [[0,1],[1,3],[3,2],[2,0],[4,5],[6,7],[8,9],[10,11],[12,13]];
export const eventColors = { hit: '#ef91ff', bounce: '#ffcf70', net: '#ff7385', out: '#b2a5ff' };

export function lines(points, color, dashed = false) {
  const material = dashed ? new THREE.LineDashedMaterial({ color, dashSize: 0.18, gapSize: 0.12 }) : new THREE.LineBasicMaterial({ color });
  const result = new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(points.map(p => new THREE.Vector3(...p))), material);
  if (dashed) result.computeLineDistances();
  return result;
}

export function addCourt(scene) {
  const ground = new THREE.Mesh(new THREE.PlaneGeometry(18, 34), new THREE.MeshBasicMaterial({ color: '#18181b', side: THREE.DoubleSide }));
  ground.position.z = -0.025;
  scene.add(ground);
  const court = new THREE.Mesh(new THREE.PlaneGeometry(courtPoints[1][0] - courtPoints[0][0], courtPoints[0][1] - courtPoints[2][1]), new THREE.MeshBasicMaterial({ color: '#0b5158', side: THREE.DoubleSide }));
  court.position.z = -0.015;
  scene.add(court, lines(courtEdges.flatMap(e => e.map(i => courtPoints[i])), '#fafafa'));
  // Doubles-court posts. Tape slopes from 1.07 m at posts to 0.914 m at centre.
  const half = courtPoints[1][0] + 0.914, net = [];
  const height = x => 0.914 + (1.07 - 0.914) * Math.abs(x) / half;
  for (let x = -half; x <= half; x += 0.3) net.push([x,0,0], [x,0,height(x)]);
  for (let z = 0.15; z < 0.914; z += 0.15) net.push([-half,0,z],[half,0,z]);
  scene.add(lines(net, '#71717a'), lines([[-half,0,1.07],[0,0,0.914],[0,0,0.914],[half,0,1.07]], '#ffffff'));
  scene.add(lines([[-half,0,0],[-half,0,1.07],[half,0,0],[half,0,1.07]], '#e4e4e7'));
  scene.add(new THREE.AxesHelper(2));
}
