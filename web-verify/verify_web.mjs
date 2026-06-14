// Confirm the WASM backend (what the browser uses) loads + runs all 3 models.
import * as ort from 'onnxruntime-web';
import fs from 'fs';
import { WakeWord } from './pipeline.js';

ort.env.wasm.numThreads = 1;

const WEB = 'C:/Users/Administrator/hotword-service/web/models';
const REF = JSON.parse(fs.readFileSync('./ref_scores.json'));

function readWavInt16(path) {
  const buf = fs.readFileSync(path);
  let off = 12;
  while (off + 8 <= buf.length) {
    const id = buf.toString('ascii', off, off + 4);
    const size = buf.readUInt32LE(off + 4);
    if (id === 'data') {
      const n = size / 2, out = new Int16Array(n);
      for (let i = 0; i < n; i++) out[i] = buf.readInt16LE(off + 8 + i * 2);
      return out;
    }
    off += 8 + size + (size % 2);
  }
  throw new Error('no data');
}
const load = (f) => ort.InferenceSession.create(new Uint8Array(fs.readFileSync(`${WEB}/${f}`)), { executionProviders: ['wasm'] });

const main = async () => {
  const mel = await load('melspectrogram.onnx');
  const emb = await load('embedding_model.onnx');
  const ww = await load('hey_kiki.onnx');
  console.log('WASM loaded all 3 models OK');

  const pcm = readWavInt16('./clip16k.wav');
  const wk = new WakeWord(ort, mel, emb, ww);
  const scores = [];
  for (let i = 0; i + 1280 <= pcm.length; i += 1280) scores.push(await wk.pushChunk(pcm.subarray(i, i + 1280)));

  let maxDiff = 0, n = 0;
  for (let f = 15; f < REF.scores.length; f++) {
    if (scores[f] == null) continue;
    maxDiff = Math.max(maxDiff, Math.abs(REF.scores[f] - scores[f])); n++;
  }
  console.log(`compared ${n} frames vs python, max diff = ${maxDiff.toExponential(3)}`);
  console.log(maxDiff < 1e-3 ? 'PASS: WASM backend matches Python' : 'FAIL');
};
main().catch(e => { console.error(e); process.exit(1); });
