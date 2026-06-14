import * as ort from 'onnxruntime-node';
import fs from 'fs';
import { WakeWord } from './pipeline.js';

const SITE = 'C:/Users/Administrator/AppData/Local/Programs/Python/Python312/Lib/site-packages/openwakeword/resources/models';
const MEL = `${SITE}/melspectrogram.onnx`;
const EMB = `${SITE}/embedding_model.onnx`;
const WW = 'C:/Users/Administrator/hotword-service/models/hey_kiki.onnx';
const WAV = './clip16k.wav';
const REF = JSON.parse(fs.readFileSync('./ref_scores.json'));

function readWavInt16(path) {
  const buf = fs.readFileSync(path);
  // scan RIFF chunks for 'data'
  let off = 12;
  while (off + 8 <= buf.length) {
    const id = buf.toString('ascii', off, off + 4);
    const size = buf.readUInt32LE(off + 4);
    if (id === 'data') {
      const n = size / 2;
      const out = new Int16Array(n);
      for (let i = 0; i < n; i++) out[i] = buf.readInt16LE(off + 8 + i * 2);
      return out;
    }
    off += 8 + size + (size % 2);
  }
  throw new Error('no data chunk');
}

const main = async () => {
  const mel = await ort.InferenceSession.create(MEL);
  const emb = await ort.InferenceSession.create(EMB);
  const ww = await ort.InferenceSession.create(WW);
  console.log('ww input/output:', ww.inputNames, ww.outputNames, '| mel out:', mel.outputNames, '| emb out:', emb.outputNames);

  const pcm = readWavInt16(WAV);
  const wk = new WakeWord(ort, mel, emb, ww);

  const scores = [];
  for (let i = 0; i + 1280 <= pcm.length; i += 1280) {
    const s = await wk.pushChunk(pcm.subarray(i, i + 1280));
    scores.push(s);
  }

  // compare frames where JS has all-real features (>=15) to Python ref
  console.log('\nframe |  python  |   js     | diff');
  let maxDiff = 0, compared = 0;
  for (let f = 15; f < REF.scores.length; f++) {
    const p = REF.scores[f];
    const j = scores[f];
    if (j == null) continue;
    const d = Math.abs(p - j);
    maxDiff = Math.max(maxDiff, d); compared++;
    console.log(`${String(f).padStart(5)} | ${p.toFixed(6)} | ${j.toFixed(6)} | ${d.toExponential(2)}`);
  }
  console.log(`\ncompared ${compared} frames, MAX ABS DIFF = ${maxDiff.toExponential(3)}`);
  console.log(maxDiff < 1e-3 ? 'PASS: JS pipeline matches Python' : 'FAIL: pipeline mismatch');
};
main().catch(e => { console.error(e); process.exit(1); });
