// Faithful JS port of openWakeWord's streaming feature pipeline.
// Works with both onnxruntime-web (browser) and onnxruntime-node — you pass in the
// `ort` module and the three created InferenceSessions.
//
// Pipeline per 1280-sample (80 ms) chunk of 16 kHz audio:
//   raw int16 -> melspectrogram model (last 1760 samples) -> +8 mel frames (x/10+2)
//   last 76 mel frames -> embedding model -> 1 x 96-dim embedding
//   last 16 embeddings -> wakeword model -> score in [0,1]

export class WakeWord {
  constructor(ort, melSess, embSess, wwSess) {
    this.ort = ort;
    this.mel = melSess;
    this.emb = embSess;
    this.ww = wwSess;
    this.wwInput = wwSess.inputNames[0];
    this.raw = [];                 // int16-valued samples
    this.melBuf = [];              // array of Float32Array(32)
    for (let i = 0; i < 76; i++) this.melBuf.push(new Float32Array(32).fill(1)); // ones((76,32))
    this.feat = [];               // array of Float32Array(96)
    this.melMax = 10 * 97;        // 970
    this.featMax = 120;
  }

  async _melspec(samples) {
    const x = Float32Array.from(samples);
    const t = new this.ort.Tensor('float32', x, [1, x.length]);
    const out = await this.mel.run({ input: t });
    const o = out[this.mel.outputNames[0]];
    const frames = o.dims[o.dims.length - 2];
    const bins = o.dims[o.dims.length - 1]; // 32
    const d = o.data;
    const rows = [];
    for (let f = 0; f < frames; f++) {
      const row = new Float32Array(bins);
      for (let b = 0; b < bins; b++) row[b] = d[f * bins + b] / 10 + 2; // transform
      rows.push(row);
    }
    return rows;
  }

  async _embed(window76) {
    // window76: array of 76 Float32Array(32) -> tensor [1,76,32,1]
    const x = new Float32Array(76 * 32);
    for (let f = 0; f < 76; f++) x.set(window76[f], f * 32);
    const t = new this.ort.Tensor('float32', x, [1, 76, 32, 1]);
    const out = await this.emb.run({ input_1: t });
    return out[this.emb.outputNames[0]].data; // length 96
  }

  // Feed exactly one 1280-sample Int16Array chunk. Returns score or null (warmup).
  async pushChunk(int16chunk) {
    for (let i = 0; i < int16chunk.length; i++) this.raw.push(int16chunk[i]);
    if (this.raw.length > 16000) this.raw = this.raw.slice(-16000);

    // melspectrogram of last 1280+480 samples
    const take = Math.min(this.raw.length, 1280 + 160 * 3);
    const newRows = await this._melspec(this.raw.slice(this.raw.length - take));
    for (const r of newRows) this.melBuf.push(r);
    if (this.melBuf.length > this.melMax) this.melBuf = this.melBuf.slice(-this.melMax);

    // one new embedding from the last 76 mel frames
    const window = this.melBuf.slice(this.melBuf.length - 76);
    if (window.length === 76) {
      const e = await this._embed(window);
      this.feat.push(Float32Array.from(e));
      if (this.feat.length > this.featMax) this.feat = this.feat.slice(-this.featMax);
    }

    // score once we have 16 real embeddings
    if (this.feat.length < 16) return null;
    const last16 = this.feat.slice(this.feat.length - 16);
    const x = new Float32Array(16 * 96);
    for (let i = 0; i < 16; i++) x.set(last16[i], i * 96);
    const t = new this.ort.Tensor('float32', x, [1, 16, 96]);
    const out = await this.ww.run({ [this.wwInput]: t });
    return out[this.ww.outputNames[0]].data[0];
  }
}
