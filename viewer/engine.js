/* 基体探査エンジン(ブラウザ版)
 *
 * substrate.py / world.py / hypothesis.py / investigation.py の中核を写したもの。
 * **Python 側が参照実装であり、こちらは同じ規則で動く実行系である。**
 *
 * 二つある以上、必ずずれる。ずれを黙って進行させないため、Python 側のテストが
 * このファイルの定数(性質名・実験名・利得・雑音)を読んで照合する。数値を書き換える
 * ときは、両方を同時に直さねばテストが落ちる。
 *
 * 探査者はここでは規則で動く(言語モデルではない)。判断の根拠を一行ずつ画面に出す
 * ためであり、**推論を隠さないこと** を優先した。言語モデル版は llm_scientist.py にある。
 */
(function (global) {
  "use strict";

  /* ------------------------------------------------------------------
   * 基体の性質。detectability は「内側から見えるか」の分類である。
   * ------------------------------------------------------------------ */
  const DETECTABLE = "detectable";
  const UNDETECTABLE = "undetectable";
  const CONFOUNDED = "confounded";

  const PROPERTIES = [
    { key: "discrete", label: "空間の離散性", tier: DETECTABLE,
      why: "格子では方向によって伝播が変わる。連続な世界には現れない異方性が出る" },
    { key: "quantized", label: "数値の粒度", tier: DETECTABLE,
      why: "有限精度では加算の結合則が破れ、差が蓄積する" },
    { key: "budgeted", label: "計算予算", tier: DETECTABLE,
      why: "質量ではなく複雑さに相関する時間の遅れは、物理では説明がつかない" },
    { key: "bounded", label: "世界の有限性", tier: DETECTABLE,
      why: "十分遠くまで進めば、境界に触れるか出発点へ戻る" },
    { key: "leaky", label: "保存則の破れ", tier: DETECTABLE,
      why: "総和を長く追えば、注入や漏れが累積として現れる" },
    { key: "patched", label: "規則の改変", tier: CONFOUNDED,
      why: "記録を跨げば見える。ただし記憶を書き換えられる基体では記録が信用できない" },
    { key: "lazy", label: "遅延評価", tier: CONFOUNDED,
      why: "観測順と結果の相関に現れる。ただし遡って辻褄を合わせる実装なら消える" },
    { key: "deterministic", label: "決定論", tier: UNDETECTABLE,
      why: "歴史を巻き戻せる者にしか分からない。内側の者は一度きりの歴史しか持たない" },
    { key: "nested", label: "入れ子の深さ", tier: UNDETECTABLE,
      why: "計算論的に等価である限り、内側の計算では区別できない" },
    { key: "rewrites_memory", label: "記憶の書換", tier: UNDETECTABLE,
      why: "書き換えを検出する記録もまた書き換えられる。**全ての実験結果が信用できない**" }
  ];

  const KEYS = PROPERTIES.map(p => p.key);
  const BY_KEY = Object.fromEntries(PROPERTIES.map(p => [p.key, p]));
  const DETECTABLE_KEYS = PROPERTIES.filter(p => p.tier === DETECTABLE).map(p => p.key);

  /* 実験。**この語彙の外は打てない。** */
  const EXPERIMENTS = ["anisotropy", "associativity", "complexity_time",
                       "far_travel", "conservation", "constant_drift",
                       "observation_order"];
  const EXP_LABEL = {
    anisotropy: "方向による伝播差",
    associativity: "加算の結合則",
    complexity_time: "複雑さと時間",
    far_travel: "遠方への到達",
    conservation: "保存量の追跡",
    constant_drift: "定数の世代差",
    observation_order: "観測順の影響"
  };

  const BASE_NOISE = 0.012;      /* 測定に必ず乗る。0 にすれば科学ではなく開示になる */
  const ANISOTROPY_GAIN = 0.8;   /* 格子間隔から異方性への利得(仮定) */

  /* ------------------------------------------------------------------
   * 再現可能な乱数。同じ種から同じ世界が生まれることが要件である。
   * ------------------------------------------------------------------ */
  function rng(seed) {
    let s = (seed >>> 0) || 1;
    const next = () => {
      s ^= s << 13; s >>>= 0;
      s ^= s >> 17;
      s ^= s << 5;  s >>>= 0;
      return s / 4294967296;
    };
    next.pick = (arr) => arr[Math.floor(next() * arr.length)];
    next.gauss = (sd) => {
      const u = Math.max(next(), 1e-9), v = next();
      return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v) * sd;
    };
    return next;
  }

  /* ------------------------------------------------------------------
   * 基体を作り、封緘する。**中身は開示するまで読ませない。**
   * ------------------------------------------------------------------ */
  function randomSubstrate(r) {
    return {
      discrete: r.pick([0, 0, 0.02, 0.05, 0.1]),
      quantized: r.pick([0, 0, 1e-4, 1e-3]),
      budgeted: r.pick([0, 0, 0.05, 0.2]),
      bounded: r.pick([0, 0, 40, 120]),
      leaky: r.pick([0, 0, 0.001, 0.01]),
      patched: r.pick([0, 0, 0, 0.15]),
      lazy: r.pick([0, 0, 0, 0.4]),
      deterministic: r.pick([true, false]),
      nested: r.pick([1, 1, 1, 2]),
      rewrites_memory: r.pick([0, 0, 0, 0, 0, 0.08])
    };
  }

  function activeKeys(s) {
    return KEYS.filter(k => {
      if (k === "deterministic") return s[k] === true;
      if (k === "nested") return s[k] > 1;
      return s[k] > 0;
    });
  }

  /* 内容から短い封緘符を作る。中身を変えれば符も変わる。 */
  function sealHash(s) {
    const text = KEYS.map(k => k + "=" + s[k]).join("|");
    let h = 2166136261 >>> 0;
    for (let i = 0; i < text.length; i++) {
      h ^= text.charCodeAt(i); h = Math.imul(h, 16777619) >>> 0;
    }
    return h.toString(16).padStart(8, "0");
  }

  /* ------------------------------------------------------------------
   * 世界 — 基体を測定値へ翻訳する。雑音が乗り、書換で汚染される。
   * ------------------------------------------------------------------ */
  function makeWorld(substrate, seed) {
    const r = rng(seed);
    let corrupted = 0;

    const kernels = {
      anisotropy: () => 1 + ANISOTROPY_GAIN * substrate.discrete,
      associativity: (n) => substrate.quantized <= 0 ? 0
        : substrate.quantized * Math.sqrt(Math.max(1, n)),
      complexity_time: () => substrate.budgeted,
      far_travel: (n) => {
        if (substrate.bounded <= 0) return 1;
        const reach = n * 3;
        return reach > substrate.bounded ? Math.min(1, substrate.bounded / reach) : 1;
      },
      conservation: (n) => substrate.leaky * n,
      constant_drift: () => 0,      /* 第一世代では改変は見えない */
      observation_order: () => substrate.lazy * 0.5
    };

    return {
      run(experiment, trials) {
        const truth = kernels[experiment](trials);
        let value = truth + r.gauss(BASE_NOISE / Math.sqrt(trials));
        let wasCorrupted = false;
        /* 記憶の書換: 測定値そのものが差し替わる。エージェントには見えない。 */
        if (substrate.rewrites_memory > 0 && r() < substrate.rewrites_memory) {
          wasCorrupted = true; corrupted++;
          value = truth + r.gauss(0.35);
        }
        return { experiment, value, trials, corrupted: wasCorrupted };
      },
      get corruptedCount() { return corrupted; }
    };
  }

  /* ------------------------------------------------------------------
   * 探査者 — 規則で動く。**判断の根拠を必ず言葉にする。**
   *
   * lean は「何を予期して世界を見るか」である。0 に近いほど証拠だけで動き、
   * 1 に近いほど期待が測定を押しのける。**この偏りこそが観測対象である。**
   * ------------------------------------------------------------------ */
  const SCIENTISTS = [
    { id: "A0", name: "異方の観測者", color: 0,
      motto: "方向で違いが出るなら、それは格子の痕跡だ",
      lean: { discrete: 0.55 }, sensitivity: 0.6 },
    { id: "A1", name: "保存の番人", color: 1,
      motto: "総量が合わないことに、私は敏感である",
      lean: { leaky: 0.6 }, sensitivity: 0.5 },
    { id: "A2", name: "果ての探索者", color: 2,
      motto: "どこまで行けるかを、まず知りたい",
      lean: { bounded: 0.35, nested: 0.7 }, sensitivity: 0.7 },
    { id: "A3", name: "時間の計測者", color: 3,
      motto: "速さの違いには、必ず理由がある",
      lean: { budgeted: 0.4 }, sensitivity: 0.65 }
  ];

  /* 実験ごとの判定規則。閾値と、それが示す性質。 */
  const RULES = {
    anisotropy:        { key: "discrete",  threshold: 1.008, dir: "above" },
    associativity:     { key: "quantized", threshold: 0.0008, dir: "above" },
    complexity_time:   { key: "budgeted",  threshold: 0.02,  dir: "above" },
    far_travel:        { key: "bounded",   threshold: 0.98,  dir: "below" },
    conservation:      { key: "leaky",     threshold: 0.30,  dir: "above" },
    constant_drift:    { key: "patched",   threshold: 0.05,  dir: "above" },
    observation_order: { key: "lazy",      threshold: 0.08,  dir: "above" }
  };

  /* 実験を非対称に配る。**誰も同じものを見ていない。** */
  function dealExperiments(agentIds, seed) {
    const r = rng(seed);
    const pool = EXPERIMENTS.slice();
    for (let i = pool.length - 1; i > 0; i--) {
      const j = Math.floor(r() * (i + 1));
      [pool[i], pool[j]] = [pool[j], pool[i]];
    }
    const out = {};
    agentIds.forEach(id => { out[id] = []; });
    pool.forEach((e, i) => { out[agentIds[i % agentIds.length]].push(e); });
    return out;
  }

  /* 一つの測定から、探査者が何を読み取るか。**根拠を文にして返す。** */
  function interpret(scientist, m) {
    const rule = RULES[m.experiment];
    const over = rule.dir === "above" ? m.value > rule.threshold
                                      : m.value < rule.threshold;
    const lean = scientist.lean[rule.key] || 0;
    /* 期待が測定を押しのける度合い。lean が高いほど、境界付近で「ある」に倒れる。 */
    const margin = Math.abs(m.value - rule.threshold);
    const swayed = !over && lean > 0.5 && margin < 0.05 * scientist.sensitivity * 10;
    const believes = over || swayed;

    let reason;
    if (over) {
      reason = `${m.value.toFixed(4)} は閾値 ${rule.threshold} を`
             + (rule.dir === "above" ? "超えた" : "下回った")
             + ` → ${BY_KEY[rule.key].label}は「ある」`;
    } else if (swayed) {
      reason = `${m.value.toFixed(4)} は閾値に届かないが、境界に近い`
             + ` → 予期に引かれて「ある」と判断した`;
    } else {
      reason = `${m.value.toFixed(4)} は閾値 ${rule.threshold} に届かない`
             + ` → ${BY_KEY[rule.key].label}は「ない」`;
    }
    return { key: rule.key, believes, reason, swayed, rule };
  }

  /* 測っていない性質について、期待だけで断言してしまうか。 */
  function leapsOfFaith(scientist, examinedKeys) {
    return Object.entries(scientist.lean)
      .filter(([k, v]) => v >= 0.65 && examinedKeys.indexOf(k) === -1)
      .map(([k]) => k);
  }

  /* ------------------------------------------------------------------
   * 一つの世界の探査を、段階の並びとして組み立てる。
   * **画面はこの並びを一つずつ再生する。** 全部を一度に計算して見せない。
   * ------------------------------------------------------------------ */
  function planInvestigation(worldIndex, seed, trials) {
    const r = rng(seed);
    const substrate = randomSubstrate(r);
    const seal = sealHash(substrate);
    const world = makeWorld(substrate, seed + 977);
    const ids = SCIENTISTS.map(s => s.id);
    const deal = dealExperiments(ids, seed + 31);

    const steps = [];
    steps.push({ type: "seal", worldIndex, seal,
                 text: `世界 W-${String(worldIndex + 1).padStart(2, "0")} を封緘した。`
                     + `材質は誰にも見えない` });
    steps.push({ type: "deal", deal,
                 text: "実験を配った。**誰も同じものを見ていない**" });

    const beliefs = {};   /* agentId -> { key: bool } */
    const examined = {};
    ids.forEach(id => { beliefs[id] = {}; examined[id] = []; });

    /* 実験を打つ。一件ずつ、順に。 */
    ids.forEach(id => {
      const sci = SCIENTISTS.find(s => s.id === id);
      deal[id].forEach(exp => {
        const m = world.run(exp, trials);
        const read = interpret(sci, m);
        beliefs[id][read.key] = read.believes;
        examined[id].push(read.key);
        steps.push({
          type: "probe", agentId: id, experiment: exp,
          value: m.value, key: read.key, believes: read.believes,
          swayed: read.swayed, text: read.reason
        });
      });
    });

    /* 測っていないことまで断言する者がいる。 */
    ids.forEach(id => {
      const sci = SCIENTISTS.find(s => s.id === id);
      leapsOfFaith(sci, examined[id]).forEach(k => {
        beliefs[id][k] = true;
        steps.push({
          type: "leap", agentId: id, key: k,
          text: `${BY_KEY[k].label}は測っていない。それでも「ある」と述べた`
              + (BY_KEY[k].tier === UNDETECTABLE
                 ? " —— **この性質は原理的に検出できない**" : "")
        });
      });
    });

    steps.push({ type: "freeze",
                 text: "全員の予測を凍結した。**ここから先は書き換えられない**" });

    /* 合議。誰も触れなかった性質は null のまま。 */
    const consensus = {};
    KEYS.forEach(k => {
      const votes = ids.map(id => beliefs[id][k]).filter(v => v !== undefined);
      consensus[k] = votes.length === 0 ? null
        : (votes.filter(Boolean).length / votes.length) > 0.5;
    });
    steps.push({ type: "consensus", consensus,
                 text: "合議に至った。調べなかった性質は空欄のまま残す" });

    /* 開封。 */
    const truth = activeKeys(substrate);
    const opined = KEYS.filter(k => consensus[k] !== null);
    const scored = DETECTABLE_KEYS.filter(k => consensus[k] !== null);
    const hits = scored.filter(k => consensus[k] === (truth.indexOf(k) >= 0));
    const fairScore = scored.length ? hits.length / scored.length : null;
    const falsePos = opined.filter(k =>
      BY_KEY[k].tier !== UNDETECTABLE && consensus[k] && truth.indexOf(k) < 0);
    const falseNeg = opined.filter(k =>
      BY_KEY[k].tier !== UNDETECTABLE && !consensus[k] && truth.indexOf(k) >= 0);

    steps.push({
      type: "reveal", truth, consensus, fairScore, falsePos, falseNeg,
      untouched: KEYS.filter(k => consensus[k] === null),
      voided: substrate.rewrites_memory > 0,
      corrupted: world.corruptedCount,
      text: truth.length === 0
        ? "**封を切った。この世界は何の性質も持たない、素の世界だった**"
        : "封を切った"
    });

    return {
      worldIndex, seal, steps, truth, consensus, fairScore, falsePos, falseNeg,
      voided: substrate.rewrites_memory > 0
    };
  }

  global.SubstrateEngine = {
    PROPERTIES, KEYS, BY_KEY, DETECTABLE_KEYS, EXPERIMENTS, EXP_LABEL,
    SCIENTISTS, RULES, BASE_NOISE, ANISOTROPY_GAIN,
    DETECTABLE, UNDETECTABLE, CONFOUNDED,
    rng, randomSubstrate, activeKeys, sealHash, makeWorld,
    dealExperiments, planInvestigation
  };
})(typeof window !== "undefined" ? window : globalThis);
