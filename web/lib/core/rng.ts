/** Mulberry32 + Box-Muller; bit-for-bit the same stream as atlime/rng.py. */
export class Rng {
  private state: number;
  private spare: number | null = null;

  constructor(seed = 0) {
    this.state = seed >>> 0;
  }

  nextU32(): number {
    this.state = (this.state + 0x6d2b79f5) >>> 0;
    const a = this.state;
    let t = Math.imul(a ^ (a >>> 15), a | 1);
    t = (t + Math.imul(t ^ (t >>> 7), t | 61)) ^ t;
    return (t ^ (t >>> 14)) >>> 0;
  }

  /** Uniform float in [0, 1). */
  uniform(): number {
    return this.nextU32() / 4294967296;
  }

  /** Standard normal (Box-Muller, second value cached). */
  normal(): number {
    if (this.spare !== null) {
      const z = this.spare;
      this.spare = null;
      return z;
    }
    const u1 = 1 - this.uniform();
    const u2 = this.uniform();
    const r = Math.sqrt(-2 * Math.log(u1));
    const theta = 2 * Math.PI * u2;
    this.spare = r * Math.sin(theta);
    return r * Math.cos(theta);
  }

  randint(n: number): number {
    return Math.floor(this.uniform() * n);
  }

  spawn(salt: number): Rng {
    return new Rng(deriveSeed(this.nextU32(), salt));
  }
}

export function deriveSeed(seed: number, salt: number): number {
  return new Rng((Math.imul(seed, 0x9e3779b1) + Math.imul(salt, 0x85ebca77)) >>> 0).nextU32();
}
