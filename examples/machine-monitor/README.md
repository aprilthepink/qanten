# machine-monitor: predictive maintenance on Q.ANT NPUs

An example project that uses `qanten` end to end the way an industrial
monitoring product would. Accelerometers on pumps, fans, compressors and
conveyors send a 0.4-second vibration window every hour. The NPU runs the
signal processing and a small neural network that names the developing
fault, so maintenance can be planned before the machine breaks.

```text
accelerometer window (4096 samples @ 10.24 kHz) + shaft speed
        │
        ▼  NPU: FIR filter bank (low band 0–450 Hz, resonance band 2.4–4.4 kHz)
        ▼  host: decimate ×8, rectify the resonance band (envelope demodulation)
        ▼  NPU: DFT of both bands, whole batch in one call
        ▼  host: resample spectra from Hz to shaft orders, add RMS / crest / kurtosis /
        ▼        band energy / envelope floor  → 92 features
        ├──▶ NPU: MLP 92 → 64 → 32 → 6 ─────▶ condition probabilities: healthy, imbalance,
        │                                     misalignment, looseness, outer-race, inner-race
        └──▶ NPU: novelty matrix (qanten::compute::NpuMatrix) ─▶ novelty score
                                              ("unlike anything I was trained on")
        │
        ▼  alarm logic: smoothing + 3 confirming hours → maintenance alarm
```

It uses these parts of the library:

| Part | Used for |
|---|---|
| `qanten::signal` (`FirBank`, `Dft`, `hann`, `design`) | the vibration-analysis front end |
| `qanten::train` (`Dense`, `Adam`, `cross_entropy`) | training the classifier with the NPU in the loop |
| `qanten::nn` (`Linear`, `Sequential`, `Module`, `StateDict`) | the deployed model, saved as `.safetensors` |
| `qanten::compute` (`gemm`, `Placement::Auto`, `NpuMatrix`, `check_product`) | novelty detector: training covariance, resident scoring matrix, daily NPU self-check |
| `qanten::pool` (`NpuPool`, `replicate`) | one pipeline copy per NPU, each hour's batch split across them |
| `qanten::profile` | per-NPU calls, MACs and bytes |
| `qanten-serve` (`AppState::add_module`, `router`) | HTTP API with dynamic batching, plus a custom route |

No hardware is needed: without the Q.ANT driver it uses the simulator
(`--sim-npus N` sets how many simulated NPUs).

## 1. Train

```sh
cargo run --release -p machine-monitor -- train
```

This simulates 2,400 labelled windows from random machines (15–45 Hz shaft
speed, different noise floors, resonances and fault severities), extracts
features on the NPU, trains for 40 epochs, fits the novelty detector and
writes `model.safetensors`. Then it tests on machines it has never seen:

```text
simulated and featurised 2400 training windows in 3.8 s
trained in 1.9 s
novelty detector: 42 components explain 90.3% of the variance; covariance on the NPU (checked error 2.3e-3); ...
test set, all severities: 606 windows, accuracy 97.9%
test set, early faults (severity 0.2-0.4): 606 windows, accuracy 93.6%
   true \ pred      healthy    imbalance misalignment    looseness   outer-race   inner-race   recall
       healthy           99            0            1            1            0            0    98.0%
     ...
novelty (score > 1): 0.5% of known-condition windows, 46.5% of cavitation windows
  (a fault the classifier never saw; it called them: outer-race 2, inner-race 99)
```

Most classifier mistakes are early faults that look healthy, which is what
you'd expect. The last line shows why the novelty detector exists: shown
pump cavitation, which it never trained on, the classifier confidently says
"inner-race bearing fault".

## Novelty detection with `qanten::compute`

`src/novelty.rs` is a standard PCA process-monitoring model. It uses the
compute module the way its guide recommends:

| Step | Where it runs | Why |
|---|---|---|
| Covariance of the 2,400 × 92 training features, `ZᵀZ` | `compute::gemm(.., Placement::Auto)`: NPU | 20M multiply-accumulates, the device-held matrix reused for 92 vectors |
| Checking that product | `compute::check_product`, CPU | `O(n²)` Freivalds check; reports 2.3e-3 relative error |
| Eigenvectors of the 92 × 92 covariance | CPU (Jacobi) | a small matrix used once has no reuse, so it is CPU work |
| Scoring every window | one `NpuMatrix` on each NPU, `apply_batch` | the matrix stays on the device; every batch streams through it |
| Daily self-check | `NoveltyDetector::self_check` → `check_product` | watches the analog NPU's accuracy in production |

The resident matrix stacks two blocks, so one batched call gives both
classic statistics:

- `P = I − U·Uᵀ` gives the **SPE** (squared prediction error): the part of a
  feature vector *outside* the patterns seen in training. This catches new
  kinds of signal.
- `W = Λ^(−½)·Uᵀ` gives **Hotelling's T²**: how far *along* those patterns it
  is. This catches familiar patterns at unfamiliar strength, like
  cavitation's raised high-frequency noise floor.

Each statistic is divided by its 99.5th percentile on known conditions; the
score is the larger of the two, so above 1 means unusual. `NpuMatrix` scales
every row before upload, so the two blocks (rows near 1 and rows near
`1/√λ`) keep the same relative precision.

## 2. Monitor a plant

```sh
cargo run --release -p machine-monitor -- --sim-npus 4 monitor --machines 24 --hours 336
```

Simulates two weeks of a 24-machine plant. A third of the machines slowly
develop a fault (the hidden ground truth), and every fourth fault is
cavitation, which the classifier has never seen. Each hour the 24 windows
go to the NPUs as one batch, and alarms print as they fire:

```text
day  3 18:00  ALARM   conveyor-01    looseness     86%          -> check mounting bolts and fits
day  5 09:00  ALARM   pump-03        outer-race    47%          -> plan bearing replacement
day  5 13:00  UPDATE  pump-03        inner-race    74%          -> plan bearing replacement (urgent)
day  6 05:00  UPDATE  pump-03        unrecognised  novelty 1.1  -> unknown pattern: send an analyst / inspect on site
...
pump-03        cavitation from day  4 07:00     outer-race at day  5 09:00      18%   0%  detected (new fault type)
8/8 faults found with the right diagnosis, 0 with the wrong one, 0 missed, 0 false alarm(s)
daily NPU self-checks: worst relative error 0.33%
```

pump-03 shows the two models working together. The classifier raises a
bearing alarm early, at 18% fault severity, so someone looks at the
machine. As the cavitation grows, the novelty score stays high and the alarm
changes to "unrecognised", telling maintenance that this is not a standard
bearing job. A sustained high novelty score overrides the classifier,
because outside its training data its probabilities mean little.

The final table compares every alarm with the ground truth, including how
far each fault had developed when it was caught. Across other `--seed`
values every fault is still found. A few runs include one false alarm: a
healthy machine that the classifier, not the novelty detector, mistakes
for a fault for 3 hours in a row. Raise `CONFIRM_HOURS` in `src/fleet.rs`
to trade warning time for fewer false alarms.

## 3. Serve it

```sh
cargo run --release -p machine-monitor -- serve --addr 127.0.0.1:8080
```

This is `qanten-serve` used as a library. The whole pipeline (raw signal to
probabilities and novelty score) is one `Module`, so the server batches
concurrent requests and spreads them over the NPUs. The project adds its own
`/v1/conditions` route. Each request row is the shaft speed in Hz followed
by the 4096 samples. Each output row has 7 values: the six probabilities,
then the novelty score.

```sh
cargo run --release -p machine-monitor -- sample --condition cavitation --rpm 1480 > request.json
curl -s -X POST localhost:8080/v1/models/diagnose/infer \
     -H 'content-type: application/json' -d @request.json
# {"model":"diagnose","outputs":{"output":{"shape":[1,7],"data":[0,0,0,0,0,1,1.21]}},...}
#   classifier says inner-race; novelty 1.21 > 1 says "don't trust that" (healthy: 0.4)
curl -s localhost:8080/v1/conditions      # what each output index means + recommended action
curl -s localhost:8080/metrics            # Prometheus metrics
```

## Using real data

The simulator in `src/sim.rs` stands in for sensors. To use real
accelerometers:

- Resample to 10.24 kHz and send 4096-sample windows with the shaft speed
  (from a tachometer or the drive). The order resampling needs the speed.
- Retrain on labelled recordings: build a `Dataset` from your windows with
  `FeatureExtractor::extract_batch` and call `model::train`. Public bearing
  datasets (e.g. CWRU, Paderborn) are a good start; rescale `BPFO`/`BPFI` and
  the resonance band to your bearings.
- Tune `THRESHOLD`, `NOVELTY_THRESHOLD` and `CONFIRM_HOURS` in
  `src/fleet.rs` for your tolerance to false alarms.
- Refit the novelty detector on healthy data from your own plant. It learns
  what "normal" looks like there, including the known faults.

## Files

| File | |
|---|---|
| `src/sim.rs` | synthetic machines and fault signatures (including cavitation, unseen by the classifier), `Condition` |
| `src/features.rs` | the NPU feature pipeline |
| `src/model.rs` | dataset, training, normalisation, `Diagnoser` |
| `src/novelty.rs` | PCA novelty detector (SPE + T²) on `qanten::compute` |
| `src/fleet.rs` | simulated plant, `AlarmTracker` |
| `src/main.rs` | the `train`, `monitor`, `serve` and `sample` commands |
| `tests/pipeline.rs` | fault signatures, speed independence, train → save → load → serve path, novelty, alarm logic |
