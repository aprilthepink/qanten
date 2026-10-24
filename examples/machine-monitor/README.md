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
        ▼  host: resample spectra from Hz to shaft orders, add RMS / crest / kurtosis
        ▼  NPU: MLP 90 → 64 → 32 → 6
condition probabilities: healthy, imbalance, misalignment, looseness, outer-race, inner-race
        │
        ▼  alarm logic: smoothing + 3 confirming hours → maintenance alarm
```

It uses these parts of the library:

| Part | Used for |
|---|---|
| `qanten::signal` (`FirBank`, `Dft`, `hann`, `design`) | the vibration-analysis front end |
| `qanten::train` (`Dense`, `Adam`, `cross_entropy`) | training the classifier with the NPU in the loop |
| `qanten::nn` (`Linear`, `Sequential`, `Module`, `StateDict`) | the deployed model, saved as `.safetensors` |
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
features on the NPU, trains for 40 epochs and writes `model.safetensors`.
Then it tests the model on machines it has never seen:

```text
simulated and featurised 2400 training windows in 9.6 s
trained in 2.1 s
test set, all severities: 606 windows, accuracy 97.7%
test set, early faults (severity 0.2-0.4): 606 windows, accuracy 93.7%
   true \ pred      healthy    imbalance misalignment    looseness   outer-race   inner-race   recall
       healthy           99            0            1            1            0            0    98.0%
     ...
```

Most mistakes are early faults that look healthy, which is what you'd expect.

## 2. Monitor a plant

```sh
cargo run --release -p machine-monitor -- --sim-npus 4 monitor --machines 24 --hours 336
```

Simulates two weeks of a 24-machine plant. A third of the machines slowly
develop a fault (the hidden ground truth). Each hour the 24 windows go to the
NPUs as one batch, and alarms print as they fire:

```text
day  3 20:00  ALARM   conveyor-01    looseness      87%  -> check mounting bolts and fits
day  5 22:00  ALARM   pump-03        outer-race     86%  -> plan bearing replacement
day  9 06:00  ALARM   pump-01        inner-race     76%  -> plan bearing replacement (urgent)
day  9 13:00  UPDATE  pump-01        now outer-race  66%  -> plan bearing replacement
...
8/8 faults found with the right diagnosis, 0 with the wrong one, 0 missed, 0 false alarm(s)
```

At the end a table compares each alarm with the ground truth, including how
far the fault had developed when it was caught (13–22% in this run). The
per-NPU profile shows the work split evenly across the pool.

## 3. Serve it

```sh
cargo run --release -p machine-monitor -- serve --addr 127.0.0.1:8080
```

This is `qanten-serve` used as a library. The whole pipeline (raw signal to
probabilities) is one `Module`, so the server batches concurrent requests and
spreads them over the NPUs. The project adds its own `/v1/conditions` route.
Each request row is the shaft speed in Hz followed by the 4096 samples:

```sh
cargo run --release -p machine-monitor -- sample --condition inner-race --rpm 1480 > request.json
curl -s -X POST localhost:8080/v1/models/diagnose/infer \
     -H 'content-type: application/json' -d @request.json
# {"model":"diagnose","outputs":{"output":{"shape":[1,6],"data":[0,0,0,0,0,1]}},...}
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
- Tune `THRESHOLD` and `CONFIRM_HOURS` in `src/fleet.rs` for your tolerance to
  false alarms.

## Files

| File | |
|---|---|
| `src/sim.rs` | synthetic machines and fault signatures, `Condition` |
| `src/features.rs` | the NPU feature pipeline |
| `src/model.rs` | dataset, training, normalisation, `Diagnoser` |
| `src/fleet.rs` | simulated plant, `AlarmTracker` |
| `src/main.rs` | the `train`, `monitor`, `serve` and `sample` commands |
| `tests/pipeline.rs` | fault signatures, speed independence, train → save → load → serve path, alarm logic |
