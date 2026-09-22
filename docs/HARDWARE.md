# Hardware integration and measured results

The repository supplies portable C projection encoding and integer linear classification, parameter/LUT export, an optional int8 TFLite model exporter, strict timing-trace replay, serial capture, and measurement analysis. It does not contain a board support package, TinyML runtime build, Android BLE app, edge-server deployment, radio driver, or a power-measurement instrument driver. Those cannot be reconstructed reliably from paper-level descriptions alone.

## Firmware integration

1. Train a model and run `scripts/export_tflite.py`. The generated `model_int8.tflite` and `model_data.cc` contain the CNN. Record the converter version and model hash.
2. Integrate with the selected device's supported inference runtime. Allocate and measure its tensor arena; verify the model's operators are supported. Flash size, RAM, cycles, stack, and energy must be measured on that target.
3. Use the exported affine input/output scales. Before the C packet encoder/head, requantize backbone features to the symmetric reference feature scale. The Python extraction script demonstrates this transformation. Calibration must match the deployed feature quantizer.
4. Export the classifier, margin, degree CDF, and exp table using `scripts/export_integer.py`. Build `embedded/rateless_codec.c` with `embedded/rateless_codec.h`. The C kernel has no heap allocation; its projection reservoir is bounded at 255 uint16 elements. Stack/ABI usage still depends on the toolchain.
5. Implement degree sampling, window IDs, packet emission, ACK handling, and transport fragmentation in the device application. Follow `PROTOCOL.md` and compare bytes against Python golden outputs.
6. Align board, receiver and power-instrument clocks with a measured synchronization procedure. A host serial timestamp is not a substitute for a radio arrival timestamp or a synchronized device event.

The Python simulator's integer head and softmax establish numerical reference behavior; the portable C contribution implements the encoder and linear head. It does not claim a complete C conformal controller/softmax/peeling firmware stack.

## Replay packet opportunities

CSV schema:

```csv
window_id,packet_id,tx_ms,arrival_ms,bytes
0,0,20,38,64
0,1,36.666667,,64
0,2,53.333333,72,64
```

Times are relative to window completion. Blank arrival means lost. Include the full available offered stream, not only packets emitted by a different ACK policy; otherwise alternative schedules cannot be inferred. Supply enough windows and opportunities for both the calibration and test replay streams. The current evaluator resets each role and reuses the supplied trace by role-local window ID. For an exact study, align those streams deliberately; this is not automatic inference of an unseen chronological trace. Frame sizes must match the evaluated configuration.

```bash
python -m rateless_har evaluate --run RUN --config CONFIG --method ours \
  --trace timing.csv --output outputs/replay
```

## Analyze synchronized current and prediction logs

Current CSV fields are `time_s,voltage_v,current_a`, in SI units, with strictly increasing times. Window CSV fields are `window_id,ready_s,prediction_s,deadline_ms,covered,set_size`. `ready_s` means acquisition is complete; a blank prediction time means no prediction. `covered` is whether the true label belongs to the actually produced set.

```bash
python -m pip install -e '.[hardware]'
python scripts/capture_serial.py --port /dev/ttyACM0 --seconds 60 --output outputs/board.jsonl
python scripts/analyze_hardware.py --windows windows.csv --current current.csv \
  --output outputs/measured
```

The analyzer integrates voltage×current using trapezoidal integration with interpolated interval endpoints. Missing trace coverage is an error. Failed predictions integrate until the deadline; late predictions integrate until the actual prediction, so late energy is retained. Measured joint success includes both coverage and deadline; add a device-specific energy-cap definition if needed. Do not double-count overlapping energy intervals when aggregating continuous experiments.

Record supply voltage, sampling rate, baseline subtraction policy, instrument uncertainty, actual radio settings, clock offset/drift, device temperature, battery state, compiler optimization, runtime versions, measured quantization agreement, and firmware/model hashes with published measurements. The default simulation energy coefficients and timing scenarios are not a substitute for these logs.
