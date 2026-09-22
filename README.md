# SLA-aware rateless uncertainty packetization for TinyML HAR

A documented **reference implementation** of the method described in *SLA-Aware Rateless Uncertainty Packetization for TinyML-Based Human Activity Recognition at the IoT Edge*.

This repository implements data preparation, quantization-aware CNN training, integer classification, sparse signed rateless packet encoding, peeling, conformal calibration, causal SLA control, link simulation/replay, baselines, ablations, sensitivity studies, plots, and measurement-log analysis. It includes a portable C encoder/head and an optional all-integer TFLite exporter.
