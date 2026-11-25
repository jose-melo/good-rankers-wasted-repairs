#!/usr/bin/env bash
# Download every dataset and third-party file from its official source into $GRWR_DATA (default: ./data).
# Sizes: CATS 1.5 GB, Causal Chambers 0.3 GB, L-Town 0.2 GB. The causal graphs are simulated by the code.
set -euo pipefail
cd "$(dirname "$0")/.."
DATA="${GRWR_DATA:-$PWD/data}"
mkdir -p "$DATA"/{cats,causal_chambers/wt_meta,ltown,third_party}
get() { [ -s "$2" ] || curl -fL --retry 3 -o "$2" "$1"; }
check_md5() { python3 -c "import hashlib,sys;h=hashlib.md5(open(sys.argv[1],'rb').read()).hexdigest();sys.exit(h!=sys.argv[2] and print('md5 mismatch',sys.argv[1]) or 0)" "$1" "$2"; }

# CATS, Controlled Anomalies Time Series v2 (Solenix Engineering), Zenodo 10.5281/zenodo.8338435, CC BY 4.0
Z=https://zenodo.org/records/8338435/files
get "$Z/data.csv?download=1" "$DATA/cats/data.csv"
get "$Z/metadata.csv?download=1" "$DATA/cats/metadata.csv"
check_md5 "$DATA/cats/data.csv" 92050471389956f07603b83f7dbf6d0e
check_md5 "$DATA/cats/metadata.csv" cb52d1d73bf4bb54ebbfb466b94618de

# Causal Chambers (Gamella, Peters, Buhlmann 2025): light tunnel lt_walks_v1 and wind tunnel wt_walks_v1, CC BY 4.0
S=https://causalchamber.s3.eu-central-1.amazonaws.com/downloadables
for ds in lt_walks_v1:dcad019186661a56de7a2d1db97fc2f0 wt_walks_v1:19bb4e92cbe0b8dff49299b9b509ac36; do
  name=${ds%%:*}; sum=${ds##*:}
  get "$S/$name.zip" "$DATA/causal_chambers/$name.zip"
  check_md5 "$DATA/causal_chambers/$name.zip" "$sum"
  [ -d "$DATA/causal_chambers/$name" ] || unzip -q "$DATA/causal_chambers/$name.zip" -d "$DATA/causal_chambers"
done
# regime targets of the wind-tunnel experiments and the published graph (CC BY 4.0; the graph file is MIT)
get https://raw.githubusercontent.com/juangamella/causal-chamber/967d59a42956872f37463cc561ecf6e9ceab9364/datasets/wt_walks_v1/regimes_targets.py \
    "$DATA/causal_chambers/wt_meta/regimes_targets.py"
get https://raw.githubusercontent.com/juangamella/causal-chamber-package/087725404b9bad99df3160a2b3bd7da250f44fe3/causalchamber/ground_truth/main.py \
    "$DATA/causal_chambers/wt_meta/ground_truth_main.py"

# L-Town, BattLeDIM (Vrachimis et al. 2022), Zenodo 10.5281/zenodo.4017659, CC BY 4.0: network and sensor list
Z=https://zenodo.org/records/4017659/files
get "$Z/L-TOWN_Real.inp?download=1" "$DATA/ltown/L-TOWN_Real.inp"
get "$Z/dataset_configuration.yaml?download=1" "$DATA/ltown/dataset_configuration.yaml"
check_md5 "$DATA/ltown/L-TOWN_Real.inp" 40fb38de25ed692a5ddd30e0f5fae332
check_md5 "$DATA/ltown/dataset_configuration.yaml" 48486401f5b4d0447023f5ce5d242c52

# ARM (Peng et al., arXiv:2608.01691, CC BY 4.0): the authors' anc/arm_core.py, used unmodified
if [ ! -s "$DATA/third_party/arm_core.py" ]; then
  tmp=$(mktemp -d); curl -fL --retry 3 -o "$tmp/src.tar.gz" https://arxiv.org/src/2608.01691v1
  tar -xzf "$tmp/src.tar.gz" -C "$tmp" anc/arm_core.py && cp "$tmp/anc/arm_core.py" "$DATA/third_party/arm_core.py"
  rm -rf "$tmp"
fi
python3 -c "import hashlib,sys;h=hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest();assert h=='486c6589c593f73492e9414a415bc27ff5e4d91d5aac06c9d488f893cfcdefbd',h" \
  "$DATA/third_party/arm_core.py"
echo "data ready in $DATA"
