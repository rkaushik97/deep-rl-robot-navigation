#!/usr/bin/env bash
# Automated result collection: pull everything written to the shared drl-results PVC
# (every sweep job's models + eval result files) onto the host.
#
#   k8s/collect-results.sh [DEST_DIR]      # default DEST: ./sweep-results
#
# Works by spinning a tiny short-lived "collector" pod that mounts the PVC read-only, then
# kubectl-cp'ing the tree out. The collector is deleted afterwards.
set -euo pipefail
NS=drlnav
DEST=${1:-./sweep-results}
POD=drl-collector

cleanup() { kubectl -n "$NS" delete pod "$POD" --ignore-not-found --wait=false >/dev/null 2>&1 || true; }
trap cleanup EXIT

kubectl -n "$NS" delete pod "$POD" --ignore-not-found >/dev/null 2>&1 || true
cat <<YAML | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: ${POD}
  namespace: ${NS}
spec:
  restartPolicy: Never
  containers:
    - name: collector
      image: busybox:1.36
      command: ["sh", "-c", "sleep 3600"]
      volumeMounts:
        - { name: results, mountPath: /data, readOnly: true }
  volumes:
    - name: results
      persistentVolumeClaim: { claimName: drl-results }
YAML

echo "waiting for collector pod..."
kubectl -n "$NS" wait --for=condition=Ready pod/"$POD" --timeout=120s

mkdir -p "$DEST"
echo "copying PVC contents -> $DEST"
kubectl -n "$NS" cp "${POD}:/data" "$DEST"

echo "done. result summaries:"
find "$DEST" -path '*/results/*.txt' -print 2>/dev/null || true
